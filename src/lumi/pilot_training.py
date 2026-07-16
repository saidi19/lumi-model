"""Correctness-first, resumable single-device training for Lumi pilot runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .configuration_lumi import LumiConfig
from .modeling_lumi import LumiForCausalLM
from .packed_data import PackedTokenDataset, validate_packed_manifest

TRAINING_SCHEMA_VERSION = 1
TRAINING_STATE_FILENAME = "training-state.json"
METRICS_FILENAME = "metrics.jsonl"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _create_optimizer(
    model: LumiForCausalLM,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )


def _batch_indices(
    *, cursor: int, count: int, dataset_length: int, seed: int
) -> list[int]:
    """Return a stateless epoch-shuffled slice and its cross-epoch continuation."""

    indices: list[int] = []
    while len(indices) < count:
        epoch, epoch_offset = divmod(cursor, dataset_length)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        permutation = torch.randperm(dataset_length, generator=generator).tolist()
        take = min(count - len(indices), dataset_length - epoch_offset)
        indices.extend(permutation[epoch_offset : epoch_offset + take])
        cursor += take
    return indices


def _trim_metrics(path: Path, committed_step: int) -> None:
    if not path.exists():
        path.touch()
        return
    retained: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if any(remaining.strip() for remaining in handle):
                    raise ValueError(
                        f"metrics contain invalid JSON before EOF at line {line_number}"
                    ) from error
                break
            if int(record["step"]) <= committed_step:
                retained.append(record)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_state(path: Path) -> dict[str, Any]:
    with (path / TRAINING_STATE_FILENAME).open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint state in {path}")
    return state


def _discover_latest_checkpoint(
    output: Path, *, run_fingerprint: str
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir() or not (path / TRAINING_STATE_FILENAME).is_file():
            continue
        checkpoint_state = _checkpoint_state(path)
        if checkpoint_state.get("run_fingerprint") != run_fingerprint:
            continue
        step = checkpoint_state.get("step")
        if isinstance(step, int) and step >= 0:
            candidates.append((step, path, checkpoint_state))
    if not candidates:
        return None
    _, path, state = max(candidates, key=lambda item: item[0])
    return path, state


def _save_checkpoint(
    *,
    output: Path,
    model: LumiForCausalLM,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> Path:
    checkpoint_name = f"checkpoint-{state['step']:08d}"
    checkpoint = output / checkpoint_name
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    temporary = output / f".{checkpoint_name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        model_device = next(model.parameters()).device
        rng_state = {"cpu": torch.get_rng_state()}
        if model_device.type == "cuda":
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        elif model_device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            rng_state["mps"] = torch.mps.get_rng_state()
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "rng_state": rng_state,
            },
            temporary / "trainer.pt",
        )
        checkpoint_state = {
            **state,
            "latest_checkpoint": checkpoint_name,
            "status": "checkpoint",
            "updated_at": _utc_now(),
        }
        _write_json_atomic(temporary / TRAINING_STATE_FILENAME, checkpoint_state)
        _fsync_tree(temporary)
        temporary.replace(checkpoint)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint


def _restore_checkpoint(
    *, checkpoint: Path, device: torch.device, learning_rate: float, weight_decay: float
) -> tuple[LumiForCausalLM, torch.optim.AdamW]:
    model = LumiForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    model.to(device)
    optimizer = _create_optimizer(
        model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trainer_payload = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=True
    )
    optimizer.load_state_dict(trainer_payload["optimizer"])
    rng_state = trainer_payload["rng_state"]
    torch.set_rng_state(rng_state["cpu"])
    if device.type == "cuda" and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    elif device.type == "mps" and "mps" in rng_state:
        torch.mps.set_rng_state(rng_state["mps"])
    return model, optimizer


def train_packed_pilot(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    output: str | Path,
    steps: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    checkpoint_steps: int = 10,
    seed: int = 7,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train a single-device Lumi pilot with exact data-position resume."""

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output = Path(output)
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if checkpoint_steps < 1:
        raise ValueError("checkpoint-steps must be positive")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("optimizer hyperparameters are invalid")
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")

    manifest = validate_packed_manifest(manifest_path)
    config_bytes = config_path.read_bytes()
    config = LumiConfig(**json.loads(config_bytes))
    sequence_length = manifest["packing"]["sequence_length"]
    if config.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("model and packed dataset vocabulary sizes differ")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("packed sequence length exceeds the model maximum")

    manifest_bytes = manifest_path.read_bytes()
    run_identity = {
        "config": {
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "data": {
            "build_fingerprint": manifest["build_fingerprint"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "optimizer": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "weight_decay": weight_decay,
        },
        "schema_version": TRAINING_SCHEMA_VERSION,
        "seed": seed,
    }
    run_specification = {
        **run_identity,
        "locations": {
            "config_path": str(config_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        },
    }
    run_fingerprint = _fingerprint(run_identity)
    state_path = output / TRAINING_STATE_FILENAME
    metrics_path = output / METRICS_FILENAME

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"training output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"training state exists; pass --resume: {state_path}")
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("training state does not match the requested run")
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"training output is not empty: {output}")
        state = {
            "best_loss": None,
            "created_at": _utc_now(),
            "last_loss": None,
            "latest_checkpoint": None,
            "run_fingerprint": run_fingerprint,
            "run_specification": run_specification,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "sequence_cursor": 0,
            "status": "active",
            "step": 0,
            "tokens_seen": 0,
            "updated_at": _utc_now(),
        }
        _write_json_atomic(state_path, state)

    recovered = _discover_latest_checkpoint(output, run_fingerprint=run_fingerprint)
    if recovered is not None:
        checkpoint, checkpoint_state = recovered
        if checkpoint_state["step"] >= state["step"]:
            state = {
                **state,
                **checkpoint_state,
                "latest_checkpoint": checkpoint.name,
                "status": "active",
            }
    _trim_metrics(metrics_path, state["step"])

    training_device = _resolve_device(device)
    if state["latest_checkpoint"] is None:
        torch.manual_seed(seed)
        model = LumiForCausalLM(config)
        model.to(training_device)
        optimizer = _create_optimizer(
            model,
            device=training_device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        model, optimizer = _restore_checkpoint(
            checkpoint=output / state["latest_checkpoint"],
            device=training_device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    dataset = PackedTokenDataset(manifest_path, verify_hashes=False)
    if len(dataset) == 0:
        raise ValueError("packed dataset contains no sequences")
    model.train()

    while state["step"] < steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        step_tokens = 0
        _synchronize(training_device)
        if training_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(training_device)
        started = time.perf_counter()
        for _ in range(gradient_accumulation_steps):
            indices = _batch_indices(
                cursor=state["sequence_cursor"],
                count=batch_size,
                dataset_length=len(dataset),
                seed=seed,
            )
            batch = torch.stack([dataset[index] for index in indices]).to(training_device)
            with _autocast_context(training_device):
                loss = model(input_ids=batch, labels=batch).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss before step {state['step'] + 1}: {loss}")
            (loss / gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach())
            state["sequence_cursor"] += batch_size
            step_tokens += batch.numel()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(
                f"non-finite gradient norm before step {state['step'] + 1}: {gradient_norm}"
            )
        optimizer.step()
        _synchronize(training_device)
        elapsed = max(time.perf_counter() - started, 1e-9)

        state["step"] += 1
        state["tokens_seen"] += step_tokens
        state["last_loss"] = accumulated_loss / gradient_accumulation_steps
        state["best_loss"] = (
            state["last_loss"]
            if state["best_loss"] is None
            else min(state["best_loss"], state["last_loss"])
        )
        state["updated_at"] = _utc_now()
        metric = {
            "device": training_device.type,
            "elapsed_seconds": elapsed,
            "gradient_norm": float(gradient_norm),
            "loss": state["last_loss"],
            "peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(training_device)
                if training_device.type == "cuda"
                else None
            ),
            "peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(training_device)
                if training_device.type == "cuda"
                else None
            ),
            "step": state["step"],
            "tokens_per_second": step_tokens / elapsed,
            "tokens_seen": state["tokens_seen"],
        }
        print(
            f"step={state['step']:06d} loss={state['last_loss']:.6f} "
            f"tokens_per_second={metric['tokens_per_second']:.1f}",
            flush=True,
        )

        should_checkpoint = state["step"] % checkpoint_steps == 0 or state["step"] == steps
        if should_checkpoint:
            _synchronize(training_device)
            checkpoint_started = time.perf_counter()
            checkpoint = _save_checkpoint(
                output=output, model=model, optimizer=optimizer, state=state
            )
            _synchronize(training_device)
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            state["latest_checkpoint"] = checkpoint.name
            state["last_checkpoint_seconds"] = checkpoint_seconds
            state["status"] = "active"
            metric["checkpoint_seconds"] = checkpoint_seconds
            print(
                f"checkpoint={checkpoint.name} seconds={checkpoint_seconds:.3f}",
                flush=True,
            )
        else:
            metric["checkpoint_seconds"] = None
        _append_metric(metrics_path, metric)
        if should_checkpoint:
            _write_json_atomic(state_path, state)

    state["status"] = "complete"
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    return state
