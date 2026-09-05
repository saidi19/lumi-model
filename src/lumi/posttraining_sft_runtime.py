"""Assistant-only Lumi SFT dataset runtime and resumable CPU correctness canary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .modeling_lumi import LumiForCausalLM
from .pilot_training import _create_optimizer, scheduled_learning_rate
from .posttraining_shards import verify_sft_shards

SFT_CANARY_STATE_SCHEMA_VERSION = "lumi.sft-cpu-canary-state.v0.1"
SFT_CANARY_CHECKPOINT_SCHEMA_VERSION = "lumi.sft-cpu-canary-checkpoint.v0.1"
SFT_CANARY_STATE_FILENAME = "sft-canary-state.json"
SFT_CANARY_METRICS_FILENAME = "sft-canary-metrics.jsonl"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _write_checksum_manifest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            entries.append((_sha256_file(path), path.relative_to(root).as_posix()))
    if not entries:
        raise RuntimeError("refusing to publish an empty SFT checkpoint")
    path = root / "manifest.sha256"
    with path.open("x", encoding="utf-8") as handle:
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_file(path)


def _verify_checksum_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    require_exact_inventory: bool = True,
) -> str:
    manifest = root / "manifest.sha256"
    if not root.is_dir() or not manifest.is_file():
        raise ValueError(f"checkpoint checksum manifest is missing: {root}")
    observed_manifest_sha256 = _sha256_file(manifest)
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("checkpoint manifest SHA-256 changed")
    declared = {"manifest.sha256"}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint manifest line {line_number}") from error
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid checkpoint digest at line {line_number}")
        relative_path = Path(relative)
        path = root / relative_path
        if (
            relative_path.is_absolute()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
        ):
            raise ValueError(f"checkpoint manifest path is invalid: {relative}")
        if _sha256_file(path) != digest:
            raise ValueError(f"checkpoint file hash changed: {relative}")
        declared.add(relative_path.as_posix())
    if len(declared) < 3:
        raise ValueError("checkpoint manifest is unexpectedly small")
    if require_exact_inventory:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual != declared:
            raise ValueError("checkpoint contains undeclared files")
    return observed_manifest_sha256


@dataclass(frozen=True)
class SFTTensorExample:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    supervised_tokens: int


@dataclass(frozen=True)
class _RecordLocation:
    labels_path: Path
    offset_tokens: int
    supervised_tokens: int
    token_count: int
    tokens_path: Path


class EncodedSFTDataset(torch.utils.data.Dataset[SFTTensorExample]):
    """Memory-map one verified split from Lumi's raw-text-free SFT shards."""

    def __init__(self, root: str | Path, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("SFT dataset split must be train or validation")
        self.root = Path(root)
        self.manifest = verify_sft_shards(self.root)
        self.split = split
        self._locations: list[_RecordLocation] = []
        self._buffers: dict[Path, torch.Tensor] = {}

        for shard in self.manifest["shards"]:
            if shard["split"] != split:
                continue
            prefix = f"{split}-{shard['index']:05d}"
            tokens_path = self.root / f"{prefix}.tokens.u32le"
            labels_path = self.root / f"{prefix}.labels.i32le"
            self._buffers[tokens_path] = torch.from_file(
                str(tokens_path),
                shared=False,
                size=tokens_path.stat().st_size // 4,
                dtype=torch.int32,
            )
            self._buffers[labels_path] = torch.from_file(
                str(labels_path),
                shared=False,
                size=labels_path.stat().st_size // 4,
                dtype=torch.int32,
            )
            index_path = self.root / f"{prefix}.index.jsonl"
            for line in index_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self._locations.append(
                    _RecordLocation(
                        labels_path=labels_path,
                        offset_tokens=entry["offset_tokens"],
                        supervised_tokens=entry["supervised_tokens"],
                        token_count=entry["tokens"],
                        tokens_path=tokens_path,
                    )
                )
        expected_records = self.manifest.get("splits", {}).get(split, {}).get("records", 0)
        if len(self._locations) != expected_records:
            raise ValueError("SFT split record inventory changed")
        if not self._locations:
            raise ValueError(f"SFT split is empty: {split}")

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int) -> SFTTensorExample:
        location = self._locations[index]
        start = location.offset_tokens
        end = start + location.token_count
        input_ids = self._buffers[location.tokens_path][start:end].to(torch.long)
        labels = self._buffers[location.labels_path][start:end].to(torch.long)
        if input_ids.shape != labels.shape or input_ids.ndim != 1:
            raise ValueError("SFT encoded input and label shapes changed")
        supervised = labels.ne(-100)
        if int(supervised.sum()) != location.supervised_tokens:
            raise ValueError("SFT encoded supervision count changed")
        if not torch.equal(labels[supervised], input_ids[supervised]):
            raise ValueError("SFT unmasked labels no longer equal input tokens")
        if not bool(labels.eq(-100).any()):
            raise ValueError("SFT prompt labels are not masked")
        return SFTTensorExample(
            input_ids=input_ids,
            labels=labels,
            supervised_tokens=location.supervised_tokens,
        )

    @property
    def maximum_record_tokens(self) -> int:
        return max(location.token_count for location in self._locations)


def collate_sft_tensors(
    examples: Sequence[SFTTensorExample],
    *,
    context_window: int,
    ignore_index: int = -100,
    pad_to_multiple_of: int = 8,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Right-pad isolated examples while preserving assistant-only supervision."""

    if not examples:
        raise ValueError("SFT batch must contain at least one example")
    if pad_to_multiple_of < 1 or context_window < 1:
        raise ValueError("SFT padding and context limits must be positive")
    maximum = max(example.input_ids.numel() for example in examples)
    padded = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    if padded > context_window:
        raise ValueError("SFT batch exceeds the configured context window")
    input_ids = torch.full((len(examples), padded), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), padded), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), padded), dtype=torch.long)
    for row, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = 1
    if not bool(labels.ne(ignore_index).any()):
        raise ValueError("SFT batch contains no supervised tokens")
    return {
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "labels": labels,
    }


class SFTCursorPlanner:
    """Deterministic epoch shuffle whose global cursor is checkpoint sufficient."""

    def __init__(self, *, dataset_length: int, seed: int) -> None:
        if dataset_length < 1:
            raise ValueError("SFT dataset length must be positive")
        self.dataset_length = dataset_length
        self.seed = seed
        self._epoch: int | None = None
        self._permutation: torch.Tensor | None = None

    def _for_epoch(self, epoch: int) -> torch.Tensor:
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(
                self.dataset_length,
                generator=generator,
                dtype=torch.int64,
            )
            self._epoch = epoch
        assert self._permutation is not None
        return self._permutation

    def indices(self, *, cursor: int, count: int) -> tuple[list[int], int]:
        if cursor < 0 or count < 1:
            raise ValueError("SFT cursor must be non-negative and count positive")
        indices: list[int] = []
        next_cursor = cursor
        while len(indices) < count:
            epoch, offset = divmod(next_cursor, self.dataset_length)
            permutation = self._for_epoch(epoch)
            take = min(count - len(indices), self.dataset_length - offset)
            indices.extend(permutation[offset : offset + take].tolist())
            next_cursor += take
        return indices, next_cursor


def verify_sft_canary_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    _verify_checksum_manifest(checkpoint)
    state_path = checkpoint / SFT_CANARY_STATE_FILENAME
    if not state_path.is_file():
        raise ValueError("SFT canary checkpoint state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("checkpoint_schema_version") != SFT_CANARY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported SFT canary checkpoint schema")
    return state


def _discover_latest_checkpoint(
    output: Path,
    *,
    run_fingerprint: str,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        state = verify_sft_canary_checkpoint(path)
        if state.get("run_fingerprint") == run_fingerprint:
            candidates.append((int(state["step"]), path, state))
    if not candidates:
        return None
    _, path, state = max(candidates, key=lambda item: item[0])
    return path, state


def _save_checkpoint(
    *,
    output: Path,
    model: LumiForCausalLM,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
) -> Path:
    name = f"checkpoint-{state['step']:08d}"
    checkpoint = output / name
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite SFT checkpoint: {checkpoint}")
    temporary = output / f".{name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
            },
            temporary / "trainer.pt",
        )
        checkpoint_state = {
            **state,
            "checkpoint_schema_version": SFT_CANARY_CHECKPOINT_SCHEMA_VERSION,
            "latest_checkpoint": name,
            "status": "checkpoint",
        }
        _write_json_atomic(temporary / SFT_CANARY_STATE_FILENAME, checkpoint_state)
        _write_checksum_manifest(temporary)
        _fsync_tree(temporary)
        temporary.replace(checkpoint)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_sft_canary_checkpoint(checkpoint)
    return checkpoint


def _restore_checkpoint(
    checkpoint: Path,
    *,
    learning_rate: float,
    optimizer_profile: str,
    weight_decay: float,
) -> tuple[LumiForCausalLM, torch.optim.Optimizer]:
    verify_sft_canary_checkpoint(checkpoint)
    model = LumiForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    model.config._attn_implementation = "eager"
    model.to(torch.device("cpu"))
    optimizer = _create_optimizer(
        model,
        device=torch.device("cpu"),
        learning_rate=learning_rate,
        optimizer_profile=optimizer_profile,
        weight_decay=weight_decay,
    )
    payload = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng_state"])
    return model, optimizer


def _trim_metrics(path: Path, *, committed_step: int) -> None:
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if int(record["step"]) <= committed_step:
                records.append(record)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_sft_cpu_canary(
    *,
    encoded_shards: str | Path,
    source_checkpoint: str | Path,
    source_checkpoint_manifest_sha256: str,
    output: str | Path,
    target_steps: int,
    stop_after_step: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    maximum_gradient_norm: float = 1.0,
    warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.1,
    optimizer_profile: str = "grouped_adamw_beta2_0.95_v0.1",
    pad_to_multiple_of: int = 8,
    checkpoint_steps: int = 1,
    seed: int = 2026,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a tiny CPU-only masked-loss canary; this cannot authorize paid SFT."""

    if target_steps < 2 or not 1 <= stop_after_step <= target_steps:
        raise ValueError("SFT CPU canary requires at least two target steps")
    if batch_size < 1 or gradient_accumulation_steps < 1 or checkpoint_steps < 1:
        raise ValueError("SFT CPU canary batch and checkpoint settings must be positive")
    if learning_rate <= 0 or weight_decay < 0 or maximum_gradient_norm <= 0:
        raise ValueError("SFT CPU canary optimizer settings are invalid")
    if warmup_steps < 0 or warmup_steps >= target_steps:
        raise ValueError("SFT CPU canary warmup does not fit its schedule")
    if len(source_checkpoint_manifest_sha256) != 64:
        raise ValueError("SFT source checkpoint manifest SHA-256 is invalid")

    encoded_shards = Path(encoded_shards)
    source_checkpoint = Path(source_checkpoint)
    output = Path(output)
    source_manifest_sha256 = _verify_checksum_manifest(
        source_checkpoint,
        expected_manifest_sha256=source_checkpoint_manifest_sha256,
    )
    shard_manifest = verify_sft_shards(encoded_shards)
    dataset = EncodedSFTDataset(encoded_shards, split="train")
    run_identity = {
        "data": {
            "manifest_sha256": shard_manifest["manifest_sha256"],
            "seal_sha256": shard_manifest["seal_sha256"],
            "sequence_order": "seeded_epoch_permutation_v0.1",
            "split": "train",
        },
        "device": "cpu",
        "objective": "assistant_and_tool_only_cross_entropy_v0.1",
        "optimizer": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "maximum_gradient_norm": maximum_gradient_norm,
            "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
            "optimizer_profile": optimizer_profile,
            "schedule": "cosine",
            "target_steps": target_steps,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
        },
        "schema_version": SFT_CANARY_STATE_SCHEMA_VERSION,
        "seed": seed,
        "source_checkpoint_manifest_sha256": source_manifest_sha256,
        "torch_version": torch.__version__,
    }
    run_fingerprint = _sha256(run_identity)
    state_path = output / SFT_CANARY_STATE_FILENAME
    metrics_path = output / SFT_CANARY_METRICS_FILENAME

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"SFT canary state exists; pass resume=True: {output}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("SFT canary state does not match the requested run")
    else:
        if resume:
            raise ValueError("cannot resume an SFT canary without state")
        if any(output.iterdir()):
            raise FileExistsError(f"SFT canary output is not empty: {output}")
        state = {
            "best_loss": None,
            "last_loss": None,
            "latest_checkpoint": None,
            "records_seen": 0,
            "run_fingerprint": run_fingerprint,
            "run_specification": run_identity,
            "schema_version": SFT_CANARY_STATE_SCHEMA_VERSION,
            "status": "active",
            "step": 0,
            "supervised_tokens_seen": 0,
            "tokens_seen": 0,
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
    _trim_metrics(metrics_path, committed_step=int(state["step"]))

    if state["latest_checkpoint"] is None:
        torch.manual_seed(seed)
        model = LumiForCausalLM.from_pretrained(source_checkpoint, local_files_only=True)
        model.config._attn_implementation = "eager"
        model.to(torch.device("cpu"))
        optimizer = _create_optimizer(
            model,
            device=torch.device("cpu"),
            learning_rate=learning_rate,
            optimizer_profile=optimizer_profile,
            weight_decay=weight_decay,
        )
    else:
        model, optimizer = _restore_checkpoint(
            output / state["latest_checkpoint"],
            learning_rate=learning_rate,
            optimizer_profile=optimizer_profile,
            weight_decay=weight_decay,
        )
    if dataset.maximum_record_tokens > model.config.max_position_embeddings:
        raise ValueError("SFT canary record exceeds the source model context window")

    planner = SFTCursorPlanner(dataset_length=len(dataset), seed=seed)
    model.train()
    while state["step"] < stop_after_step:
        learning_rate_at_step = scheduled_learning_rate(
            base_learning_rate=learning_rate,
            step=state["step"] + 1,
            schedule="cosine",
            warmup_steps=warmup_steps,
            schedule_steps=target_steps,
            min_learning_rate_ratio=minimum_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_step
        optimizer.zero_grad(set_to_none=True)
        cursor = int(state["records_seen"])
        microbatches = []
        step_tokens = 0
        step_supervised_tokens = 0
        for _ in range(gradient_accumulation_steps):
            indices, cursor = planner.indices(cursor=cursor, count=batch_size)
            batch = collate_sft_tensors(
                [dataset[index] for index in indices],
                context_window=model.config.max_position_embeddings,
                pad_to_multiple_of=pad_to_multiple_of,
                pad_token_id=model.config.pad_token_id,
            )
            supervised = int(batch["labels"][:, 1:].ne(-100).sum())
            if supervised < 1:
                raise ValueError("SFT canary microbatch has no shifted supervised labels")
            microbatches.append((batch, supervised))
            step_tokens += int(batch["attention_mask"].sum())
            step_supervised_tokens += supervised

        started = time.perf_counter()
        weighted_loss = 0.0
        for batch, supervised in microbatches:
            output_value = model(**batch)
            loss = output_value.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("SFT CPU canary produced a non-finite loss")
            weight = supervised / step_supervised_tokens
            (loss * weight).backward()
            weighted_loss += float(loss.detach()) * weight
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            maximum_gradient_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("SFT CPU canary produced a non-finite gradient")
        optimizer.step()
        elapsed = max(time.perf_counter() - started, 1e-9)

        state["step"] += 1
        state["records_seen"] = cursor
        state["tokens_seen"] += step_tokens
        state["supervised_tokens_seen"] += step_supervised_tokens
        state["last_loss"] = weighted_loss
        state["best_loss"] = (
            weighted_loss if state["best_loss"] is None else min(state["best_loss"], weighted_loss)
        )
        metric = {
            "elapsed_seconds": elapsed,
            "gradient_norm": float(gradient_norm),
            "learning_rate": learning_rate_at_step,
            "loss": weighted_loss,
            "step": state["step"],
            "supervised_tokens": step_supervised_tokens,
            "tokens": step_tokens,
            "tokens_per_second": step_tokens / elapsed,
        }
        should_checkpoint = (
            state["step"] % checkpoint_steps == 0 or state["step"] == stop_after_step
        )
        if should_checkpoint:
            checkpoint = _save_checkpoint(
                output=output,
                model=model,
                optimizer=optimizer,
                state=state,
            )
            state["latest_checkpoint"] = checkpoint.name
        _append_jsonl(metrics_path, metric)
        _write_json_atomic(state_path, state)

    state["status"] = "complete" if state["step"] == target_steps else "paused"
    _write_json_atomic(state_path, state)
    return state
