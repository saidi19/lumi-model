"""Resumable loss, generation, speed, and memory evaluation for Lumi checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .evaluation_data import validate_heldout_manifest
from .modeling_lumi import LumiForCausalLM
from .packed_data import PackedTokenDataset

EVALUATION_SCHEMA_VERSION = 1
EVALUATION_STATE_FILENAME = "evaluation-state.json"
EVALUATION_REPORT_FILENAME = "evaluation-report.json"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "bytes": path.stat().st_size,
        "name": path.name,
        "sha256": _sha256_file(path),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _checkpoint_artifact(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise NotADirectoryError(path)
    config = path / "config.json"
    weight_files = sorted(path.glob("model*.safetensors"))
    if not weight_files and (path / "pytorch_model.bin").is_file():
        weight_files = [path / "pytorch_model.bin"]
    if not weight_files:
        raise FileNotFoundError(f"checkpoint has no model weights: {path}")
    index_files = [
        candidate
        for candidate in (
            path / "model.safetensors.index.json",
            path / "pytorch_model.bin.index.json",
        )
        if candidate.is_file()
    ]
    return {
        "files": [
            _file_artifact(config),
            *map(_file_artifact, index_files),
            *map(_file_artifact, weight_files),
        ]
    }


def _tokenizer_artifact(path: Path, expected_files: list[dict[str, Any]]) -> dict[str, Any]:
    files = [_file_artifact(path / item["name"]) for item in expected_files]
    expected = [
        {"bytes": item["bytes"], "name": item["name"], "sha256": item["sha256"]}
        for item in expected_files
    ]
    if files != expected:
        raise ValueError("evaluation tokenizer does not match the held-out artifact")
    return {"files": files}


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _device_memory(device: torch.device) -> dict[str, int | None]:
    if device.type == "cuda":
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "driver_bytes": torch.cuda.memory_reserved(device),
        }
    if device.type == "mps":
        return {
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
    return {"allocated_bytes": None, "driver_bytes": None}


def _merge_memory_peak(
    peak: dict[str, int | None], current: dict[str, int | None]
) -> dict[str, int | None]:
    return {
        key: (
            None
            if current[key] is None and peak[key] is None
            else max(current[key] or 0, peak[key] or 0)
        )
        for key in peak
    }


def _peak_rss_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum if sys.platform == "darwin" else maximum * 1024


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < math.log(sys.float_info.max) else math.inf


def _stratum_at_cursor(strata: list[dict[str, Any]], cursor: int) -> dict[str, Any]:
    for item in strata:
        if item["sequence_start"] <= cursor < item["sequence_end"]:
            return item
    raise RuntimeError(f"evaluation cursor {cursor} is outside held-out source ranges")


def _initial_state(
    *, fingerprint: str, strata: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    return {
        "batches": 0,
        "created_at": _utc_now(),
        "device": device.type,
        "device_memory_peak_observed": {
            "allocated_bytes": None,
            "driver_bytes": None,
        },
        "evaluation_fingerprint": fingerprint,
        "forward_seconds": 0.0,
        "input_tokens": 0,
        "prediction_tokens": 0,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "sequence_cursor": 0,
        "status": "active",
        "strata": {
            item["id"]: {
                "input_tokens": 0,
                "prediction_tokens": 0,
                "sequences": 0,
                "total_negative_log_likelihood": 0.0,
            }
            for item in strata
        },
        "total_negative_log_likelihood": 0.0,
        "updated_at": _utc_now(),
    }


def _validate_state(
    state: dict[str, Any], *, fingerprint: str, strata: list[dict[str, Any]], total: int
) -> None:
    if state.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation checkpoint schema")
    if state.get("evaluation_fingerprint") != fingerprint:
        raise ValueError("evaluation checkpoint does not match the requested run")
    if state.get("status") != "active":
        raise ValueError("evaluation checkpoint is not active")
    cursor = state.get("sequence_cursor")
    if not isinstance(cursor, int) or not 0 <= cursor <= total:
        raise ValueError("evaluation checkpoint has an invalid sequence cursor")
    if set(state.get("strata", {})) != {item["id"] for item in strata}:
        raise ValueError("evaluation checkpoint has invalid source strata")


def _validate_report(
    report: dict[str, Any],
    *,
    fingerprint: str,
    heldout_manifest: dict[str, Any],
    prompts: list[str],
) -> dict[str, Any]:
    if report.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation report schema")
    if report.get("status") != "complete":
        raise ValueError("cached evaluation report is not complete")
    if report.get("evaluation_fingerprint") != fingerprint:
        raise ValueError("cached evaluation report does not match the requested run")

    sequence_length = heldout_manifest["packing"]["sequence_length"]
    expected_sequences = heldout_manifest["statistics"]["sequences"]
    expected_input_tokens = heldout_manifest["statistics"]["tokens"]
    expected_prediction_tokens = expected_sequences * (sequence_length - 1)
    runtime = report["runtime"]
    loss = report["loss"]
    if runtime.get("sequences") != expected_sequences:
        raise ValueError("evaluation report sequence total changed")
    if runtime.get("input_tokens") != expected_input_tokens:
        raise ValueError("evaluation report input-token total changed")
    if loss.get("prediction_tokens") != expected_prediction_tokens:
        raise ValueError("evaluation report prediction-token total changed")
    recomputed_loss = loss["total_negative_log_likelihood"] / expected_prediction_tokens
    if not math.isclose(loss["mean"], recomputed_loss, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("evaluation report mean loss changed")
    if not math.isclose(
        loss["perplexity"], _perplexity(loss["mean"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("evaluation report perplexity changed")
    if not math.isclose(
        runtime["tokens_per_second"],
        expected_input_tokens / runtime["forward_seconds"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("evaluation report throughput changed")

    expected_strata = {item["id"]: item for item in heldout_manifest["selection"]["strata"]}
    actual_items = loss.get("per_stratum", [])
    actual_strata = {item["id"]: item for item in actual_items}
    if len(actual_strata) != len(actual_items) or set(actual_strata) != set(expected_strata):
        raise ValueError("evaluation report source strata changed")
    stratum_nll = 0.0
    for stratum_id, expected in expected_strata.items():
        actual = actual_strata[stratum_id]
        prediction_tokens = expected["target_sequences"] * (sequence_length - 1)
        if actual["sequences"] != expected["target_sequences"]:
            raise ValueError("evaluation report source sequence count changed")
        if actual["input_tokens"] != expected["target_tokens"]:
            raise ValueError("evaluation report source input-token count changed")
        if actual["prediction_tokens"] != prediction_tokens:
            raise ValueError("evaluation report source prediction-token count changed")
        actual_loss = actual["total_negative_log_likelihood"] / prediction_tokens
        if not math.isclose(actual["loss"], actual_loss, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("evaluation report source loss changed")
        if not math.isclose(
            actual["perplexity"],
            _perplexity(actual["loss"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("evaluation report source perplexity changed")
        stratum_nll += actual["total_negative_log_likelihood"]
    if not math.isclose(
        stratum_nll,
        loss["total_negative_log_likelihood"],
        rel_tol=1e-12,
        abs_tol=1e-8,
    ):
        raise ValueError("evaluation report source losses do not match the total")

    generations = report.get("generation", [])
    if [item.get("prompt") for item in generations] != prompts:
        raise ValueError("evaluation report fixed prompts changed")
    for item in generations:
        if item.get("new_tokens") != len(item.get("new_token_ids", [])):
            raise ValueError("evaluation report generated-token count changed")
    return report


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
    prompts: list[str],
    generation_max_new_tokens: int = 32,
    batch_size: int = 1,
    checkpoint_batches: int = 10,
    device: str = "auto",
    resume: bool = False,
    _interrupt_after_sequences: int | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on a sealed held-out artifact with exact resume."""

    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    tokenizer_path = Path(tokenizer_path)
    output = Path(output)
    if batch_size < 1 or checkpoint_batches < 1:
        raise ValueError("evaluation batch and checkpoint sizes must be positive")
    if generation_max_new_tokens < 1:
        raise ValueError("generation_max_new_tokens must be positive")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("evaluation requires non-empty fixed prompts")

    heldout_manifest = validate_heldout_manifest(manifest_path)
    manifest_artifact = _file_artifact(manifest_path)
    checkpoint_artifact = _checkpoint_artifact(checkpoint_path)
    tokenizer_artifact = _tokenizer_artifact(
        tokenizer_path, heldout_manifest["tokenizer"]["files"]
    )
    evaluation_device = _resolve_device(device)
    identity = {
        "batch_size": batch_size,
        "checkpoint": checkpoint_artifact,
        "data": {
            "build_fingerprint": heldout_manifest["build_fingerprint"],
            "manifest": manifest_artifact,
        },
        "device": evaluation_device.type,
        "generation_max_new_tokens": generation_max_new_tokens,
        "prompts": prompts,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "tokenizer": tokenizer_artifact,
    }
    evaluation_fingerprint = _fingerprint(identity)
    report_path = output / EVALUATION_REPORT_FILENAME
    state_path = output / EVALUATION_STATE_FILENAME

    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"evaluation report already exists: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return _validate_report(
            report,
            fingerprint=evaluation_fingerprint,
            heldout_manifest=heldout_manifest,
            prompts=prompts,
        )
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    output.mkdir(parents=True, exist_ok=True)

    strata = heldout_manifest["selection"]["strata"]
    dataset = PackedTokenDataset(manifest_path, verify_hashes=False)
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"evaluation state exists; pass --resume: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_state(
            state,
            fingerprint=evaluation_fingerprint,
            strata=strata,
            total=len(dataset),
        )
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"evaluation output is not empty: {output}")
        state = _initial_state(
            fingerprint=evaluation_fingerprint,
            strata=strata,
            device=evaluation_device,
        )
        _write_json_atomic(state_path, state)

    checkpoint_config = json.loads((checkpoint_path / "config.json").read_text())
    if checkpoint_config.get("vocab_size") != heldout_manifest["tokenizer"]["vocab_size"]:
        raise ValueError("checkpoint and held-out tokenizer vocabularies differ")
    sequence_length = heldout_manifest["packing"]["sequence_length"]
    if checkpoint_config.get("max_position_embeddings", 0) < sequence_length:
        raise ValueError("checkpoint context length is smaller than held-out sequences")

    load_started = time.perf_counter()
    model = LumiForCausalLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to(evaluation_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    _synchronize(evaluation_device)
    load_seconds = time.perf_counter() - load_started
    state["device_memory_peak_observed"] = _merge_memory_peak(
        state["device_memory_peak_observed"], _device_memory(evaluation_device)
    )
    if evaluation_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluation_device)

    with torch.inference_mode():
        while state["sequence_cursor"] < len(dataset):
            cursor = state["sequence_cursor"]
            stratum = _stratum_at_cursor(strata, cursor)
            count = min(batch_size, stratum["sequence_end"] - cursor)
            _synchronize(evaluation_device)
            started = time.perf_counter()
            batch = torch.stack(
                [dataset[index] for index in range(cursor, cursor + count)]
            ).to(evaluation_device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite held-out loss at sequence {cursor}: {loss}")
            _synchronize(evaluation_device)
            elapsed = max(time.perf_counter() - started, 1e-9)
            memory = _device_memory(evaluation_device)
            loss_value = float(loss.detach().cpu())
            prediction_tokens = count * (sequence_length - 1)
            input_tokens = count * sequence_length
            negative_log_likelihood = loss_value * prediction_tokens

            state["batches"] += 1
            state["forward_seconds"] += elapsed
            state["input_tokens"] += input_tokens
            state["prediction_tokens"] += prediction_tokens
            state["sequence_cursor"] += count
            state["total_negative_log_likelihood"] += negative_log_likelihood
            state["device_memory_peak_observed"] = _merge_memory_peak(
                state["device_memory_peak_observed"], memory
            )
            stratum_state = state["strata"][stratum["id"]]
            stratum_state["input_tokens"] += input_tokens
            stratum_state["prediction_tokens"] += prediction_tokens
            stratum_state["sequences"] += count
            stratum_state["total_negative_log_likelihood"] += negative_log_likelihood
            state["updated_at"] = _utc_now()
            del outputs, loss, batch

            should_checkpoint = (
                state["batches"] % checkpoint_batches == 0
                or state["sequence_cursor"] == len(dataset)
                or (
                    _interrupt_after_sequences is not None
                    and state["sequence_cursor"] >= _interrupt_after_sequences
                )
            )
            if should_checkpoint:
                _write_json_atomic(state_path, state)
                current_loss = (
                    state["total_negative_log_likelihood"]
                    / state["prediction_tokens"]
                )
                print(
                    f"evaluated_sequences={state['sequence_cursor']}/{len(dataset)} "
                    f"loss={current_loss:.6f}",
                    flush=True,
                )
            if (
                _interrupt_after_sequences is not None
                and state["sequence_cursor"] >= _interrupt_after_sequences
            ):
                raise RuntimeError("simulated checkpoint evaluation interruption")

        generations = []
        torch.manual_seed(0)
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            bos_token_id = model.config.bos_token_id
            if bos_token_id is not None:
                prompt_ids = [bos_token_id, *prompt_ids]
            if len(prompt_ids) + generation_max_new_tokens > model.config.max_position_embeddings:
                raise ValueError(f"fixed prompt exceeds checkpoint context window: {prompt!r}")
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=evaluation_device)
            _synchronize(evaluation_device)
            started = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
            _synchronize(evaluation_device)
            elapsed = max(time.perf_counter() - started, 1e-9)
            new_token_ids = generated[0, len(prompt_ids) :].tolist()
            generations.append(
                {
                    "continuation": tokenizer.decode(
                        new_token_ids, skip_special_tokens=False
                    ),
                    "elapsed_seconds": elapsed,
                    "full_text": tokenizer.decode(
                        generated[0].tolist(), skip_special_tokens=False
                    ),
                    "new_token_ids": new_token_ids,
                    "new_tokens": len(new_token_ids),
                    "prompt": prompt,
                    "tokens_per_second": len(new_token_ids) / elapsed,
                }
            )
            state["device_memory_peak_observed"] = _merge_memory_peak(
                state["device_memory_peak_observed"], _device_memory(evaluation_device)
            )

    mean_loss = state["total_negative_log_likelihood"] / state["prediction_tokens"]
    per_stratum = []
    for item in strata:
        values = state["strata"][item["id"]]
        stratum_loss = (
            values["total_negative_log_likelihood"] / values["prediction_tokens"]
        )
        per_stratum.append(
            {
                "category": item["category"],
                "component": item["component"],
                "id": item["id"],
                "loss": stratum_loss,
                "perplexity": _perplexity(stratum_loss),
                **values,
            }
        )

    if evaluation_device.type == "cuda":
        cuda_peak = {
            "allocated_bytes": torch.cuda.max_memory_allocated(evaluation_device),
            "driver_bytes": torch.cuda.max_memory_reserved(evaluation_device),
        }
        state["device_memory_peak_observed"] = _merge_memory_peak(
            state["device_memory_peak_observed"], cuda_peak
        )
    report = {
        "artifacts": {
            "checkpoint": checkpoint_artifact,
            "heldout_manifest": manifest_artifact,
            "heldout_build_fingerprint": heldout_manifest["build_fingerprint"],
            "tokenizer": tokenizer_artifact,
        },
        "completed_at": _utc_now(),
        "evaluation_fingerprint": evaluation_fingerprint,
        "generation": generations,
        "locations": {
            "checkpoint": str(checkpoint_path.resolve()),
            "heldout_manifest": str(manifest_path.resolve()),
            "tokenizer": str(tokenizer_path.resolve()),
        },
        "loss": {
            "mean": mean_loss,
            "per_stratum": per_stratum,
            "perplexity": _perplexity(mean_loss),
            "prediction_tokens": state["prediction_tokens"],
            "total_negative_log_likelihood": state[
                "total_negative_log_likelihood"
            ],
        },
        "memory": {
            **state["device_memory_peak_observed"],
            "measurement": "peak observed at synchronized evaluation boundaries",
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "runtime": {
            "batches": state["batches"],
            "device": evaluation_device.type,
            "forward_seconds": state["forward_seconds"],
            "input_tokens": state["input_tokens"],
            "load_seconds": load_seconds,
            "machine": platform.machine(),
            "model_parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            ),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "platform": platform.platform(),
            "sequences": state["sequence_cursor"],
            "torch_version": torch.__version__,
            "tokens_per_second": state["input_tokens"] / state["forward_seconds"],
        },
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
    }
    _validate_report(
        report,
        fingerprint=evaluation_fingerprint,
        heldout_manifest=heldout_manifest,
        prompts=prompts,
    )
    _write_json_atomic(report_path, report)
    state["status"] = "complete"
    state["updated_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    return report
