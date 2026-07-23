"""Portable Hugging Face export for one verified Lumi distributed checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from .configuration_lumi import LumiConfig
from .distributed_training import verify_distributed_checkpoint
from .modeling_lumi import LumiForCausalLM

EXPORT_REPORT_FILENAME = "distributed-export-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifacts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != EXPORT_REPORT_FILENAME
    ]


def validate_distributed_hf_export(
    output: str | Path,
    *,
    checkpoint_manifest_sha256: str,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Validate an existing immutable export and its exact DCP binding."""

    output = Path(output)
    report_path = output / EXPORT_REPORT_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("unsupported distributed HF export report")
    if report.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256:
        raise ValueError("distributed HF export targets another checkpoint manifest")
    if report.get("checkpoint_step") != checkpoint_step:
        raise ValueError("distributed HF export targets another checkpoint step")
    if report.get("files") != _artifacts(output):
        raise ValueError("distributed HF export file inventory changed")
    if not (output / "config.json").is_file():
        raise FileNotFoundError("distributed HF export has no config.json")
    if not list(output.glob("model*.safetensors")):
        raise FileNotFoundError("distributed HF export has no safetensors weights")
    return report


def export_distributed_checkpoint_hf(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Load one DCP checkpoint into a plain model and publish a sealed HF export."""

    config_path = Path(config_path)
    checkpoint = Path(checkpoint)
    output = Path(output)
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("distributed HF export dtype must be bfloat16 or float32")
    checkpoint_state = verify_distributed_checkpoint(checkpoint)
    checkpoint_manifest_sha256 = _sha256(checkpoint / "manifest.sha256")
    checkpoint_step = int(checkpoint_state["step"])
    if output.is_dir() and (output / EXPORT_REPORT_FILENAME).is_file():
        return validate_distributed_hf_export(
            output,
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            checkpoint_step=checkpoint_step,
        )
    if output.exists():
        raise FileExistsError(f"distributed HF export output exists: {output}")

    config = LumiConfig(**json.loads(config_path.read_bytes()))
    model = LumiForCausalLM(config)
    state = model.state_dict()
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model", no_dist=True)
    incompatible = model.load_state_dict(state)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"distributed HF export restore was incompatible: {incompatible}")
    if dtype == "bfloat16":
        model.to(torch.bfloat16)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        files = _artifacts(temporary)
        if not files:
            raise RuntimeError("distributed HF export produced no files")
        report = {
            "checkpoint": checkpoint.name,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_step": checkpoint_step,
            "dtype": dtype,
            "files": files,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "schema_version": 1,
            "status": "complete",
        }
        _write_json(temporary / EXPORT_REPORT_FILENAME, report)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_distributed_hf_export(
        output,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        checkpoint_step=checkpoint_step,
    )
