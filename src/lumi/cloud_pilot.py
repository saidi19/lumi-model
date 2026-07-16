"""Portable artifact and CUDA preflight for the paid Lumi pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .configuration_lumi import LumiConfig
from .packed_data import validate_packed_manifest
from .parameter_count import parameter_breakdown

CLOUD_PILOT_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_run_plan(plan: dict[str, Any], *, sequence_length: int, tokens: int) -> None:
    smoke = plan.get("smoke", {})
    first_steps = _positive_int(smoke.get("first_steps"), name="smoke.first_steps")
    resume_steps = _positive_int(smoke.get("resume_steps"), name="smoke.resume_steps")
    if resume_steps <= first_steps:
        raise ValueError("smoke.resume_steps must be greater than smoke.first_steps")
    _positive_int(smoke.get("batch_size"), name="smoke.batch_size")
    _positive_int(
        smoke.get("gradient_accumulation_steps"),
        name="smoke.gradient_accumulation_steps",
    )
    _positive_int(smoke.get("checkpoint_steps"), name="smoke.checkpoint_steps")

    full = plan.get("full", {})
    steps = _positive_int(full.get("steps"), name="full.steps")
    batch_size = _positive_int(full.get("batch_size"), name="full.batch_size")
    accumulation = _positive_int(
        full.get("gradient_accumulation_steps"),
        name="full.gradient_accumulation_steps",
    )
    checkpoint_steps = _positive_int(
        full.get("checkpoint_steps"), name="full.checkpoint_steps"
    )
    if checkpoint_steps > steps:
        raise ValueError("full.checkpoint_steps cannot exceed full.steps")

    tokens_per_step = sequence_length * batch_size * accumulation
    processed_tokens = tokens_per_step * steps
    repeated_tokens = processed_tokens - tokens
    if processed_tokens < tokens or repeated_tokens >= tokens_per_step:
        raise ValueError("full run must cover the corpus with less than one repeated step")
    expected = {
        "tokens_per_step": tokens_per_step,
        "processed_tokens": processed_tokens,
        "repeated_tokens": repeated_tokens,
    }
    for name, value in expected.items():
        if full.get(name) != value:
            raise ValueError(f"full.{name} does not match the derived value {value}")

    optimizer = plan.get("optimizer", {})
    for name in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = optimizer.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"optimizer.{name} must be a non-negative number")
    if optimizer["learning_rate"] == 0 or optimizer["max_grad_norm"] == 0:
        raise ValueError("learning rate and maximum gradient norm must be positive")
    _positive_int(optimizer.get("seed"), name="optimizer.seed")


def load_cloud_pilot_plan(plan_path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a pinned cloud-pilot plan."""

    plan_path = Path(plan_path)
    plan = _read_json(plan_path)
    if plan.get("schema_version") != CLOUD_PILOT_SCHEMA_VERSION:
        raise ValueError("unsupported cloud-pilot plan schema version")
    if not isinstance(plan.get("requires_bf16"), bool):
        raise ValueError("requires_bf16 must be a boolean")

    artifact = plan.get("artifact", {})
    tokens = _positive_int(artifact.get("tokens"), name="artifact.tokens")
    sequences = _positive_int(artifact.get("sequences"), name="artifact.sequences")
    sequence_length = _positive_int(
        artifact.get("sequence_length"), name="artifact.sequence_length"
    )
    if tokens != sequences * sequence_length:
        raise ValueError("artifact token count is not sequence aligned")
    _positive_int(artifact.get("source_strata"), name="artifact.source_strata")

    model = plan.get("model", {})
    if not isinstance(model.get("config"), str) or not model["config"]:
        raise ValueError("model.config must be a nonempty relative path")
    if Path(model["config"]).is_absolute() or ".." in Path(model["config"]).parts:
        raise ValueError("model.config must stay inside the plan directory")
    _positive_int(model.get("parameters"), name="model.parameters")

    _validate_run_plan(plan, sequence_length=sequence_length, tokens=tokens)
    return plan


def _verify_tokenizer(
    *, tokenizer_path: Path, manifest: dict[str, Any], expected_vocab_size: int
) -> None:
    expected_files = manifest["tokenizer"]["files"]
    names = [item["name"] for item in expected_files]
    if len(names) != len(set(names)):
        raise ValueError("packed manifest contains duplicate tokenizer files")
    for item in expected_files:
        path = tokenizer_path / item["name"]
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer artifact is missing {path}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"tokenizer file size changed: {path}")
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"tokenizer file SHA-256 changed: {path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if len(tokenizer) != expected_vocab_size:
        raise ValueError("tokenizer vocabulary does not match the pilot plan")
    expected_doc_token = manifest["tokenizer"]["doc_token"]
    expected_doc_token_id = manifest["tokenizer"]["doc_token_id"]
    if tokenizer.convert_tokens_to_ids(expected_doc_token) != expected_doc_token_id:
        raise ValueError("tokenizer document token does not match the packed manifest")


def _runtime_report(*, require_cuda: bool, require_bf16: bool) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": properties.total_memory,
                }
            )
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
    if require_cuda and not cuda_available:
        raise RuntimeError("cloud preflight requires CUDA, but CUDA is unavailable")
    if require_bf16 and not bf16_supported:
        raise RuntimeError("cloud preflight requires CUDA BF16 support")
    return {
        "bf16_supported": bf16_supported,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "devices": devices,
        "torch_version": torch.__version__,
    }


def preflight_cloud_pilot(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    require_cuda: bool = False,
    verify_shards: bool = True,
) -> dict[str, Any]:
    """Verify portable pilot inputs and, on cloud, the CUDA/BF16 runtime."""

    plan_path = Path(plan_path)
    manifest_path = Path(manifest_path)
    tokenizer_path = Path(tokenizer_path)
    plan = load_cloud_pilot_plan(plan_path)
    artifact = plan["artifact"]
    model_plan = plan["model"]

    if _sha256_file(manifest_path) != artifact["manifest_sha256"]:
        raise ValueError("packed manifest SHA-256 does not match the cloud-pilot plan")
    manifest = validate_packed_manifest(manifest_path, verify_hashes=verify_shards)
    actual_artifact = {
        "build_fingerprint": manifest["build_fingerprint"],
        "sequence_length": manifest["packing"]["sequence_length"],
        "sequences": manifest["statistics"]["sequences"],
        "source_strata": len(manifest.get("selection", {}).get("strata", [])),
        "tokens": manifest["statistics"]["tokens"],
    }
    for name, expected in artifact.items():
        if name == "manifest_sha256":
            continue
        if actual_artifact.get(name) != expected:
            raise ValueError(f"packed artifact {name} does not match the cloud-pilot plan")

    config_path = plan_path.parent / model_plan["config"]
    if _sha256_file(config_path) != model_plan["config_sha256"]:
        raise ValueError("model config SHA-256 does not match the cloud-pilot plan")
    config = LumiConfig.from_json_file(config_path)
    parameters = parameter_breakdown(config).total
    if parameters != model_plan["parameters"]:
        raise ValueError("model parameter count does not match the cloud-pilot plan")
    if config.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("model and packed tokenizer vocabulary sizes differ")
    if config.doc_token_id != manifest["tokenizer"]["doc_token_id"]:
        raise ValueError("model and packed document token IDs differ")
    if config.max_position_embeddings < artifact["sequence_length"]:
        raise ValueError("model context length is shorter than packed sequences")

    _verify_tokenizer(
        tokenizer_path=tokenizer_path,
        manifest=manifest,
        expected_vocab_size=config.vocab_size,
    )
    runtime = _runtime_report(
        require_cuda=require_cuda,
        require_bf16=require_cuda and bool(plan.get("requires_bf16", True)),
    )
    return {
        "artifact": actual_artifact,
        "manifest_sha256": artifact["manifest_sha256"],
        "model_config_sha256": model_plan["config_sha256"],
        "parameters": parameters,
        "plan_sha256": _sha256_file(plan_path),
        "runtime": runtime,
        "status": "pass",
    }
