"""Fail-closed planning and evidence helpers for Lumi's paid CUDA smoke."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .configuration_lumi import LumiConfig
from .distributed_training import (
    DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
    PAID_SMOKE_MAX_STEPS,
    _validate_source_commit,
    build_distributed_run_identity,
)
from .packed_data import validate_packed_manifest
from .parameter_count import parameter_breakdown
from .pilot_training import _fingerprint

SMOKE_CONTRACT_SCHEMA_VERSION = 1
SMOKE_RUN_KIND = "lumi_distributed_smoke_v0.1"
SMOKE_OWNER_ACKNOWLEDGEMENT = (
    "I approve this paid Lumi distributed smoke run through the recorded step."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != SMOKE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported distributed CUDA smoke contract")
    if contract.get("status") != "planning_only":
        raise ValueError("distributed CUDA smoke contract must remain planning-only")
    approval = contract.get("approval", {})
    if approval != {"full_4b_training": False, "paid_gpu_launch": False}:
        raise ValueError("repository smoke contract cannot authorize paid or full training")
    run = contract.get("run", {})
    required = {
        "attention_implementation": "flash_attention_2",
        "checkpoint_steps": 1,
        "document_isolation": True,
        "fresh_through_step": 1,
        "gradient_checkpointing": True,
        "gradient_sync_mode": "last_microbatch",
        "keep_checkpoints": 1,
        "lr_schedule": "constant",
        "optimizer_profile": "legacy",
        "precision": "bf16",
        "resume_through_step": 2,
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
        "world_size": 2,
    }
    for key, expected in required.items():
        if run.get(key) != expected:
            raise ValueError(f"distributed CUDA smoke contract changed {key}")
    if run["resume_through_step"] > PAID_SMOKE_MAX_STEPS:
        raise ValueError("distributed CUDA smoke exceeds the trainer's paid step cap")
    if run.get("per_rank_batch_size") != 1 or run.get("gradient_accumulation_steps") != 1:
        raise ValueError("distributed CUDA smoke batch contract changed")
    return run


def plan_distributed_cuda_smoke(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    source_commit: str,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Return the exact fingerprint that a later owner authorization may approve."""

    contract_path = Path(contract_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    run = _validate_contract(contract)
    normalized_commit = _validate_source_commit(source_commit)
    assert normalized_commit is not None

    manifest = validate_packed_manifest(manifest_path, verify_hashes=True)
    data_contract = contract["data"]
    provenance = manifest.get("provenance", {})
    if provenance.get("build_fingerprint") != data_contract["parent_build_fingerprint"]:
        raise ValueError("smoke fixture parent build fingerprint changed")
    if provenance.get("manifest_sha256") != data_contract["parent_manifest_sha256"]:
        raise ValueError("smoke fixture parent manifest changed")
    if manifest["packing"]["sequence_length"] != data_contract["sequence_length"]:
        raise ValueError("smoke fixture sequence length changed")
    if manifest["statistics"]["sequences"] != data_contract["smoke_sequences"]:
        raise ValueError("smoke fixture sequence count changed")
    if int(manifest["statistics"].get("document_markers", 0)) < 1:
        raise ValueError("smoke fixture has no document boundaries")

    config_bytes = config_path.read_bytes()
    config = LumiConfig(**json.loads(config_bytes))
    parameters = parameter_breakdown(config).total
    if parameters != contract["model"]["parameters"]:
        raise ValueError("smoke model parameter count changed")
    if config.initialization_strategy != "residual_scaled_v0.1":
        raise ValueError("smoke model does not use the selected initialization")
    if config.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("smoke model vocabulary differs from the fixture")
    if config.doc_token_id != manifest["tokenizer"]["doc_token_id"]:
        raise ValueError("smoke model document token differs from the fixture")

    runtime: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }
    if require_cuda:
        if not runtime["cuda_available"]:
            raise RuntimeError("distributed CUDA smoke planning requires CUDA")
        if runtime["cuda_device_count"] < run["world_size"]:
            raise RuntimeError("distributed CUDA smoke requires at least two visible GPUs")
        capabilities = [
            list(torch.cuda.get_device_capability(index))
            for index in range(run["world_size"])
        ]
        names = [torch.cuda.get_device_name(index) for index in range(run["world_size"])]
        if any(capability[0] < 8 for capability in capabilities):
            raise RuntimeError("distributed CUDA smoke GPUs do not support BF16")
        try:
            import flash_attn
        except ImportError as error:
            raise RuntimeError("FlashAttention is unavailable for CUDA smoke planning") from error
        runtime.update(
            {
                "cuda_capabilities": capabilities,
                "flash_attention_version": flash_attn.__version__,
                "gpu_names": names,
            }
        )

    manifest_sha256 = _sha256(manifest_path)
    identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        backend="nccl",
        world_size=run["world_size"],
        strategy=run["strategy"],
        per_rank_batch_size=run["per_rank_batch_size"],
        gradient_accumulation_steps=run["gradient_accumulation_steps"],
        learning_rate=run["learning_rate"],
        weight_decay=run["weight_decay"],
        max_grad_norm=run["max_grad_norm"],
        keep_checkpoints=run["keep_checkpoints"],
        seed=run["seed"],
        gradient_checkpointing=run["gradient_checkpointing"],
        lr_schedule=run["lr_schedule"],
        warmup_steps=0,
        schedule_steps=None,
        min_learning_rate_ratio=0.1,
        attention_implementation=run["attention_implementation"],
        document_isolation=run["document_isolation"],
        optimizer_profile=run["optimizer_profile"],
        precision=run["precision"],
        gradient_sync_mode=run["gradient_sync_mode"],
        torch_compile=run["torch_compile"],
        source_commit=normalized_commit,
    )
    run_fingerprint = _fingerprint(identity)
    global_tokens_per_step = (
        run["world_size"]
        * run["per_rank_batch_size"]
        * run["gradient_accumulation_steps"]
        * manifest["packing"]["sequence_length"]
    )
    plan = {
        "authorization_template": {
            "approved_through_step": run["resume_through_step"],
            "full_4b_training": False,
            "owner_acknowledgement": SMOKE_OWNER_ACKNOWLEDGEMENT,
            "paid_gpu": False,
            "run_fingerprint": run_fingerprint,
            "run_kind": SMOKE_RUN_KIND,
            "schema_version": DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
            "source_commit": normalized_commit,
            "status": "planning_only",
            "strategy": run["strategy"],
            "world_size": run["world_size"],
        },
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
        },
        "created_at": _utc_now(),
        "data": {
            "build_fingerprint": manifest["build_fingerprint"],
            "document_markers": manifest["statistics"]["document_markers"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "sequences": manifest["statistics"]["sequences"],
            "tokens": manifest["statistics"]["tokens"],
        },
        "global_tokens_per_step": global_tokens_per_step,
        "model": {
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "parameters": parameters,
        },
        "run_fingerprint": run_fingerprint,
        "run_identity": identity,
        "runtime": runtime,
        "schema_version": SMOKE_CONTRACT_SCHEMA_VERSION,
        "source_commit": normalized_commit,
        "status": "planning_only",
        "target": {
            "fresh_through_step": run["fresh_through_step"],
            "resume_through_step": run["resume_through_step"],
            "tokens_through_resume": run["resume_through_step"]
            * global_tokens_per_step,
        },
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return plan


def write_owner_authorization(
    *,
    plan_path: str | Path,
    output: str | Path,
    provider: str,
    instance_type: str,
    hourly_rate_usd: float,
    max_runtime_minutes: int,
) -> dict[str, Any]:
    """Convert one immutable plan into the narrowly bounded owner artifact."""

    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_plan_sha256 = plan.get("plan_sha256")
    without_sha = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected_plan_sha256 != _fingerprint(without_sha):
        raise ValueError("distributed CUDA smoke plan hash changed")
    if plan.get("status") != "planning_only":
        raise ValueError("only a planning-only smoke plan can be authorized")
    template = plan.get("authorization_template", {})
    if template.get("paid_gpu") is not False or template.get("full_4b_training") is not False:
        raise ValueError("smoke plan authorization template is unsafe")
    if not provider.strip() or not instance_type.strip():
        raise ValueError("provider and instance type are required")
    if hourly_rate_usd <= 0 or hourly_rate_usd > 100:
        raise ValueError("hourly rate is outside the smoke safety range")
    if max_runtime_minutes < 10 or max_runtime_minutes > 180:
        raise ValueError("smoke maximum runtime must be between 10 and 180 minutes")
    authorization = {
        **template,
        "approved_at": _utc_now(),
        "full_4b_training": False,
        "hourly_rate_usd": hourly_rate_usd,
        "instance_type": instance_type.strip(),
        "max_runtime_minutes": max_runtime_minutes,
        "paid_gpu": True,
        "plan_sha256": expected_plan_sha256,
        "provider": provider.strip(),
        "status": "owner_approved",
    }
    _write_json_atomic(Path(output), authorization)
    return authorization


def validate_owner_authorization(
    *, plan_path: str | Path, authorization_path: str | Path
) -> dict[str, Any]:
    """Validate every material owner boundary before a paid shell starts."""

    plan_path = Path(plan_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_plan_sha256 = plan.get("plan_sha256")
    without_sha = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected_plan_sha256 != _fingerprint(without_sha):
        raise ValueError("distributed CUDA smoke plan hash changed")
    template = plan.get("authorization_template", {})
    exact_fields = (
        "approved_through_step",
        "full_4b_training",
        "owner_acknowledgement",
        "run_fingerprint",
        "run_kind",
        "schema_version",
        "source_commit",
        "strategy",
        "world_size",
    )
    for field in exact_fields:
        if authorization.get(field) != template.get(field):
            raise ValueError(f"distributed CUDA authorization changed {field}")
    if authorization.get("status") != "owner_approved":
        raise ValueError("distributed CUDA smoke is not owner-approved")
    if authorization.get("paid_gpu") is not True:
        raise ValueError("distributed CUDA smoke paid boundary is not explicit")
    if authorization.get("full_4b_training") is not False:
        raise ValueError("distributed CUDA smoke cannot authorize full 4B training")
    if authorization.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("distributed CUDA authorization targets another plan")
    if not isinstance(authorization.get("max_runtime_minutes"), int) or not (
        10 <= authorization["max_runtime_minutes"] <= 180
    ):
        raise ValueError("distributed CUDA authorization runtime cap is invalid")
    hourly_rate = authorization.get("hourly_rate_usd")
    if not isinstance(hourly_rate, (int, float)) or isinstance(hourly_rate, bool):
        raise ValueError("distributed CUDA authorization hourly rate is invalid")
    if not 0 < float(hourly_rate) <= 100:
        raise ValueError("distributed CUDA authorization hourly rate is unsafe")
    return authorization
