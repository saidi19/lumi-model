"""Fail-closed planning and authorization for Lumi's bounded 4B learning pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .configuration_lumi import LumiConfig
from .distributed_training import (
    DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
    PAID_LEARNING_PILOT_MAX_RUNTIME_MINUTES,
    PAID_LEARNING_PILOT_MAX_STEPS,
    PAID_LEARNING_PILOT_MAX_TOKENS,
    PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT,
    PAID_LEARNING_PILOT_RUN_KIND,
    PAID_LEARNING_PILOT_WORLD_SIZES,
    _validate_source_commit,
    build_distributed_run_identity,
)
from .evaluation_data import validate_heldout_manifest
from .packed_data import validate_packed_manifest
from .parameter_count import parameter_breakdown
from .pilot_training import _fingerprint
from .production_training import validate_final_corpus_training_manifest

LEARNING_PILOT_SCHEMA_VERSION = 1


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


def _max_runtime_minutes(plan: dict[str, Any]) -> int:
    world_size = plan.get("selected_topology", {}).get("world_size")
    if world_size not in PAID_LEARNING_PILOT_MAX_RUNTIME_MINUTES:
        raise ValueError("Lumi-4B learning-pilot plan has an unsupported topology")
    return PAID_LEARNING_PILOT_MAX_RUNTIME_MINUTES[world_size]


def prepare_learning_pilot_fixture(
    *,
    contract_path: str | Path,
    final_manifest_path: str | Path,
    source_prefix_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate and seal the exact sequence-aligned byte-range pilot prefix."""

    contract_path = Path(contract_path).resolve()
    final_manifest_path = Path(final_manifest_path).resolve()
    source_prefix_path = Path(source_prefix_path).resolve()
    output = Path(output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    corpus = contract["corpus"]
    final_manifest = validate_final_corpus_training_manifest(
        final_manifest_path,
        sequence_length=corpus["sequence_length"],
        vocab_size=corpus["vocab_size"],
        doc_token_id=corpus["doc_token_id"],
        expected_manifest_sha256=corpus["final_manifest_sha256"],
        expected_plan_fingerprint=corpus["final_plan_fingerprint"],
        expected_selection_result_sha256=corpus["final_selection_result_sha256"],
    )
    source_contract = corpus["source_shard"]
    source = final_manifest["shards"][source_contract["sequence"]]
    expected_source = {
        "bytes": source_contract["bytes"],
        "category": source_contract["category"],
        "sequence": source_contract["sequence"],
        "sha256": source_contract["sha256"],
        "s3_uri": source_contract["s3_uri"],
        "tokens": source_contract["tokens"],
    }
    observed_source = {
        "bytes": source["bytes"],
        "category": source["category"],
        "sequence": source["sequence"],
        "sha256": source["sha256"],
        "s3_uri": source["s3_uri"],
        "tokens": source["tokens"],
    }
    if observed_source != expected_source:
        raise ValueError("learning-pilot source shard changed")
    prefix_contract = corpus["fixture_prefix"]
    if source_prefix_path.stat().st_size != prefix_contract["bytes"]:
        raise ValueError("learning-pilot source prefix byte size changed")
    if _sha256(source_prefix_path) != prefix_contract["sha256"]:
        raise ValueError("learning-pilot source prefix SHA-256 changed")

    manifest_path = output / "manifest.json"
    report_path = output / "fixture-report.json"
    if manifest_path.is_file() and report_path.is_file():
        manifest = validate_packed_manifest(manifest_path, verify_hashes=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            manifest["statistics"]["tokens"] != corpus["fixture_tokens"]
            or report.get("manifest_sha256") != _sha256(manifest_path)
        ):
            raise ValueError("existing learning-pilot fixture changed")
        return report
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"learning-pilot fixture output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    output_shard = output / "shard-00000.bin"
    output_bytes = corpus["fixture_tokens"] * 2
    if output_bytes != prefix_contract["bytes"]:
        raise ValueError("learning-pilot fixture and source prefix sizes differ")
    document_markers = 0
    digest = hashlib.sha256()
    with (
        source_prefix_path.open("rb") as source_handle,
        output_shard.open("wb") as output_handle,
    ):
        remaining = output_bytes
        while remaining:
            block = source_handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError("learning-pilot source shard ended early")
            if len(block) % 2:
                raise RuntimeError("learning-pilot uint16 block is not aligned")
            output_handle.write(block)
            digest.update(block)
            values = array("H")
            values.frombytes(block)
            if sys.byteorder != "little":
                values.byteswap()
            document_markers += values.count(corpus["doc_token_id"])
            remaining -= len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if document_markers < 1:
        raise RuntimeError("learning-pilot fixture contains no document markers")
    if document_markers != prefix_contract["document_markers"]:
        raise RuntimeError("learning-pilot source prefix marker count changed")
    if digest.hexdigest() != prefix_contract["sha256"]:
        raise RuntimeError("learning-pilot fixture prefix digest changed")

    provenance = {
        "final_manifest_sha256": corpus["final_manifest_sha256"],
        "final_plan_fingerprint": corpus["final_plan_fingerprint"],
        "final_selection_result_sha256": corpus["final_selection_result_sha256"],
        "source_shard_bytes": source_contract["bytes"],
        "source_shard_category": source_contract["category"],
        "source_shard_sequence": source_contract["sequence"],
        "source_shard_sha256": source_contract["sha256"],
        "source_shard_s3_uri": source_contract["s3_uri"],
        "source_shard_tokens": source_contract["tokens"],
        "source_prefix_bytes": prefix_contract["bytes"],
        "source_prefix_document_markers": prefix_contract["document_markers"],
        "source_prefix_range_end": prefix_contract["range_end"],
        "source_prefix_range_start": prefix_contract["range_start"],
        "source_prefix_sha256": prefix_contract["sha256"],
        "source_prefix_tokens": prefix_contract["tokens"],
        "start_token": 0,
    }
    build_specification = {
        "format": {
            "dtype": "uint16",
            "endianness": "little",
            "layout": "flat_fixed_sequences",
            "token_bytes": 2,
        },
        "packing": {
            "sequence_length": corpus["sequence_length"],
            "sequences": corpus["fixture_sequences"],
        },
        "provenance": provenance,
        "tokenizer": {
            "doc_token_id": corpus["doc_token_id"],
            "vocab_size": corpus["vocab_size"],
        },
    }
    build_fingerprint = _fingerprint(build_specification)
    manifest = {
        "build_fingerprint": build_fingerprint,
        "format": build_specification["format"],
        "packing": {
            "purpose": "lumi_4b_learning_pilot_v0.1",
            "sequence_length": corpus["sequence_length"],
        },
        "provenance": provenance,
        "schema_version": 1,
        "shards": [
            {
                "bytes": output_bytes,
                "name": output_shard.name,
                "sequences": corpus["fixture_sequences"],
                "sha256": digest.hexdigest(),
                "tokens": corpus["fixture_tokens"],
            }
        ],
        "statistics": {
            "document_markers": document_markers,
            "sequences": corpus["fixture_sequences"],
            "tokens": corpus["fixture_tokens"],
        },
        "status": "complete",
        "tokenizer": {
            "doc_token": "<|doc|>",
            "doc_token_id": corpus["doc_token_id"],
            "vocab_size": corpus["vocab_size"],
        },
    }
    _write_json_atomic(manifest_path, manifest)
    validate_packed_manifest(manifest_path, verify_hashes=True)
    report = {
        "build_fingerprint": build_fingerprint,
        "document_markers": document_markers,
        "fixture_sha256": digest.hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sequences": corpus["fixture_sequences"],
        "source_shard_category": corpus["source_shard"]["category"],
        "source_manifest_sha256": corpus["final_manifest_sha256"],
        "source_prefix_sha256": corpus["fixture_prefix"]["sha256"],
        "source_shard_sha256": corpus["source_shard"]["sha256"],
        "status": "pass",
        "tokens": corpus["fixture_tokens"],
    }
    _write_json_atomic(report_path, report)
    return report


def _validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema_version") != LEARNING_PILOT_SCHEMA_VERSION
        or contract.get("status") != "planning_only"
    ):
        raise ValueError("unsupported Lumi-4B learning-pilot contract")
    if contract.get("approval") != {
        "full_4b_training": False,
        "paid_multi_gpu_pilot": False,
    }:
        raise ValueError("repository learning-pilot contract cannot authorize paid work")
    run = contract.get("run", {})
    required = {
        "attention_implementation": "flash_attention_2",
        "checkpoint_steps": 10,
        "document_isolation": True,
        "fresh_through_step": 10,
        "global_sequences_per_step": 256,
        "gradient_checkpointing": True,
        "gradient_sync_mode": "last_microbatch",
        "keep_checkpoints": 2,
        "lr_schedule": "cosine",
        "optimizer_profile": "legacy",
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "resume_through_step": PAID_LEARNING_PILOT_MAX_STEPS,
        "schedule_steps": PAID_LEARNING_PILOT_MAX_STEPS,
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
        "warmup_steps": 10,
    }
    for key, expected in required.items():
        if run.get(key) != expected:
            raise ValueError(f"Lumi-4B learning-pilot contract changed {key}")
    corpus = contract.get("corpus", {})
    if corpus.get("fixture_tokens") != PAID_LEARNING_PILOT_MAX_TOKENS:
        raise ValueError("Lumi-4B learning-pilot token cap changed")
    prefix = corpus.get("fixture_prefix")
    if not isinstance(prefix, dict):
        raise ValueError("Lumi-4B learning-pilot source prefix is missing")
    if prefix.get("tokens") != corpus.get("fixture_tokens"):
        raise ValueError("Lumi-4B learning-pilot source-prefix token count changed")
    if prefix.get("bytes") != prefix.get("tokens") * 2:
        raise ValueError("Lumi-4B learning-pilot source-prefix byte count changed")
    if prefix.get("range_start") != 0 or prefix.get("range_end") != prefix.get("bytes") - 1:
        raise ValueError("Lumi-4B learning-pilot source-prefix byte range changed")
    if not isinstance(prefix.get("document_markers"), int) or prefix["document_markers"] < 1:
        raise ValueError("Lumi-4B learning-pilot source prefix has no markers")
    prefix_sha256 = prefix.get("sha256")
    if (
        not isinstance(prefix_sha256, str)
        or len(prefix_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prefix_sha256)
    ):
        raise ValueError("Lumi-4B learning-pilot source-prefix SHA-256 is invalid")
    topologies = run.get("topologies")
    if not isinstance(topologies, list) or len(topologies) != 2:
        raise ValueError("Lumi-4B learning-pilot topologies changed")
    observed_topologies = {
        (item.get("world_size"), item.get("gradient_accumulation_steps"))
        for item in topologies
        if isinstance(item, dict)
    }
    if observed_topologies != {(4, 64), (8, 32)}:
        raise ValueError("Lumi-4B learning-pilot topology geometry changed")
    if {item[0] for item in observed_topologies} != PAID_LEARNING_PILOT_WORLD_SIZES:
        raise ValueError("Lumi-4B learning-pilot world sizes changed")
    if contract.get("hardware") != {
        "compute_capability": [9, 0],
        "gpu_model_contains": "H100",
        "maximum_memory_bytes": 90_000_000_000,
        "minimum_free_disk_bytes": 300_000_000_000,
        "minimum_host_memory_bytes": 128_000_000_000,
        "minimum_memory_bytes": 80_000_000_000,
    }:
        raise ValueError("Lumi-4B learning-pilot hardware contract changed")
    expected_tokens = (
        run["global_sequences_per_step"]
        * corpus["sequence_length"]
        * run["resume_through_step"]
    )
    if expected_tokens != PAID_LEARNING_PILOT_MAX_TOKENS:
        raise ValueError("Lumi-4B learning-pilot batch geometry changed")
    return run


def plan_4b_learning_pilot(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    config_path: str | Path,
    source_commit: str,
    world_size: int,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Return the exact fingerprint that one bounded paid pilot may execute."""

    contract_path = Path(contract_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    config_path = Path(config_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    run = _validate_contract(contract)
    if world_size not in PAID_LEARNING_PILOT_WORLD_SIZES:
        raise ValueError("Lumi-4B learning pilot requires world size 4 or 8")
    topology = next(
        item for item in run["topologies"] if item["world_size"] == world_size
    )
    gradient_accumulation_steps = topology["gradient_accumulation_steps"]
    normalized_commit = _validate_source_commit(source_commit)
    assert normalized_commit is not None

    manifest = validate_packed_manifest(manifest_path, verify_hashes=True)
    corpus = contract["corpus"]
    provenance = manifest.get("provenance", {})
    expected_provenance = {
        "final_manifest_sha256": corpus["final_manifest_sha256"],
        "final_plan_fingerprint": corpus["final_plan_fingerprint"],
        "final_selection_result_sha256": corpus["final_selection_result_sha256"],
        "source_shard_bytes": corpus["source_shard"]["bytes"],
        "source_shard_category": corpus["source_shard"]["category"],
        "source_shard_sequence": corpus["source_shard"]["sequence"],
        "source_shard_sha256": corpus["source_shard"]["sha256"],
        "source_shard_s3_uri": corpus["source_shard"]["s3_uri"],
        "source_shard_tokens": corpus["source_shard"]["tokens"],
        "source_prefix_bytes": corpus["fixture_prefix"]["bytes"],
        "source_prefix_document_markers": corpus["fixture_prefix"]["document_markers"],
        "source_prefix_range_end": corpus["fixture_prefix"]["range_end"],
        "source_prefix_range_start": corpus["fixture_prefix"]["range_start"],
        "source_prefix_sha256": corpus["fixture_prefix"]["sha256"],
        "source_prefix_tokens": corpus["fixture_prefix"]["tokens"],
        "start_token": 0,
    }
    if provenance != expected_provenance:
        raise ValueError("learning-pilot fixture provenance changed")
    if manifest["packing"]["sequence_length"] != corpus["sequence_length"]:
        raise ValueError("learning-pilot fixture sequence length changed")
    if manifest["statistics"]["sequences"] != corpus["fixture_sequences"]:
        raise ValueError("learning-pilot fixture sequence count changed")
    if manifest["statistics"]["tokens"] != corpus["fixture_tokens"]:
        raise ValueError("learning-pilot fixture token count changed")
    if int(manifest["statistics"].get("document_markers", 0)) < 1:
        raise ValueError("learning-pilot fixture has no document boundaries")

    heldout = validate_heldout_manifest(heldout_manifest_path)
    evaluation = contract["evaluation"]
    if _sha256(heldout_manifest_path) != evaluation["heldout_manifest_sha256"]:
        raise ValueError("learning-pilot held-out manifest changed")
    if heldout["statistics"]["sequences"] != evaluation["heldout_sequences"]:
        raise ValueError("learning-pilot held-out sequence count changed")
    if heldout["statistics"]["tokens"] != evaluation["heldout_tokens"]:
        raise ValueError("learning-pilot held-out token count changed")

    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != contract["model"]["config_sha256"]:
        raise ValueError("Lumi-4B learning-pilot model config changed")
    config = LumiConfig(**json.loads(config_bytes))
    if parameter_breakdown(config).total != contract["model"]["parameters"]:
        raise ValueError("Lumi-4B learning-pilot parameter count changed")
    if config.initialization_strategy != "residual_scaled_v0.1":
        raise ValueError("Lumi-4B learning-pilot initialization changed")
    if config.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("Lumi-4B model vocabulary differs from the fixture")
    if config.doc_token_id != manifest["tokenizer"]["doc_token_id"]:
        raise ValueError("Lumi-4B document token differs from the fixture")

    runtime: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }
    if require_cuda:
        if not runtime["cuda_available"]:
            raise RuntimeError("Lumi-4B learning-pilot planning requires CUDA")
        if runtime["cuda_device_count"] < world_size:
            raise RuntimeError(
                f"Lumi-4B learning pilot requires {world_size} visible GPUs"
            )
        capabilities = [
            list(torch.cuda.get_device_capability(index))
            for index in range(world_size)
        ]
        names = [torch.cuda.get_device_name(index) for index in range(world_size)]
        hardware = contract["hardware"]
        memories = [
            torch.cuda.get_device_properties(index).total_memory
            for index in range(world_size)
        ]
        if any(capability != hardware["compute_capability"] for capability in capabilities):
            raise RuntimeError("Lumi-4B learning-pilot GPUs are not pinned sm_90")
        if len({tuple(capability) for capability in capabilities}) != 1:
            raise RuntimeError("Lumi-4B learning-pilot GPUs are not homogeneous")
        if len(set(names)) != 1 or any(
            hardware["gpu_model_contains"].lower() not in name.lower() for name in names
        ):
            raise RuntimeError("Lumi-4B learning pilot requires homogeneous H100 GPUs")
        if any(
            not hardware["minimum_memory_bytes"]
            <= memory
            <= hardware["maximum_memory_bytes"]
            for memory in memories
        ):
            raise RuntimeError("Lumi-4B learning pilot requires 80 GB H100 GPUs")
        host_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        free_disk = shutil.disk_usage(manifest_path.parent).free
        if host_memory < hardware["minimum_host_memory_bytes"]:
            raise RuntimeError("Lumi-4B learning pilot has insufficient host RAM")
        if free_disk < hardware["minimum_free_disk_bytes"]:
            raise RuntimeError("Lumi-4B learning pilot has insufficient free disk")
        try:
            import flash_attn
        except ImportError as error:
            raise RuntimeError("FlashAttention is unavailable for pilot planning") from error
        runtime.update(
            {
                "cuda_capabilities": capabilities,
                "flash_attention_version": flash_attn.__version__,
                "gpu_names": names,
                "gpu_total_memory_bytes": memories,
                "host_memory_bytes": host_memory,
                "manifest_volume_free_bytes": free_disk,
            }
        )

    manifest_sha256 = _sha256(manifest_path)
    identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        backend="nccl",
        world_size=world_size,
        strategy=run["strategy"],
        per_rank_batch_size=run["per_rank_batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=run["learning_rate"],
        weight_decay=run["weight_decay"],
        max_grad_norm=run["max_grad_norm"],
        keep_checkpoints=run["keep_checkpoints"],
        seed=run["seed"],
        gradient_checkpointing=run["gradient_checkpointing"],
        lr_schedule=run["lr_schedule"],
        warmup_steps=run["warmup_steps"],
        schedule_steps=run["schedule_steps"],
        min_learning_rate_ratio=run["min_learning_rate_ratio"],
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
        world_size
        * run["per_rank_batch_size"]
        * gradient_accumulation_steps
        * manifest["packing"]["sequence_length"]
    )
    plan: dict[str, Any] = {
        "authorization_template": {
            "approved_through_step": run["resume_through_step"],
            "approved_token_cap": corpus["fixture_tokens"],
            "full_4b_training": False,
            "owner_acknowledgement": PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT,
            "paid_gpu": False,
            "run_fingerprint": run_fingerprint,
            "run_kind": PAID_LEARNING_PILOT_RUN_KIND,
            "schema_version": DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION,
            "source_commit": normalized_commit,
            "status": "planning_only",
            "strategy": run["strategy"],
            "world_size": world_size,
        },
        "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        "created_at": _utc_now(),
        "data": {
            "build_fingerprint": manifest["build_fingerprint"],
            "document_markers": manifest["statistics"]["document_markers"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "sequences": manifest["statistics"]["sequences"],
            "tokens": manifest["statistics"]["tokens"],
        },
        "evaluation": {
            **evaluation,
            "heldout_build_fingerprint": heldout["build_fingerprint"],
            "heldout_manifest_path": str(heldout_manifest_path),
        },
        "global_tokens_per_step": global_tokens_per_step,
        "model": {
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "parameters": contract["model"]["parameters"],
        },
        "run_fingerprint": run_fingerprint,
        "run_identity": identity,
        "runtime": runtime,
        "selected_topology": {
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": world_size,
        },
        "schema_version": LEARNING_PILOT_SCHEMA_VERSION,
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
    """Convert one immutable plan into one narrowly bounded owner artifact."""

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    expected_plan_sha256 = plan.get("plan_sha256")
    if expected_plan_sha256 != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("Lumi-4B learning-pilot plan hash changed")
    template = plan.get("authorization_template", {})
    if template.get("paid_gpu") is not False or template.get("full_4b_training") is not False:
        raise ValueError("Lumi-4B learning-pilot authorization template is unsafe")
    if not provider.strip() or not instance_type.strip():
        raise ValueError("provider and instance type are required")
    if hourly_rate_usd <= 0 or hourly_rate_usd > 60:
        raise ValueError("hourly rate is outside the learning-pilot safety range")
    max_allowed_runtime = _max_runtime_minutes(plan)
    if max_runtime_minutes < 30 or max_runtime_minutes > max_allowed_runtime:
        raise ValueError(
            "learning-pilot runtime must be between 30 and "
            f"{max_allowed_runtime} minutes"
        )
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
    """Validate every material owner boundary before the paid runner starts."""

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    expected_plan_sha256 = plan.get("plan_sha256")
    if expected_plan_sha256 != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("Lumi-4B learning-pilot plan hash changed")
    template = plan.get("authorization_template", {})
    exact_fields = (
        "approved_through_step",
        "approved_token_cap",
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
            raise ValueError(f"Lumi-4B learning-pilot authorization changed {field}")
    if authorization.get("status") != "owner_approved":
        raise ValueError("Lumi-4B learning pilot is not owner-approved")
    if authorization.get("paid_gpu") is not True:
        raise ValueError("Lumi-4B learning-pilot paid boundary is not explicit")
    if authorization.get("full_4b_training") is not False:
        raise ValueError("Lumi-4B learning pilot cannot authorize full training")
    if authorization.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("Lumi-4B learning-pilot authorization targets another plan")
    runtime = authorization.get("max_runtime_minutes")
    max_allowed_runtime = _max_runtime_minutes(plan)
    if (
        not isinstance(runtime, int)
        or isinstance(runtime, bool)
        or not 30 <= runtime <= max_allowed_runtime
    ):
        raise ValueError("Lumi-4B learning-pilot runtime cap is invalid")
    hourly_rate = authorization.get("hourly_rate_usd")
    if not isinstance(hourly_rate, (int, float)) or isinstance(hourly_rate, bool):
        raise ValueError("Lumi-4B learning-pilot hourly rate is invalid")
    if not 0 < float(hourly_rate) <= 60:
        raise ValueError("Lumi-4B learning-pilot hourly rate is unsafe")
    if not str(authorization.get("provider", "")).strip():
        raise ValueError("Lumi-4B learning-pilot provider is missing")
    if not str(authorization.get("instance_type", "")).strip():
        raise ValueError("Lumi-4B learning-pilot instance type is missing")
    return authorization
