"""Portable artifact and CUDA preflight for the paid Lumi pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .configuration_lumi import LumiConfig
from .packed_data import validate_packed_manifest
from .parameter_count import parameter_breakdown
from .scaling_pilot import SCALING_SELECTION_MODE, load_scaling_pilot_plan

CLOUD_PILOT_SCHEMA_VERSION = 1
SCALING_CLOUD_PILOT_SCHEMA_VERSION = 2


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


def _nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_file(plan_path: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must stay inside the plan directory")
    path = plan_path.parent / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _pinned_contract(plan_path: Path, specification: Any, *, name: str) -> tuple[Path, str]:
    if not isinstance(specification, dict):
        raise ValueError(f"contracts.{name} must be an object")
    path = _relative_file(
        plan_path,
        specification.get("path"),
        name=f"contracts.{name}.path",
    )
    expected_hash = _sha256(
        specification.get("sha256"),
        name=f"contracts.{name}.sha256",
    )
    if _sha256_file(path) != expected_hash:
        raise ValueError(f"contracts.{name} SHA-256 changed")
    return path, expected_hash


def _validate_scaling_cloud_plan(plan_path: Path, plan: dict[str, Any]) -> None:
    if plan.get("pilot_kind") != "scaling_1b":
        raise ValueError("schema-v2 cloud plans must use pilot_kind scaling_1b")

    artifact = plan.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be an object")
    for field in (
        "build_fingerprint",
        "index_build_fingerprint",
        "index_manifest_sha256",
        "manifest_sha256",
        "selection_proof_sha256",
        "selection_state_sha256",
    ):
        _sha256(artifact.get(field), name=f"artifact.{field}")
    integer_fields = (
        "documents_seen",
        "documents_selected",
        "heldout_exact_documents",
        "heldout_family_documents",
        "manifest_bytes",
        "selection_seed",
        "sequence_length",
        "sequences",
        "sequences_per_shard",
        "shard_bytes",
        "shards",
        "source_strata",
        "target_tokens_requested",
        "token_bytes",
        "tokens",
    )
    for field in integer_fields:
        _positive_int(artifact.get(field), name=f"artifact.{field}")
    if artifact.get("purpose") != "final":
        raise ValueError("schema-v2 cloud plans require the final packed artifact")
    if (
        artifact.get("dtype") != "uint16"
        or artifact.get("endianness") != "little"
        or artifact.get("layout") != "flat_fixed_sequences"
        or artifact["token_bytes"] != 2
    ):
        raise ValueError("the 65K 1B artifact must use little-endian uint16 tokens")
    if artifact.get("selection_mode") != SCALING_SELECTION_MODE:
        raise ValueError("the 1B cloud artifact changed its selection mode")
    if artifact["tokens"] != artifact["sequences"] * artifact["sequence_length"]:
        raise ValueError("artifact token count is not sequence aligned")
    if artifact["shard_bytes"] != artifact["tokens"] * artifact["token_bytes"]:
        raise ValueError("artifact shard-byte total does not match its tokens")
    if artifact["documents_selected"] > artifact["documents_seen"]:
        raise ValueError("artifact selects more documents than it scanned")
    if artifact["heldout_family_documents"] < artifact["heldout_exact_documents"]:
        raise ValueError("held-out family exclusions cannot be smaller than exact exclusions")
    leakage = artifact.get("leakage")
    if not isinstance(leakage, dict):
        raise ValueError("artifact.leakage must be an object")
    expected_leakage_fields = {
        "content_sha256_overlap",
        "document_id_overlap",
        "near_duplicate_sha256_overlap",
    }
    if set(leakage) != expected_leakage_fields:
        raise ValueError("artifact.leakage fields changed")
    for field in sorted(expected_leakage_fields):
        if _nonnegative_int(leakage.get(field), name=f"artifact.leakage.{field}") != 0:
            raise ValueError("the 1B cloud artifact must keep zero held-out leakage")

    model = plan.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    _relative_file(plan_path, model.get("config"), name="model.config")
    _sha256(model.get("config_sha256"), name="model.config_sha256")
    parameters = _positive_int(model.get("parameters"), name="model.parameters")

    contracts = plan.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("contracts must be an object")
    scaling_path, _ = _pinned_contract(
        plan_path,
        contracts.get("scaling_pilot"),
        name="scaling_pilot",
    )
    data_path, _ = _pinned_contract(
        plan_path,
        contracts.get("scaling_data"),
        name="scaling_data",
    )
    scaling = load_scaling_pilot_plan(scaling_path)
    data_contract = _read_json(data_path)
    if data_contract.get("schema_version") != 1:
        raise ValueError("unsupported scaling-data contract schema version")

    scaling_model = scaling["model"]
    expected_model = {
        "config": scaling_model["config"],
        "config_sha256": scaling_model["config_sha256"],
        "parameters": scaling_model["parameters"],
    }
    if model != expected_model:
        raise ValueError("cloud model does not match the scaling-pilot contract")
    selection = scaling["data"]["selection"]
    expected_artifact_values = {
        "documents_seen": scaling["data"]["source_sample"]["documents"],
        "heldout_exact_documents": scaling["data"]["heldout"]["documents"],
        "selection_mode": selection["mode"],
        "selection_seed": selection["seed"],
        "sequence_length": scaling["data"]["sequence_length"],
        "source_strata": selection["source_strata"],
        "target_tokens_requested": selection["target_tokens_requested"],
        "tokens": selection["target_tokens_effective"],
    }
    for field, expected in expected_artifact_values.items():
        if artifact.get(field) != expected:
            raise ValueError(f"artifact.{field} does not match the scaling-pilot contract")
    final_data = data_contract.get("final", {})
    packing_data = data_contract.get("packing", {})
    expected_data_values = {
        "sequences_per_shard": final_data.get("sequences_per_shard"),
        "shards": final_data.get("expected_shards"),
        "sequence_length": packing_data.get("sequence_length"),
        "target_tokens_requested": final_data.get("target_tokens_requested"),
        "tokens": final_data.get("target_tokens_effective"),
    }
    for field, expected in expected_data_values.items():
        if artifact.get(field) != expected:
            raise ValueError(f"artifact.{field} does not match the scaling-data contract")

    runtime = plan.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    if runtime.get("requires_bf16") is not True:
        raise ValueError("the 1B cloud runtime requires BF16")
    for field in (
        "minimum_cuda_devices",
        "minimum_cuda_memory_bytes",
        "minimum_host_memory_bytes",
    ):
        _positive_int(runtime.get(field), name=f"runtime.{field}")
    capability = runtime.get("minimum_compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in capability
        )
    ):
        raise ValueError("runtime.minimum_compute_capability must contain two integers")

    storage = plan.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("storage must be an object")
    checkpoint_bytes = _positive_int(
        storage.get("checkpoint_bytes_per_parameter"),
        name="storage.checkpoint_bytes_per_parameter",
    )
    peak_copies = _positive_int(
        storage.get("peak_checkpoint_copies"),
        name="storage.peak_checkpoint_copies",
    )
    fixed_overhead = _nonnegative_int(
        storage.get("fixed_overhead_bytes"),
        name="storage.fixed_overhead_bytes",
    )
    minimum_free = _positive_int(
        storage.get("minimum_free_disk_bytes"),
        name="storage.minimum_free_disk_bytes",
    )
    derived_minimum = (
        parameters * checkpoint_bytes * peak_copies + artifact["shard_bytes"] + fixed_overhead
    )
    if minimum_free < derived_minimum:
        raise ValueError("storage.minimum_free_disk_bytes is below the checkpoint-aware floor")

    expected_approval = {
        "full_1b_stage": False,
        "full_4b_training": False,
        "paid_gpu_launch": False,
    }
    if plan.get("approval") != expected_approval:
        raise ValueError("cloud preflight cannot pre-approve paid or full training")


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
    checkpoint_steps = _positive_int(full.get("checkpoint_steps"), name="full.checkpoint_steps")
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
    schema_version = plan.get("schema_version")
    if schema_version == SCALING_CLOUD_PILOT_SCHEMA_VERSION:
        _validate_scaling_cloud_plan(plan_path, plan)
        return plan
    if schema_version != CLOUD_PILOT_SCHEMA_VERSION:
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


def _verify_auxiliary_file(manifest_path: Path, specification: Any, *, label: str) -> None:
    if not isinstance(specification, dict):
        raise ValueError(f"packed manifest {label} specification is missing")
    name = specification.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"packed manifest has an invalid {label} filename")
    expected_bytes = _positive_int(specification.get("bytes"), name=f"packed {label} bytes")
    expected_hash = _sha256(specification.get("sha256"), name=f"packed {label} SHA-256")
    path = manifest_path.parent / name
    if not path.is_file():
        raise FileNotFoundError(f"packed {label} is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"packed {label} size changed: {path}")
    if _sha256_file(path) != expected_hash:
        raise ValueError(f"packed {label} SHA-256 changed: {path}")


def _actual_scaling_artifact(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _verify_auxiliary_file(
        manifest_path,
        manifest.get("selection_proof"),
        label="selection proof",
    )
    _verify_auxiliary_file(
        manifest_path,
        manifest.get("selection_state"),
        label="selection state",
    )
    packing = manifest.get("packing", {})
    token_format = manifest.get("format", {})
    selection = manifest.get("selection", {})
    statistics = manifest.get("statistics", {})
    leakage = manifest.get("leakage", {})
    source_index = selection.get("source_index", {})
    if not all(
        isinstance(value, dict)
        for value in (packing, token_format, selection, statistics, leakage, source_index)
    ):
        raise ValueError("packed scaling manifest has malformed sections")
    if statistics.get("documents_selected") != statistics.get("documents_packed"):
        raise ValueError("selected and packed document counts differ")
    if packing.get("target_tokens_effective") != statistics.get("tokens"):
        raise ValueError("effective packed target does not match the shard total")
    selection_proof = manifest["selection_proof"]
    selection_state = manifest["selection_state"]
    return {
        "build_fingerprint": manifest.get("build_fingerprint"),
        "documents_seen": statistics.get("documents_seen"),
        "documents_selected": statistics.get("documents_selected"),
        "dtype": token_format.get("dtype"),
        "endianness": token_format.get("endianness"),
        "heldout_exact_documents": leakage.get("exact_documents_excluded"),
        "heldout_family_documents": leakage.get("family_documents_excluded"),
        "index_build_fingerprint": source_index.get("build_fingerprint"),
        "index_manifest_sha256": source_index.get("sha256"),
        "leakage": {
            "content_sha256_overlap": leakage.get("content_sha256_overlap"),
            "document_id_overlap": leakage.get("document_id_overlap"),
            "near_duplicate_sha256_overlap": leakage.get("near_duplicate_sha256_overlap"),
        },
        "layout": token_format.get("layout"),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": _sha256_file(manifest_path),
        "purpose": packing.get("purpose"),
        "selection_mode": selection.get("mode"),
        "selection_proof_sha256": selection_proof.get("sha256"),
        "selection_seed": selection.get("seed"),
        "selection_state_sha256": selection_state.get("sha256"),
        "sequence_length": packing.get("sequence_length"),
        "sequences": statistics.get("sequences"),
        "sequences_per_shard": packing.get("sequences_per_shard"),
        "shard_bytes": sum(item.get("bytes", 0) for item in manifest.get("shards", [])),
        "shards": len(manifest.get("shards", [])),
        "source_strata": len(selection.get("strata", [])),
        "target_tokens_requested": packing.get("target_tokens_requested"),
        "token_bytes": token_format.get("token_bytes"),
        "tokens": statistics.get("tokens"),
    }


def _host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(physical_pages, int):
        return None
    if page_size < 1 or physical_pages < 1:
        return None
    return page_size * physical_pages


def _runtime_report(
    *,
    require_cuda: bool,
    require_bf16: bool,
    workspace_path: Path | None = None,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    report = {
        "bf16_supported": bf16_supported,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "devices": devices,
        "torch_version": torch.__version__,
    }
    if requirements is None:
        return report
    if workspace_path is None or not workspace_path.is_dir():
        raise NotADirectoryError(f"cloud workspace does not exist: {workspace_path}")
    disk = shutil.disk_usage(workspace_path)
    host_memory = _host_memory_bytes()
    minimum_capability = tuple(requirements["minimum_compute_capability"])
    minimum_device_memory = requirements["minimum_cuda_memory_bytes"]
    qualified_devices = [
        device["index"]
        for device in devices
        if tuple(device["capability"]) >= minimum_capability
        and device["total_memory_bytes"] >= minimum_device_memory
    ]
    gates = {
        "bf16_supported": bf16_supported,
        "cuda_available": cuda_available,
        "minimum_cuda_devices": len(devices) >= requirements["minimum_cuda_devices"],
        "qualified_cuda_device": bool(qualified_devices),
        "minimum_host_memory": (
            host_memory is not None and host_memory >= requirements["minimum_host_memory_bytes"]
        ),
        "minimum_free_disk": disk.free >= requirements["minimum_free_disk_bytes"],
    }
    failed = [name for name, passed in gates.items() if not passed]
    if require_cuda and failed:
        raise RuntimeError("cloud runtime requirements failed: " + ", ".join(sorted(failed)))
    return {
        **report,
        "disk": {
            "free_bytes": disk.free,
            "total_bytes": disk.total,
            "used_bytes": disk.used,
        },
        "gates": gates,
        "host_memory_bytes": host_memory,
        "qualified_device_indices": qualified_devices,
        "requirements": requirements,
        "runtime_ready": not failed,
        "workspace": str(workspace_path.resolve()),
    }


def preflight_cloud_pilot(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    require_cuda: bool = False,
    verify_shards: bool = True,
    workspace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify portable pilot inputs and, on cloud, the CUDA/BF16 runtime."""

    plan_path = Path(plan_path)
    manifest_path = Path(manifest_path)
    tokenizer_path = Path(tokenizer_path)
    plan = load_cloud_pilot_plan(plan_path)
    artifact = plan["artifact"]
    model_plan = plan["model"]
    scaling_plan = plan.get("schema_version") == SCALING_CLOUD_PILOT_SCHEMA_VERSION
    if scaling_plan and require_cuda and not verify_shards:
        raise ValueError("cloud runtime preflight cannot skip shard hashes")

    if scaling_plan and manifest_path.stat().st_size != artifact["manifest_bytes"]:
        raise ValueError("packed manifest byte count does not match the cloud-pilot plan")
    if _sha256_file(manifest_path) != artifact["manifest_sha256"]:
        raise ValueError("packed manifest SHA-256 does not match the cloud-pilot plan")
    manifest = validate_packed_manifest(manifest_path, verify_hashes=verify_shards)
    actual_artifact = (
        _actual_scaling_artifact(manifest_path, manifest)
        if scaling_plan
        else {
            "build_fingerprint": manifest["build_fingerprint"],
            "sequence_length": manifest["packing"]["sequence_length"],
            "sequences": manifest["statistics"]["sequences"],
            "source_strata": len(manifest.get("selection", {}).get("strata", [])),
            "tokens": manifest["statistics"]["tokens"],
        }
    )
    for name, expected in artifact.items():
        if not scaling_plan and name == "manifest_sha256":
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
    if not scaling_plan:
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

    contracts = plan["contracts"]
    scaling_path, scaling_hash = _pinned_contract(
        plan_path,
        contracts["scaling_pilot"],
        name="scaling_pilot",
    )
    data_path, data_hash = _pinned_contract(
        plan_path,
        contracts["scaling_data"],
        name="scaling_data",
    )
    scaling = load_scaling_pilot_plan(scaling_path)
    storage_plan = plan["storage"]
    checkpoint_floor = (
        parameters
        * storage_plan["checkpoint_bytes_per_parameter"]
        * storage_plan["peak_checkpoint_copies"]
        + actual_artifact["shard_bytes"]
        + storage_plan["fixed_overhead_bytes"]
    )
    requirements = {
        **plan["runtime"],
        "minimum_free_disk_bytes": storage_plan["minimum_free_disk_bytes"],
    }
    workspace = manifest_path.parent if workspace_path is None else Path(workspace_path)
    runtime = _runtime_report(
        require_cuda=require_cuda,
        require_bf16=require_cuda and requirements["requires_bf16"],
        requirements=requirements,
        workspace_path=workspace,
    )
    return {
        "artifact": actual_artifact,
        "authorization": {
            **plan["approval"],
            "preflight_does_not_authorize_launch": True,
        },
        "contracts": {
            "scaling_data": {
                "path": contracts["scaling_data"]["path"],
                "sha256": data_hash,
            },
            "scaling_pilot": {
                "path": contracts["scaling_pilot"]["path"],
                "sha256": scaling_hash,
            },
        },
        "integrity": {
            "auxiliary_hashes_verified": True,
            "shard_hashes_verified": verify_shards,
            "tokenizer_hashes_verified": True,
        },
        "kind": plan["pilot_kind"],
        "model_config_sha256": model_plan["config_sha256"],
        "parameters": parameters,
        "plan_sha256": _sha256_file(plan_path),
        "runtime": runtime,
        "storage": {
            **storage_plan,
            "checkpoint_aware_floor_bytes": checkpoint_floor,
        },
        "status": "pass",
        "training": {
            "optimizer": scaling["optimizer"],
            "stages": scaling["stages"],
            "training": scaling["training"],
        },
    }
