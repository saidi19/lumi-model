"""Metadata-budget SFT launch-contract successor; no transport, ML or execution.

This is a control-plane development artifact, NOT a resource-enforcing launcher.
The legacy worker initializes CUDA and reads inputs before its generic approval
check. Its argument previews must not be dispatched until a separately reviewed
SFT worker/supervisor successor exists. Even fully populated inputs cannot enable
execution here. Receipt references are cross-checked, not opened or authenticated.
The new encoded-metadata read cap is mandatory, positive and separately approved;
v0.1 contracts/approvals cannot authorize this extended budget. Draft caps and
approvals remain unbound. No allowance is inferred from binary/index caps.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import PurePosixPath
from typing import Any

SCHEMA = "lumi.sft-launch-contract.v0.2"
AUTH_SCHEMA = "lumi.sft-execution-authorization.v0.2"
REPORT_SCHEMA = "lumi.sft-launch-contract-report.v0.2"
SCOPE = "private_research_nonproduction"
BASE_SHA = "1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172"
DATA_SHA = "281fba028c58f5bc0fd489c1d70694e54d37d5dffd09dcc79676d42029497d33"
REGISTRY_SHA = "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb"
FIVE_HASHES = (
    "inventory_sha256",
    "manifest_file_sha256",
    "manifest_internal_sha256",
    "metadata_inventory_sha256",
    "seal_internal_sha256",
)
PROFILE_KEYS = (
    "world_size",
    "strategy",
    "device",
    "backend",
    "precision",
    "attention_implementation",
    "record_limit",
    "per_rank_batch_size",
    "gradient_accumulation_steps",
    "context_window",
    "target_steps",
    "learning_rate",
    "weight_decay",
    "maximum_gradient_norm",
    "lr_schedule",
    "warmup_steps",
    "minimum_learning_rate_ratio",
    "optimizer_profile",
    "checkpoint_steps",
    "seed",
    "pad_to_multiple_of",
    "gradient_sync_mode",
    "sequence",
)
PROHIBITIONS = {
    key: True
    for key in (
        "no_external_network_or_s3_access",
        "no_remote_or_instance_action",
        "no_inference",
        "no_source_mutation",
        "no_registry_mutation",
        "no_promotion",
        "no_production_seal",
        "no_public_release",
        "no_checkpoint_pruning",
        "no_retry_after_started_invocation",
        "no_raw_training_or_exception_text",
    )
}
SLOTS = (
    "target",
    "import_receipt",
    "relocation_receipt",
    "cuda_acceptance",
    "resource_caps",
    "execution_authorization",
)
ALWAYS_BLOCKED = (
    "sft_worker_and_resource_enforcing_supervisor_not_integrated",
    "referenced_target_receipts_not_verified_by_this_metadata_compiler",
    "execution_disabled_in_local_development_v2",
)
CAP_FIELDS = (
    "maximum_combined_runtime_seconds",
    "maximum_cost_microusd",
    "total_host_hourly_rate_microusd",
    "maximum_host_ram_bytes",
    "maximum_per_gpu_memory_bytes",
    "maximum_private_write_bytes",
    "maximum_checkpoint_read_bytes",
    "maximum_encoded_binary_read_bytes",
    "maximum_encoded_index_read_bytes",
    "maximum_encoded_metadata_read_bytes",
    "maximum_log_bytes",
)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("noncanonical_json") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("nonfinite_json_number")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json") from exc
    require(type(value) is dict, "json_object_required")
    canonical(value)
    return value


def keys(value: Any, expected: set[str], code: str) -> None:
    require(type(value) is dict and set(value) == expected, code)


def sha(value: Any) -> None:
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid_sha256"
    )


def positive_int(value: Any) -> None:
    require(type(value) is int and 0 < value <= 2**63 - 1, "positive_integer_required")


def label(value: Any) -> None:
    require(
        type(value) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is not None,
        "invalid_identity",
    )


def absolute_path(value: Any) -> None:
    require(
        type(value) is str
        and len(value) <= 4096
        and value.startswith("/")
        and not value.startswith("//"),
        "absolute_target_path_required",
    )
    require(all(ord(c) >= 32 and ord(c) != 127 for c in value), "path_control_character")
    path = PurePosixPath(value)
    require(
        str(path) == value and ".." not in path.parts and len(path.parts) >= 3, "unsafe_target_path"
    )


def make_draft(base: dict, cpu_input: dict, readiness_draft: dict) -> dict:
    """Use already hash-bound metadata; never substitute the CPU tiny checkpoint."""
    require(
        base["model_id"] == "lumi-4b-base" and base["checkpoint_stage"] == "base",
        "final_base_identity_changed",
    )
    require(
        base["checkpoint"]["step"] == 76294
        and base["checkpoint"]["training_tokens"] == 80_000_000_000,
        "final_base_boundary_changed",
    )
    require(base["s3_backup"]["status"] == "verified", "historical_backup_not_verified")
    for name in FIVE_HASHES:
        sha(cpu_input[name])
    profile = {name: copy.deepcopy(readiness_draft[name]) for name in PROFILE_KEYS}
    return {
        "schema_version": SCHEMA,
        "scope": SCOPE,
        "operation": "sft_gpu_smoke_and_fresh_resume_only",
        "execution_enabled": False,
        "production_eligible": False,
        "source": {
            "final_base_binding_sha256": BASE_SHA,
            "historical_binding": copy.deepcopy(base),
            "remote_availability_rechecked": False,
        },
        "data": {
            "data_contract_sha256": DATA_SHA,
            "source_registry_sha256": REGISTRY_SHA,
            "origin_encoded_directory": cpu_input["encoded_directory"],
            "origin_loader_hashes": {name: cpu_input[name] for name in FIVE_HASHES},
            "origin_metadata_binding_is_portable": False,
        },
        "profile": profile,
        "prohibitions": copy.deepcopy(PROHIBITIONS),
        "execution_adapter": None,
        **{slot: None for slot in SLOTS},
    }


def geometry(profile: dict) -> dict:
    world, records = profile["world_size"], profile["record_limit"]
    for name in (
        "world_size",
        "record_limit",
        "per_rank_batch_size",
        "gradient_accumulation_steps",
        "context_window",
        "target_steps",
    ):
        positive_int(profile[name])
    require(world >= 2 and records <= 12, "bounded_distributed_geometry_changed")
    batch = world * profile["per_rank_batch_size"]
    require(batch <= records, "selected_prefix_cannot_fill_global_batch")
    sequence = profile["sequence"]
    require(
        type(sequence) is list
        and canonical(sequence)
        == canonical(
            [
                {"run": "reference", "from_step": 0, "stop_step": 2, "resume": False},
                {"run": "resumed", "from_step": 0, "stop_step": 1, "resume": False},
                {"run": "resumed", "from_step": 1, "stop_step": 2, "resume": True},
            ]
        ),
        "smoke_resume_sequence_changed",
    )
    steps = sum(item["stop_step"] - item["from_step"] for item in sequence)
    rows = len(sequence) * world * records
    return {
        "sequential_invocations": len(sequence),
        "global_optimizer_steps": steps,
        "rank_optimizer_steps": world * steps,
        "logical_sample_events": steps * batch * profile["gradient_accumulation_steps"],
        "maximum_unique_train_records": records,
        "rank_local_input_loads": len(sequence) * world,
        "maximum_encoded_index_rows": rows,
        "maximum_selected_binary_bytes": rows * profile["context_window"] * 8,
        "maximum_selected_index_bytes": rows * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }


def authorization_subject(contract: dict) -> str:
    return digest({k: v for k, v in contract.items() if k != "execution_authorization"})


def validate_authorization(
    auth: dict, contract: dict, expected_sha: str | None, now_unix: int | None
) -> bool:
    """Validate a claimed approval, never mint one or authenticate a signer."""
    keys(
        auth,
        {
            "schema_version",
            "status",
            "run_kind",
            "scope",
            "subject_sha256",
            "owner_approval_text_sha256",
            "not_before_unix",
            "expires_unix",
            "gpu_execution_approved",
            "production_eligible",
        },
        "authorization_schema",
    )
    require(
        auth["schema_version"] == AUTH_SCHEMA
        and auth["status"] == "owner_approved"
        and auth["run_kind"] == "sft_gpu_smoke_and_fresh_resume_only"
        and auth["scope"] == SCOPE,
        "not_sft_specific_authorization",
    )
    require(
        auth["gpu_execution_approved"] is True and auth["production_eligible"] is False,
        "authorization_scope_changed",
    )
    sha(auth["subject_sha256"])
    sha(auth["owner_approval_text_sha256"])
    require(
        auth["subject_sha256"] == authorization_subject(contract), "authorization_subject_mismatch"
    )
    positive_int(auth["not_before_unix"])
    positive_int(auth["expires_unix"])
    require(auth["not_before_unix"] < auth["expires_unix"], "authorization_time_window")
    if expected_sha is None or now_unix is None:
        return False
    sha(expected_sha)
    positive_int(now_unix)
    require(digest(auth) == expected_sha, "authorization_hash_mismatch")
    require(auth["not_before_unix"] <= now_unix < auth["expires_unix"], "authorization_not_current")
    return True


def validate_target(target: dict, profile: dict) -> None:
    keys(
        target,
        {
            "host_id",
            "gpu_uuids",
            "runtime_lock_sha256",
            "sft_source_commit",
            "python_path",
            "checkpoint_directory",
            "encoded_directory",
            "private_output_root",
            "authorization_path",
            "local_rendezvous",
        },
        "target_schema",
    )
    label(target["host_id"])
    sha(target["runtime_lock_sha256"])
    require(
        type(target["sft_source_commit"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", target["sft_source_commit"]) is not None,
        "invalid_sft_source_commit",
    )
    rendezvous = target["local_rendezvous"]
    keys(rendezvous, {"address", "port", "node_count", "maximum_restarts"}, "rendezvous_schema")
    require(
        rendezvous["address"] == "127.0.0.1"
        and type(rendezvous["port"]) is int
        and 1024 <= rendezvous["port"] <= 65535
        and type(rendezvous["node_count"]) is int
        and rendezvous["node_count"] == 1
        and type(rendezvous["maximum_restarts"]) is int
        and rendezvous["maximum_restarts"] == 0,
        "nonlocal_or_retrying_rendezvous",
    )
    require(
        type(target["gpu_uuids"]) is list and len(target["gpu_uuids"]) == profile["world_size"],
        "gpu_count_mismatch",
    )
    for item in target["gpu_uuids"]:
        label(item)
    require(len(set(target["gpu_uuids"])) == len(target["gpu_uuids"]), "duplicate_gpu_identity")
    path_fields = (
        "python_path",
        "checkpoint_directory",
        "encoded_directory",
        "private_output_root",
        "authorization_path",
    )
    for field in path_fields:
        absolute_path(target[field])
    paths = [PurePosixPath(target[field]) for field in path_fields]
    require(
        all(
            not (left.is_relative_to(right) or right.is_relative_to(left))
            for i, left in enumerate(paths)
            for right in paths[i + 1 :]
        ),
        "target_paths_overlap",
    )


def validate_receipts(contract: dict) -> None:
    target, imported = contract["target"], contract["import_receipt"]
    relocated, acceptance = contract["relocation_receipt"], contract["cuda_acceptance"]
    for receipt in (imported, relocated, acceptance):
        if receipt is not None:
            require(target is not None, "receipt_requires_target")
    if imported is not None:
        keys(
            imported,
            {
                "file_sha256",
                "internal_sha256",
                "verification_file_sha256",
                "host_id",
                "checkpoint_directory",
                "real_4b_import",
                "original_dcp_manifest_sha256",
                "hf_checksum_manifest_sha256",
                "tokenizer_manifest_sha256",
                "final_base_binding_sha256",
            },
            "import_receipt_schema",
        )
        for name in imported:
            if name.endswith("sha256"):
                sha(imported[name])
        base = contract["source"]["historical_binding"]
        require(imported["real_4b_import"] is True, "synthetic_import_cannot_qualify_real_base")
        require(
            imported["host_id"] == target["host_id"]
            and imported["checkpoint_directory"] == target["checkpoint_directory"],
            "import_target_mismatch",
        )
        require(
            imported["original_dcp_manifest_sha256"] == base["checkpoint"]["manifest_sha256"]
            and imported["tokenizer_manifest_sha256"] == base["tokenizer"]["manifest_sha256"]
            and imported["final_base_binding_sha256"] == BASE_SHA,
            "import_lineage_mismatch",
        )
        require(
            imported["hf_checksum_manifest_sha256"] != imported["original_dcp_manifest_sha256"],
            "dcp_manifest_is_not_hf_manifest",
        )
    if relocated is not None:
        keys(
            relocated,
            {
                "file_sha256",
                "internal_sha256",
                "verification_file_sha256",
                "host_id",
                "encoded_directory",
                "content_verified",
                "private_modes_verified",
                "origin_metadata_inventory_sha256",
                "loader_hashes",
            },
            "relocation_receipt_schema",
        )
        for field in (
            "file_sha256",
            "internal_sha256",
            "verification_file_sha256",
            "origin_metadata_inventory_sha256",
        ):
            sha(relocated[field])
        require(
            relocated["content_verified"] is True and relocated["private_modes_verified"] is True,
            "relocation_not_content_verified",
        )
        require(
            relocated["host_id"] == target["host_id"]
            and relocated["encoded_directory"] == target["encoded_directory"],
            "relocation_target_mismatch",
        )
        keys(relocated["loader_hashes"], set(FIVE_HASHES), "five_loader_hashes_required")
        origin = contract["data"]["origin_loader_hashes"]
        for field in FIVE_HASHES:
            sha(relocated["loader_hashes"][field])
            if field != "metadata_inventory_sha256":
                require(
                    relocated["loader_hashes"][field] == origin[field],
                    "relocated_content_binding_changed",
                )
        require(
            relocated["origin_metadata_inventory_sha256"] == origin["metadata_inventory_sha256"],
            "relocation_origin_metadata_mismatch",
        )
        require(
            relocated["loader_hashes"]["metadata_inventory_sha256"]
            != origin["metadata_inventory_sha256"],
            "origin_metadata_not_portable",
        )
    if acceptance is not None:
        keys(
            acceptance,
            {"criteria_file_sha256", "host_id", "runtime_lock_sha256", "criteria"},
            "cuda_acceptance_schema",
        )
        sha(acceptance["criteria_file_sha256"])
        require(
            acceptance["host_id"] == target["host_id"]
            and acceptance["runtime_lock_sha256"] == target["runtime_lock_sha256"],
            "cuda_criteria_target_mismatch",
        )
        require(
            acceptance["criteria"]
            == [
                "masking",
                "precision",
                "memory",
                "import",
                "fresh_process_resume",
                "fault_handling",
            ],
            "cuda_criteria_incomplete",
        )


def validate_caps(caps: dict, counts: dict) -> None:
    keys(
        caps,
        set(CAP_FIELDS)
        | {
            "maximum_invocations",
            "maximum_global_optimizer_steps",
            "maximum_external_network_bytes",
            "maximum_inference_calls",
        },
        "resource_caps_schema",
    )
    for field in CAP_FIELDS:
        positive_int(caps[field])
    for field, expected in (
        ("maximum_invocations", counts["sequential_invocations"]),
        ("maximum_global_optimizer_steps", counts["global_optimizer_steps"]),
        ("maximum_external_network_bytes", 0),
        ("maximum_inference_calls", 0),
    ):
        require(type(caps[field]) is int and caps[field] == expected, "operation_cap_changed")
    # Integer micro-USD avoids binary floating point, NaN, and under-rounding.
    minimum_cost = (
        caps["total_host_hourly_rate_microusd"] * caps["maximum_combined_runtime_seconds"] + 3599
    ) // 3600
    require(caps["maximum_cost_microusd"] >= minimum_cost, "runtime_exceeds_cost_cap")
    require(
        caps["maximum_encoded_binary_read_bytes"] >= counts["maximum_selected_binary_bytes"]
        and caps["maximum_encoded_index_read_bytes"] >= counts["maximum_selected_index_bytes"],
        "rank_multiplied_data_read_budget_too_small",
    )
    require(
        caps["maximum_log_bytes"] <= caps["maximum_private_write_bytes"],
        "log_cap_exceeds_private_write_cap",
    )


def argument_previews(contract: dict) -> list[list[str]] | None:
    """Data-only worker options, deliberately no Python/torchrun/script prefix."""
    target, imported, relocated = (
        contract[k] for k in ("target", "import_receipt", "relocation_receipt")
    )
    caps = contract["resource_caps"]
    if any(item is None for item in (target, imported, relocated, caps)):
        return None
    profile = contract["profile"]
    options = {
        "source-checkpoint": target["checkpoint_directory"],
        "source-checkpoint-manifest-sha256": imported["hf_checksum_manifest_sha256"],
        "encoded-shards": target["encoded_directory"],
        "encoded-shards-record-limit": profile["record_limit"],
        "authorization": target["authorization_path"],
        "source-commit": target["sft_source_commit"],
        "timeout-seconds": min(1800, caps["maximum_combined_runtime_seconds"]),
    }
    for flag, field in (
        ("inventory", "inventory_sha256"),
        ("manifest-file", "manifest_file_sha256"),
        ("manifest", "manifest_internal_sha256"),
        ("metadata-inventory", "metadata_inventory_sha256"),
        ("seal", "seal_internal_sha256"),
    ):
        options[f"encoded-shards-{flag}-sha256"] = relocated["loader_hashes"][field]
    for field in (
        "strategy",
        "device",
        "backend",
        "precision",
        "attention_implementation",
        "per_rank_batch_size",
        "gradient_accumulation_steps",
        "target_steps",
        "learning_rate",
        "weight_decay",
        "lr_schedule",
        "warmup_steps",
        "optimizer_profile",
        "checkpoint_steps",
        "seed",
        "pad_to_multiple_of",
        "gradient_sync_mode",
    ):
        options[field.replace("_", "-")] = profile[field]
    options["max-grad-norm"] = profile["maximum_gradient_norm"]
    options["min-learning-rate-ratio"] = profile["minimum_learning_rate_ratio"]
    result = []
    for invocation in profile["sequence"]:
        values = {
            **options,
            "stop-after-step": invocation["stop_step"],
            "output": str(PurePosixPath(target["private_output_root"]) / invocation["run"]),
        }
        argv = [part for key, value in sorted(values.items()) for part in ("--" + key, str(value))]
        # A later full-model digest needs its own explicit I/O budget. No pruning.
        argv.append("--skip-final-model-digest")
        if invocation["resume"]:
            argv.append("--resume")
        result.append(argv)
    return result


def analyze(
    contract: dict,
    reference: dict,
    *,
    expected_authorization_sha256: str | None = None,
    now_unix: int | None = None,
) -> dict:
    keys(contract, set(reference), "contract_schema")
    canonical(contract)
    require(
        contract["schema_version"] == SCHEMA and reference["schema_version"] == SCHEMA,
        "frozen_contract_field_changed",
    )
    for field in set(reference) - set(SLOTS):
        require(
            canonical(contract[field]) == canonical(reference[field]),
            "frozen_contract_field_changed",
        )
    require(
        contract["execution_enabled"] is False and contract["execution_adapter"] is None,
        "execution_forbidden_in_v2",
    )
    for value in contract["profile"].values():
        if type(value) is float:
            require(math.isfinite(value), "nonfinite_profile")
    counts = geometry(contract["profile"])
    if contract["target"] is not None:
        validate_target(contract["target"], contract["profile"])
    validate_receipts(contract)
    if contract["resource_caps"] is not None:
        validate_caps(contract["resource_caps"], counts)
    trusted_approval = False
    if contract["execution_authorization"] is not None:
        require(
            all(contract[k] is not None for k in SLOTS if k != "execution_authorization"),
            "approval_requires_fully_bound_inputs",
        )
        trusted_approval = validate_authorization(
            contract["execution_authorization"], contract, expected_authorization_sha256, now_unix
        )
    pending = [field + "_unbound" for field in SLOTS if contract[field] is None]
    if contract["execution_authorization"] is not None and not trusted_approval:
        pending.append("execution_approval_not_externally_bound")
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_local_contract_validation_execution_blocked",
        "contract_sha256": digest(contract),
        "authorization_subject_sha256": authorization_subject(contract),
        "execution_permitted": False,
        "gpu_launch_ready": False,
        "production_eligible": False,
        "pending_gates": pending + list(ALWAYS_BLOCKED),
        "geometry": counts,
        "worker_option_previews": argument_previews(contract),
        "executable_argv": None,
        "previews_are_not_dispatchable": True,
        "resource_caps_enforced_by_this_compiler": False,
        "receipt_contents_verified": False,
    }


def deny_execution() -> None:
    """Call before any path access; no flag, approval object, or filled slot bypasses it."""
    raise PermissionError("execution_disabled_in_local_development_v2")
