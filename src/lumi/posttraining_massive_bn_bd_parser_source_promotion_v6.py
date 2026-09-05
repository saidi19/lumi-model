"""Materialize the preflight-approved MASSIVE bn-BD registry-v0.22 entry.

This contract is deliberately narrower than a candidate-mix audit.  It binds
the clean v0.5 two-pass preflight, preserves all 24 predecessor registry
entries byte-for-byte as JSON values, appends one non-production MASSIVE
source entry, and independently verifies the exact successor bytes.  It does
not read candidate or source rows, seal a dataset, or start training.
"""

from __future__ import annotations

import copy
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_massive_bn_bd_parser_owner_review_v4 as owner_v4
from . import posttraining_massive_bn_bd_parser_source_promotion_preflight_v5 as preflight_v5
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource, load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-massive-bn-bd-parser-source-promotion-plan.v0.6"
RECEIPT_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-source-promotion-receipt.v0.6"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-source-promotion-verification.v0.6"
)
PLAN_STATUS = "authorized_registry_v0_22_materialization_nonproduction"
RECEIPT_STATUS = (
    "PASS_massive_bn_bd_parser_registry_v0_22_materialized_nonproduction"
)
VERIFICATION_STATUS = (
    "PASS_verified_massive_bn_bd_parser_registry_v0_22_materialized_"
    "nonproduction"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-massive-bn-bd-parser-source-promotion-v0.6.json"
)
EXPECTED_AUTHORIZATION_TEXT = "ok shuru kor next step skip korish na kichu"
EXPECTED_AUTHORIZATION_SHA256 = (
    "38aed29f3bcc5b45c5c1fd88ed5e953656545141a88abe05569a523d432afe1a"
)
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_massive_bn_bd_parser_source_promotion_v6.py"
    ),
    "runner": "scripts/promote_sft_massive_bn_bd_parser_source_v6.py",
    "verifier": "scripts/verify_sft_massive_bn_bd_parser_source_promotion_v6.py",
    "test": "tests/test_massive_bn_bd_parser_source_promotion_v6.py",
}
EXPECTED_INPUTS = {
    "data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.1.json",
        "bytes": 1_990,
        "sha256": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    },
    "predecessor_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.21.json",
        "bytes": 28_405,
        "sha256": "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1",
    },
    "preflight_plan": {
        "path": (
            "configs/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
            "preflight-v0.5.json"
        ),
        "bytes": 7_352,
        "sha256": "1c7da43a44577a657697ebeeeb0e96080d907f251530a2d8d3733fa4c54fab5f",
    },
    "preflight_report": {
        "path": (
            "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
            "preflight-v0.5.json"
        ),
        "bytes": 5_710,
        "sha256": "1cbe58c9fb526a963f7b0497259f4caec41c52130dbb2ea7748f67b6ade009b2",
        "internal_sha256": (
            "5ebfc3fa87645d49c25dac4a29e38360b07942f2d74a8ae32a36c23151f04088"
        ),
    },
    "preflight_verification": {
        "path": (
            "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
            "preflight-verification-v0.5.json"
        ),
        "bytes": 1_740,
        "sha256": "fddab3e4d433374d364090486f8c7e633249f27daf326a33c3558da2a2ecde22",
        "internal_sha256": (
            "af9fe57a2fa05431b058c72f6cbdb5456017a29ef80f41e438b9cdc910630c0f"
        ),
    },
}
EXPECTED_OUTPUT = {
    "receipt_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-v0.6.json"
    ),
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.22.json",
    "verification_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
        "verification-v0.6.json"
    ),
}
EXPECTED_LIMITS = {
    "passes": 2,
    "maximum_candidate_row_reads": 0,
    "maximum_source_line_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_predecessor_registry_writes": 0,
    "maximum_successor_registry_writes": 1,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_seconds_per_process": 300,
    "maximum_runtime_seconds_total": 600,
    "maximum_network_bytes": 0,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "maximum_retries_after_started_pass": 0,
}
EXPECTED_AUTHORIZATION = {
    "bound_preflight_evidence_read_authorized": True,
    "append_only_source_registry_v0_22_materialization_authorized": True,
    "promotion_receipt_and_verification_write_authorized": True,
    "candidate_or_source_row_read_authorized": False,
    "candidate_or_source_mutation_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "full_candidate_mix_reaudit_authorized": False,
    "production_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "sft_training_authorized": False,
    "network_gpu_inference_or_remote_access_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_SUCCESSOR = {
    "bytes": 29_673,
    "canonical_sha256": (
        "02a8de18c8a22dd62ca77e4cc5f094ab7e60116999dea0b02f7d7f5e38c1a240"
    ),
    "file_sha256": "c472d2ca1fe16beff9700719d53a3f8b539073bf52a6a2fa297da192d944f6b8",
    "observed_at": "2026-08-31",
    "sources": 25,
}
EXPECTED_SOURCE = copy.deepcopy(preflight_v5.EXPECTED_NEW_SOURCE)
EXPECTED_CANDIDATE_PROJECTION = {
    "candidate_sources": 18,
    "records": 314_780,
    "train_supervised_tokens": 10_610_600,
    "validation_supervised_tokens": 628_604,
}
EXPECTED_RECEIPT_MARKER = (
    b'{"started":"massive_bn_bd_parser_source_promotion_v0.6"}\n'
)
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"massive_bn_bd_parser_source_promotion_verification_v0.6"}\n'
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PUBLIC_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "record_id",
    "reviewer_id",
    "source_text",
    "user_prompt",
}

canonical_json_bytes = owner_v4.canonical_json_bytes
json_bytes = owner_v4.json_bytes
sha256_bytes = owner_v4.sha256_bytes
sha256_value = owner_v4.sha256_value
file_sha256 = owner_v4.file_sha256
strict_json_bytes = owner_v4.strict_json_bytes
start_marker = owner_v4.start_marker
finish_marker = owner_v4.finish_marker


def _resolve_repo_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("MASSIVE source-promotion path escaped repository")
    return resolved


def _input_path(root: Path, name: str) -> Path:
    return _resolve_repo_path(root, str(EXPECTED_INPUTS[name]["path"]))


def _read_bound_bytes(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
            raise ValueError("MASSIVE source-promotion bound input metadata changed")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("MASSIVE source-promotion bound input content changed")
    return payload


def _read_input(root: Path, name: str) -> bytes:
    binding = EXPECTED_INPUTS[name]
    return _read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )


def _read_json_input(root: Path, name: str) -> Any:
    return strict_json_bytes(_read_input(root, name))


def _validate_internal_seal(
    value: Any,
    binding: Mapping[str, Any],
    seal_key: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("MASSIVE source-promotion sealed root changed")
    unsigned = {key: item for key, item in value.items() if key != seal_key}
    expected = binding["internal_sha256"]
    if value.get(seal_key) != expected or sha256_value(unsigned) != expected:
        raise ValueError("MASSIVE source-promotion internal seal changed")


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("raw or private source-promotion key retained")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    if resolved != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("MASSIVE source-promotion plan path changed")
    plan = strict_json_bytes(resolved.read_bytes())
    expected_keys = {
        "authorization",
        "authorization_sha256",
        "authorization_text",
        "implementation",
        "inputs",
        "limits",
        "output",
        "projected_registry",
        "runner",
        "runtime",
        "schema_version",
        "status",
        "test",
        "verifier",
    }
    expected_projection = {
        **EXPECTED_SUCCESSOR,
        "entry": EXPECTED_SOURCE,
        "predecessor_sha256": EXPECTED_INPUTS["predecessor_registry"]["sha256"],
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(EXPECTED_AUTHORIZATION_TEXT.encode())
        != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("projected_registry") != expected_projection
        or plan.get("runtime")
        != {
            "python_executable": "/Users/afnan/Documents/lumi-model/.venv/bin/python",
            "python_version": "3.12.13",
        }
    ):
        raise ValueError("MASSIVE source-promotion plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("MASSIVE source-promotion code binding changed")
        code_path = _resolve_repo_path(root, expected_path)
        if file_sha256(code_path) != binding["sha256"]:
            raise ValueError("MASSIVE source-promotion code hash changed")
    for name, binding in EXPECTED_INPUTS.items():
        info = _input_path(root, name).lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != binding["bytes"]
        ):
            raise ValueError("MASSIVE source-promotion input metadata changed")
    return plan


def _validate_preflight_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_v5.load_plan(_input_path(root, "preflight_plan"), root)
    report = _read_json_input(root, "preflight_report")
    verification = _read_json_input(root, "preflight_verification")
    _validate_internal_seal(
        report,
        EXPECTED_INPUTS["preflight_report"],
        "report_sha256",
    )
    _validate_internal_seal(
        verification,
        EXPECTED_INPUTS["preflight_verification"],
        "verification_sha256",
    )
    preflight_v5.validate_report(report)
    preflight_v5.validate_verification(verification)
    audit = report.get("cross_source_audit")
    projection = report.get("registry_projection")
    decision = verification.get("decision")
    if (
        report.get("status") != preflight_v5.PASS_STATUS
        or not isinstance(audit, Mapping)
        or audit.get("total_collision_rows") != 0
        or audit.get("retained_massive_rows") != 64
        or not isinstance(projection, Mapping)
        or projection.get("projected_file_bytes") != EXPECTED_SUCCESSOR["bytes"]
        or projection.get("projected_file_sha256")
        != EXPECTED_SUCCESSOR["file_sha256"]
        or projection.get("projected_canonical_sha256")
        != EXPECTED_SUCCESSOR["canonical_sha256"]
        or projection.get("materialized") is not False
        or projection.get("promotion_preflight_passed") is not True
        or report.get("registry_successor_materialized") is not False
        or report.get("production_eligible") is not False
        or verification.get("status") != preflight_v5.PASS_VERIFICATION_STATUS
        or verification.get("report_file_sha256")
        != EXPECTED_INPUTS["preflight_report"]["sha256"]
        or verification.get("report_internal_sha256")
        != EXPECTED_INPUTS["preflight_report"]["internal_sha256"]
        or not isinstance(decision, Mapping)
        or decision.get("cross_source_clean") is not True
        or decision.get("promotion_preflight_passed") is not True
        or decision.get("registry_successor_materialized") is not False
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("MASSIVE clean v0.5 preflight evidence changed")
    return report, verification


def _predecessor_registry(root: Path) -> dict[str, Any]:
    value = _read_json_input(root, "predecessor_registry")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "lumi.sft.sources.v0.1"
        or not isinstance(value.get("sources"), list)
        or len(value["sources"]) != 24
        or len(
            {
                item.get("source_id")
                for item in value["sources"]
                if isinstance(item, Mapping)
            }
        )
        != 24
        or any(
            isinstance(item, Mapping)
            and item.get("source_id") == EXPECTED_SOURCE["source_id"]
            for item in value["sources"]
        )
    ):
        raise ValueError("MASSIVE predecessor registry v0.21 changed")
    return value


def project_successor_registry(
    predecessor: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != 24:
        raise ValueError("MASSIVE predecessor source inventory changed")
    if any(
        not isinstance(item, Mapping)
        or item.get("source_id") == EXPECTED_SOURCE["source_id"]
        or item.get("dataset_id") == EXPECTED_SOURCE["dataset_id"]
        for item in sources
    ):
        raise ValueError("MASSIVE source already exists in predecessor registry")
    predecessor_sources = copy.deepcopy(sources)
    policy = load_sft_policy(_input_path(root, "data_contract"))
    SFTSource.from_mapping(
        EXPECTED_SOURCE,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    successor = copy.deepcopy(dict(predecessor))
    successor["observed_at"] = EXPECTED_SUCCESSOR["observed_at"]
    successor["sources"].append(copy.deepcopy(EXPECTED_SOURCE))
    if (
        successor["sources"][:-1] != predecessor_sources
        or set(successor) != set(predecessor)
        or any(
            successor[key] != predecessor[key]
            for key in predecessor
            if key not in {"observed_at", "sources"}
        )
    ):
        raise ValueError("MASSIVE successor changed predecessor sources")
    payload = json_bytes(successor)
    if (
        len(payload) != EXPECTED_SUCCESSOR["bytes"]
        or sha256_bytes(payload) != EXPECTED_SUCCESSOR["file_sha256"]
        or sha256_value(successor) != EXPECTED_SUCCESSOR["canonical_sha256"]
        or len(successor["sources"]) != EXPECTED_SUCCESSOR["sources"]
    ):
        raise ValueError("MASSIVE exact successor registry v0.22 changed")
    return successor


def build_receipt(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del plan
    if _SHA256_RE.fullmatch(plan_file_sha256) is None:
        raise ValueError("MASSIVE source-promotion plan hash changed")
    root = Path(repository_root).resolve()
    _validate_preflight_evidence(root)
    predecessor = _predecessor_registry(root)
    successor = project_successor_registry(predecessor, repository_root=root)
    receipt: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
            "data_contract_file_sha256": EXPECTED_INPUTS["data_contract"]["sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_file_sha256": EXPECTED_INPUTS[
                "predecessor_registry"
            ]["sha256"],
            "preflight_plan_file_sha256": EXPECTED_INPUTS["preflight_plan"][
                "sha256"
            ],
            "preflight_report_file_sha256": EXPECTED_INPUTS["preflight_report"][
                "sha256"
            ],
            "preflight_report_internal_sha256": EXPECTED_INPUTS[
                "preflight_report"
            ]["internal_sha256"],
            "preflight_verification_file_sha256": EXPECTED_INPUTS[
                "preflight_verification"
            ]["sha256"],
            "preflight_verification_internal_sha256": EXPECTED_INPUTS[
                "preflight_verification"
            ]["internal_sha256"],
        },
        "candidate_projection": EXPECTED_CANDIDATE_PROJECTION,
        "checks": {
            "candidate_or_source_rows_read": False,
            "candidate_or_source_rows_mutated": False,
            "clean_two_pass_cross_source_preflight_bound": True,
            "dataset_sealed": False,
            "full_candidate_mix_reaudit_run": False,
            "network_gpu_inference_or_remote_used": False,
            "predecessor_24_sources_preserved_exactly": True,
            "predecessor_registry_mutated": False,
            "production_promoted": False,
            "registry_v0_22_append_only_source_additions": 1,
            "sft_training_started": False,
        },
        "next_gate": {
            "action": "full_registry_v0_22_candidate_mix_reaudit",
            "authorized": False,
            "separate_authorization_required": True,
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry": {
            "added_source_id": EXPECTED_SOURCE["source_id"],
            "added_sources": 1,
            "changed_predecessor_sources": 0,
            "predecessor_bytes": EXPECTED_INPUTS["predecessor_registry"]["bytes"],
            "predecessor_path": EXPECTED_INPUTS["predecessor_registry"]["path"],
            "predecessor_sha256": EXPECTED_INPUTS["predecessor_registry"]["sha256"],
            "predecessor_sources": 24,
            "removed_sources": 0,
            "successor_bytes": EXPECTED_SUCCESSOR["bytes"],
            "successor_canonical_sha256": EXPECTED_SUCCESSOR["canonical_sha256"],
            "successor_observed_at": EXPECTED_SUCCESSOR["observed_at"],
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": EXPECTED_SUCCESSOR["file_sha256"],
            "successor_sources": EXPECTED_SUCCESSOR["sources"],
        },
        "remaining_blockers": [
            "full_registry_v0_22_cross_source_hygiene_audit_not_complete",
            "candidate_mix_conversation_factual_banglish_and_natural_code_switching_"
            "floors_not_met",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "resource_accounting": {
            "candidate_row_reads": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "predecessor_registry_writes": 0,
            "successor_registry_writes": 1,
        },
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_after": EXPECTED_SOURCE,
        "status": RECEIPT_STATUS,
        "zero_use": {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        },
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    validate_receipt(receipt)
    return receipt, successor


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("receipt_sha256") != sha256_value(unsigned)
        or receipt.get("authorization") != EXPECTED_AUTHORIZATION
        or receipt.get("candidate_projection") != EXPECTED_CANDIDATE_PROJECTION
        or receipt.get("source_after") != EXPECTED_SOURCE
        or receipt.get("production_eligible") is not False
        or receipt.get("raw_training_text_retained") is not False
        or receipt.get("registry")
        != {
            "added_source_id": EXPECTED_SOURCE["source_id"],
            "added_sources": 1,
            "changed_predecessor_sources": 0,
            "predecessor_bytes": EXPECTED_INPUTS["predecessor_registry"]["bytes"],
            "predecessor_path": EXPECTED_INPUTS["predecessor_registry"]["path"],
            "predecessor_sha256": EXPECTED_INPUTS["predecessor_registry"]["sha256"],
            "predecessor_sources": 24,
            "removed_sources": 0,
            "successor_bytes": EXPECTED_SUCCESSOR["bytes"],
            "successor_canonical_sha256": EXPECTED_SUCCESSOR["canonical_sha256"],
            "successor_observed_at": EXPECTED_SUCCESSOR["observed_at"],
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": EXPECTED_SUCCESSOR["file_sha256"],
            "successor_sources": EXPECTED_SUCCESSOR["sources"],
        }
        or receipt.get("resource_accounting")
        != {
            "candidate_row_reads": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "predecessor_registry_writes": 0,
            "successor_registry_writes": 1,
        }
        or receipt.get("zero_use")
        != {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        }
        or receipt.get("next_gate")
        != {
            "action": "full_registry_v0_22_candidate_mix_reaudit",
            "authorized": False,
            "separate_authorization_required": True,
        }
    ):
        raise ValueError("MASSIVE source-promotion receipt changed")
    _assert_no_forbidden_keys(receipt)


def build_verification(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    receipt_file_bytes: int,
    plan_file_sha256: str,
) -> dict[str, Any]:
    validate_receipt(receipt)
    if (
        _SHA256_RE.fullmatch(receipt_file_sha256) is None
        or receipt_file_bytes <= 0
        or receipt.get("bindings", {}).get("plan_file_sha256")
        != plan_file_sha256
    ):
        raise ValueError("MASSIVE source-promotion receipt file binding changed")
    verification: dict[str, Any] = {
        "checks": {
            "candidate_source_test_objects_unread": True,
            "clean_v0_5_preflight_rebound": True,
            "predecessor_registry_hash_and_24_sources_reproduced": True,
            "production_dataset_seal_and_training_unchanged": True,
            "raw_training_text_absent": True,
            "receipt_internal_seal_verified": True,
            "successor_is_exact_append_only_25_source_registry": True,
            "successor_registry_schema_validated": True,
            "zero_network_gpu_inference_or_remote_use": True,
        },
        "combined_resource_accounting": {
            "candidate_row_reads": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "predecessor_registry_writes": 0,
            "successor_registry_writes": 1,
            "materialization_passes": 1,
            "verification_passes": 1,
        },
        "decision": {
            "next_gate": "full_registry_v0_22_candidate_mix_reaudit",
            "next_gate_authorized": False,
            "registry_v0_22_materialized": True,
            "separate_authorization_required": True,
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "receipt_file_bytes": receipt_file_bytes,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "successor_registry": {
            "bytes": EXPECTED_SUCCESSOR["bytes"],
            "canonical_sha256": EXPECTED_SUCCESSOR["canonical_sha256"],
            "file_sha256": EXPECTED_SUCCESSOR["file_sha256"],
            "source_count": EXPECTED_SUCCESSOR["sources"],
        },
        "verification_sha256": "",
        "zero_use": {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        },
    }
    verification["verification_sha256"] = sha256_value(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("successor_registry")
        != {
            "bytes": EXPECTED_SUCCESSOR["bytes"],
            "canonical_sha256": EXPECTED_SUCCESSOR["canonical_sha256"],
            "file_sha256": EXPECTED_SUCCESSOR["file_sha256"],
            "source_count": EXPECTED_SUCCESSOR["sources"],
        }
        or verification.get("combined_resource_accounting")
        != {
            "candidate_row_reads": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "predecessor_registry_writes": 0,
            "successor_registry_writes": 1,
            "materialization_passes": 1,
            "verification_passes": 1,
        }
        or verification.get("decision")
        != {
            "next_gate": "full_registry_v0_22_candidate_mix_reaudit",
            "next_gate_authorized": False,
            "registry_v0_22_materialized": True,
            "separate_authorization_required": True,
        }
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256_RE.fullmatch(verification[field]) is None
            for field in (
                "plan_file_sha256",
                "receipt_file_sha256",
                "receipt_internal_sha256",
            )
        )
        or not isinstance(verification.get("receipt_file_bytes"), int)
        or verification["receipt_file_bytes"] <= 0
    ):
        raise ValueError("MASSIVE source-promotion verification changed")
    _assert_no_forbidden_keys(verification)


def validate_materialized_successor(
    path: Path,
    *,
    repository_root: str | Path,
) -> None:
    root = Path(repository_root).resolve()
    payload = _read_bound_bytes(
        path,
        expected_bytes=EXPECTED_SUCCESSOR["bytes"],
        expected_sha256=EXPECTED_SUCCESSOR["file_sha256"],
    )
    value = strict_json_bytes(payload)
    predecessor = _predecessor_registry(root)
    expected = project_successor_registry(predecessor, repository_root=root)
    if value != expected or json_bytes(value) != payload:
        raise ValueError("MASSIVE materialized registry v0.22 is not reproducible")
    policy = load_sft_policy(_input_path(root, "data_contract"))
    registry = load_sft_source_registry(
        path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_SUCCESSOR["file_sha256"],
    )
    if (
        len(registry.sources) != EXPECTED_SUCCESSOR["sources"]
        or registry.sources[-1].source_id != EXPECTED_SOURCE["source_id"]
        or registry.sources[-1].production_eligible is not False
    ):
        raise ValueError("MASSIVE registry v0.22 parsed inventory changed")


def preflight_promoter(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path, Path]:
    successor = _resolve_repo_path(root, str(plan["output"]["successor_registry_path"]))
    receipt = _resolve_repo_path(root, str(plan["output"]["receipt_path"]))
    verification = _resolve_repo_path(root, str(plan["output"]["verification_path"]))
    if any(os.path.lexists(path) for path in (successor, receipt, verification)):
        raise FileExistsError("MASSIVE registry-v0.22 promotion already started")
    return successor, receipt, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path, Path]:
    successor = _resolve_repo_path(root, str(plan["output"]["successor_registry_path"]))
    receipt = _resolve_repo_path(root, str(plan["output"]["receipt_path"]))
    verification = _resolve_repo_path(root, str(plan["output"]["verification_path"]))
    if (
        not successor.is_file()
        or successor.stat().st_size != EXPECTED_SUCCESSOR["bytes"]
        or not receipt.is_file()
        or receipt.stat().st_size <= len(EXPECTED_RECEIPT_MARKER)
        or os.path.lexists(verification)
    ):
        raise FileExistsError("MASSIVE registry-v0.22 verification preflight changed")
    return successor, receipt, verification


def write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("MASSIVE registry-v0.22 write made no progress")
            written += count
        os.fsync(descriptor)
        if os.fstat(descriptor).st_size != len(payload):
            raise OSError("MASSIVE registry-v0.22 write size changed")
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    successor_bytes: bytes,
    receipt_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += _resolve_repo_path(root, str(plan[name]["path"])).stat().st_size
    return (
        static
        + len(EXPECTED_RECEIPT_MARKER)
        + len(successor_bytes)
        + len(receipt_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


def runtime_cap_reached(started: float) -> bool:
    return (
        time.monotonic() - started
        > EXPECTED_LIMITS["maximum_runtime_seconds_per_process"]
    )


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_SUCCESSOR",
    "EXPECTED_VERIFICATION_MARKER",
    "build_receipt",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_bytes",
    "load_plan",
    "preflight_promoter",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "project_successor_registry",
    "runtime_cap_reached",
    "sha256_bytes",
    "start_marker",
    "strict_json_bytes",
    "validate_materialized_successor",
    "validate_receipt",
    "validate_verification",
    "write_new_file",
]
