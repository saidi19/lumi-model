"""Authorized immutable registry successor materialization for AI-assisted v0.3."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_source_promotion_preflight import (
    SOURCE_ID,
    SOURCE_REVISION,
    build_ai_v3_promotion_preflight_report,
    build_projected_successor,
    load_ai_v3_promotion_preflight_plan,
    verify_ai_v3_promotion_preflight_report,
)
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource, SFTSourceRegistry

PLAN_SCHEMA_VERSION = "lumi.sft-ai-v3-source-promotion-plan.v0.1"
PLAN_STATUS = "authorized_candidate_registry_successor_materialization"
RECEIPT_SCHEMA_VERSION = "lumi.sft-ai-v3-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-ai-v3-source-promotion-verification.v0.1"

PREFLIGHT_INTERNAL_SHA256 = (
    "1e553054c576a7c167a8c3aef271b5cca62664c3106587321e5f2c06ef768795"
)
SUCCESSOR_SHA256 = "d20c89ca34a284fc120a115d0ac9578c7e03ae3c915452d3f0ea144928e9a519"

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "candidate_row_modification_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "materialize_exact_preflighted_ai_v0_3_candidate_registry_v0_8_only",
    "unrelated_access_authorized": False,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def resolve_binding(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("AI v0.3 promotion binding path is missing")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"AI v0.3 promotion binding SHA-256 invalid: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"AI v0.3 promotion binding escapes repository: {relative}")
    if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
        raise ValueError(f"AI v0.3 promotion binding changed: {relative}")
    return resolved


def load_ai_v3_source_promotion_plan(
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = read_object(path.resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 source-promotion plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("AI v0.3 source-promotion status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("AI v0.3 promotion authorization boundary changed")
    approval = plan.get("owner_approval")
    if approval != {
        "maximum_minutes": 30,
        "preflight_report_sha256": PREFLIGHT_INTERNAL_SHA256,
        "projected_successor_sha256": SUCCESSOR_SHA256,
        "records": 320,
        "source_id": SOURCE_ID,
        "train_supervised_tokens": 9314,
        "validation_supervised_tokens": 2138,
    }:
        raise ValueError("AI v0.3 owner approval boundary changed")
    limits = plan.get("limits")
    if limits != {
        "added_sources": 1,
        "changed_predecessor_sources": 0,
        "predecessor_sources": 12,
        "removed_sources": 0,
        "successor_sources": 13,
    }:
        raise ValueError("AI v0.3 registry delta limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("AI v0.3 promotion bindings are missing")
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise ValueError("AI v0.3 promotion binding is invalid")
        resolve_binding(root, binding)
    source = plan.get("source_after")
    if not isinstance(source, dict):
        raise ValueError("AI v0.3 authorized source entry is missing")
    if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("AI v0.3 authorized source identity changed")
    output = plan.get("output")
    if output != {
        "active_production_data_contract_update_permitted": False,
        "predecessor_registry_mutation_permitted": False,
        "production_dataset_promotions": 0,
        "receipt_contains_raw_training_text": False,
        "receipt_path": "evidence/lumi-4b-sft-ai-v3-source-promotion-v0.1.json",
        "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.8.json",
        "verification_path": (
            "evidence/lumi-4b-sft-ai-v3-source-promotion-verification-v0.1.json"
        ),
    }:
        raise ValueError("AI v0.3 promotion outputs changed")
    return plan


def reconstruct_authorized_successor(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the preflight and return predecessor, successor, and preflight report."""

    root = repository_root.resolve()
    bindings = plan["bindings"]
    preflight_plan_path = resolve_binding(root, bindings["preflight_plan"])
    preflight_plan = load_ai_v3_promotion_preflight_plan(
        preflight_plan_path,
        repository_root=root,
    )
    expected_preflight = build_ai_v3_promotion_preflight_report(
        preflight_plan,
        plan_file_sha256=file_sha256(preflight_plan_path),
        repository_root=root,
    )
    verify_ai_v3_promotion_preflight_report(expected_preflight)
    preflight_report_path = resolve_binding(root, bindings["preflight_report"])
    preflight_report = read_object(preflight_report_path)
    if preflight_report != expected_preflight:
        raise ValueError("AI v0.3 promotion preflight reconstruction changed")
    if preflight_report.get("report_sha256") != PREFLIGHT_INTERNAL_SHA256:
        raise ValueError("AI v0.3 authorized preflight seal changed")

    predecessor_path = resolve_binding(root, bindings["predecessor_registry"])
    predecessor = read_object(predecessor_path)
    successor = build_projected_successor(
        predecessor,
        source_after=plan["source_after"],
    )
    successor_sha256 = hashlib.sha256(json_file_bytes(successor)).hexdigest()
    if successor_sha256 != SUCCESSOR_SHA256:
        raise ValueError("AI v0.3 projected successor is not the authorized artifact")
    if preflight_report.get("projected_successor", {}).get("sha256") != successor_sha256:
        raise ValueError("AI v0.3 preflight successor binding changed")

    policy = load_sft_policy(resolve_binding(root, bindings["data_contract"]))
    parsed = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in successor["sources"]
    )
    registry = SFTSourceRegistry(
        observed_at=str(successor["observed_at"]),
        registry_sha256=successor_sha256,
        sources=parsed,
    )
    if len(registry.sources) != 13 or registry.sources[-1].source_id != SOURCE_ID:
        raise ValueError("AI v0.3 authorized successor parsing changed")
    return predecessor, successor, preflight_report


def build_source_promotion_receipt(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    predecessor: Mapping[str, Any],
    successor: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    before = predecessor["sources"]
    after = successor["sources"]
    if after[:12] != before or after[12] != plan["source_after"]:
        raise ValueError("AI v0.3 receipt registry delta changed")
    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": copy.deepcopy(plan["authorization"]),
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": plan["bindings"]["predecessor_registry"][
                "sha256"
            ],
            "preflight_report_file_sha256": plan["bindings"]["preflight_report"][
                "sha256"
            ],
            "preflight_report_internal_sha256": preflight_report["report_sha256"],
            "successor_registry_sha256": SUCCESSOR_SHA256,
        },
        "candidate_source_registry_promotions": 1,
        "counts": {
            "added_sources": 1,
            "changed_predecessor_sources": 0,
            "predecessor_sources": 12,
            "removed_sources": 0,
            "successor_sources": 13,
        },
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "record_provenance": {
            "owner_sample_accepted": 32,
            "owner_sample_rejected": 0,
            "per_record_human_review_status_counts": {
                "owner_spot_check_pending": 320
            },
        },
        "registry_delta": {
            "added_source_id": SOURCE_ID,
            "changed_predecessor_sources": 0,
            "removed_sources": 0,
            "status_after": "candidate_eligible",
        },
        "remaining_blockers": [
            "per_record_owner_review_provenance_not_materialized",
            "complete_learning_and_quality_expansion_mix_not_ready",
            "full_mix_cross_source_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "production_dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": copy.deepcopy(plan["source_after"]),
        "status": (
            "reviewed_ai_v0_3_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        ),
        "token_metrics": {
            "records": 320,
            "train": {"records": 240, "supervised_tokens": 9314},
            "validation": {"records": 80, "supervised_tokens": 2138},
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_source_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 source-promotion receipt")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("AI v0.3 source-promotion receipt seal mismatch")
    if receipt.get("candidate_source_registry_promotions") != 1:
        raise ValueError("AI v0.3 candidate source promotion count changed")
    if receipt.get("production_dataset_promotions") != 0:
        raise ValueError("AI v0.3 receipt promoted production data")
    if receipt.get("active_production_data_contract_updated") is not False:
        raise ValueError("AI v0.3 receipt changed the production data contract")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("AI v0.3 receipt retained raw training text")
    if receipt.get("production_eligible") is not False:
        raise ValueError("AI v0.3 receipt became production eligible")


def build_source_promotion_verification(
    *,
    plan: Mapping[str, Any],
    receipt_file_sha256: str,
    receipt_internal_sha256: str,
) -> dict[str, Any]:
    verification: dict[str, Any] = {
        "checks": {
            "active_production_data_contract_updated": False,
            "candidate_rows_modified": False,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "receipt_reconstructed": True,
            "successor_adds_exactly_one_source": True,
            "successor_registry_reconstructed": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt_internal_sha256,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_ai_v0_3_candidate_registry_promotion_nonproduction",
        "successor_registry_sha256": SUCCESSOR_SHA256,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


def verify_source_promotion_verification(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 source-promotion verification")
    unsigned = dict(value)
    claimed = unsigned.pop("verification_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("AI v0.3 source-promotion verification seal mismatch")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or not all(
        check is True or check is False for check in checks.values()
    ):
        raise ValueError("AI v0.3 verification checks are invalid")
    if checks.get("active_production_data_contract_updated") is not False:
        raise ValueError("AI v0.3 verification changed the production contract")
    if checks.get("candidate_rows_modified") is not False:
        raise ValueError("AI v0.3 verification modified candidate rows")
    if checks.get("production_dataset_promotions") != 0:
        raise ValueError("AI v0.3 verification promoted production data")
