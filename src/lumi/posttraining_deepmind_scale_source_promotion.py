"""Fail-closed registry promotion for the reviewed DeepMind scale candidate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_deepmind_scale_generation import file_sha256, sha256_value
from .posttraining_deepmind_scale_owner_review import (
    validate_owner_review_receipt,
)
from .posttraining_deepmind_scale_promotion_preflight import (
    EXPECTED_CANDIDATE,
    EXPECTED_SOURCE_AFTER,
    build_proposed_successor,
    file_sha256_from_bytes,
    json_file_bytes,
    validate_preflight_report,
)
from .posttraining_sources import SFTSource

PLAN_SCHEMA_VERSION = "lumi.sft-deepmind-scale-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-deepmind-scale-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-deepmind-scale-source-promotion-verification.v0.1"
)

PREDECESSOR_REGISTRY_SHA256 = (
    "97b7a4399810f95d9c2e39dfe236a0129bd005abac81e067b67b8d0c0fb0baac"
)
ACCEPTED_RECORDS_SHA256 = (
    "5601aab088a11c2664e0e8067b509495bf3788e07b4e8e3fa548afef7ad1c7a9"
)
OWNER_REVIEW_INTERNAL_SHA256 = (
    "97c898d6c16f55f5620ecd1ab22f8836f445c5d1ed73eb1303d073887df1d44f"
)
PREFLIGHT_REPORT_INTERNAL_SHA256 = (
    "491036a98e6e27208f22492aff66ac84e75823766ab69ca8ae3b63b866821f5d"
)
PREFLIGHT_VERIFICATION_INTERNAL_SHA256 = (
    "4047397833112eb1ae93e284f85594b15b913010c0bc553ca7ba9b0b424b4502"
)
SUCCESSOR_REGISTRY_SHA256 = (
    "272b88f69ae19da25d3e979cf9d74ab0d4c8a1e6d276b9adcc148fd10738490f"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_reviewed_deepmind_scale_registry_v0_10_successor_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "changed_predecessor_sources": 1,
    "candidate_records": 236_698,
    "maximum_runtime_minutes": 60,
    "predecessor_sources": 13,
    "successor_sources": 13,
}
EXPECTED_OUTPUT = {
    "active_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": (
        "evidence/lumi-4b-sft-deepmind-scale-source-promotion-v0.1.json"
    ),
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.10.json",
    "verification_path": (
        "evidence/"
        "lumi-4b-sft-deepmind-scale-source-promotion-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "data_contract",
        "implementation",
        "owner_review_receipt",
        "predecessor_registry",
        "preflight_implementation",
        "preflight_report",
        "preflight_verification",
        "runner",
        "source_registry_implementation",
        "verifier",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_binding(
    repository_root: Path, specification: Mapping[str, Any], *, name: str
) -> Path:
    workspace = repository_root.parent.resolve()
    path = (repository_root / str(specification.get("path", ""))).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise FileNotFoundError(f"DeepMind scale promotion binding missing: {name}")
    expected = _require_sha256(specification.get("sha256"), f"{name}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"DeepMind scale promotion binding changed: {name}")
    return path


def load_promotion_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "candidate": EXPECTED_CANDIDATE,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
        "proposed_successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_after": EXPECTED_SOURCE_AFTER,
        "status": "authorized_exact_candidate_registry_successor_promotion",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"DeepMind scale promotion plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("DeepMind scale promotion binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"DeepMind scale promotion binding invalid: {name}")
        resolve_binding(repository_root, specification, name=name)
    if bindings["accepted_records"]["sha256"] != ACCEPTED_RECORDS_SHA256:
        raise ValueError("DeepMind scale accepted-records authorization changed")
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("DeepMind scale predecessor authorization changed")
    return plan


def verify_promotion_prerequisites(
    *, plan: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bindings = plan["bindings"]
    preflight = _load_json(
        resolve_binding(
            repository_root, bindings["preflight_report"], name="preflight_report"
        )
    )
    validate_preflight_report(preflight)
    if preflight.get("report_sha256") != PREFLIGHT_REPORT_INTERNAL_SHA256:
        raise ValueError("DeepMind scale preflight report is not authorized")
    if preflight.get("proposed_successor", {}).get("file_sha256") != (
        SUCCESSOR_REGISTRY_SHA256
    ):
        raise ValueError("DeepMind scale proposed successor changed")
    preflight_verification = _load_json(
        resolve_binding(
            repository_root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    if _verify_internal_seal(
        preflight_verification, "verification_sha256"
    ) != PREFLIGHT_VERIFICATION_INTERNAL_SHA256:
        raise ValueError("DeepMind scale preflight verification is not authorized")
    if (
        preflight_verification.get("status")
        != "verified_deepmind_scale_promotion_preflight_nonproduction"
        or preflight_verification.get("report_internal_sha256")
        != PREFLIGHT_REPORT_INTERNAL_SHA256
        or preflight_verification.get("proposed_successor_file_sha256")
        != SUCCESSOR_REGISTRY_SHA256
    ):
        raise ValueError("DeepMind scale preflight verification changed")
    owner_receipt = _load_json(
        resolve_binding(
            repository_root,
            bindings["owner_review_receipt"],
            name="owner_review_receipt",
        )
    )
    validate_owner_review_receipt(owner_receipt)
    if (
        owner_receipt.get("receipt_sha256") != OWNER_REVIEW_INTERNAL_SHA256
        or owner_receipt.get("status")
        != "owner_quality_sample_passed_source_promotion_still_separate"
        or owner_receipt.get("accepted_records") != 32
        or owner_receipt.get("rejected_records") != 0
        or owner_receipt.get("candidate") != EXPECTED_CANDIDATE
    ):
        raise ValueError("DeepMind scale owner-review gate changed")
    predecessor = _load_json(
        resolve_binding(
            repository_root,
            bindings["predecessor_registry"],
            name="predecessor_registry",
        )
    )
    return preflight, owner_receipt, predecessor


def build_and_validate_successor(
    *,
    plan: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    successor, source_before = build_proposed_successor(
        predecessor, source_after=plan["source_after"]
    )
    if file_sha256_from_bytes(json_file_bytes(successor)) != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("DeepMind scale successor SHA-256 changed")
    before_sources = predecessor.get("sources")
    after_sources = successor.get("sources")
    if not isinstance(before_sources, list) or not isinstance(after_sources, list):
        raise ValueError("DeepMind scale registry source arrays changed")
    if len(before_sources) != len(after_sources) or len(after_sources) != 13:
        raise ValueError("DeepMind scale registry source count changed")
    changed = [
        index
        for index, (before, after) in enumerate(
            zip(before_sources, after_sources, strict=True)
        )
        if before != after
    ]
    if changed != [11]:
        raise ValueError("DeepMind scale promotion would change unexpected sources")
    parsed = [
        SFTSource.from_mapping(
            source,
            allowed_licenses=frozenset(policy["allowed_licenses"]),
            allowed_languages=frozenset(policy["allowed_languages"]),
            allowed_splits=frozenset(policy["allowed_splits"]),
            allowed_tasks=frozenset(policy["task_mix"]),
        )
        for source in after_sources
    ]
    source_ids = [source.source_id for source in parsed]
    dataset_ids = [source.dataset_id for source in parsed]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("DeepMind scale successor source IDs are not unique")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("DeepMind scale successor dataset IDs are not unique")
    return successor, source_before


def build_promotion_receipt(
    *,
    owner_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    preflight: Mapping[str, Any],
    source_before: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    receipt: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_records_file_sha256": ACCEPTED_RECORDS_SHA256,
            "owner_review_internal_sha256": owner_receipt["receipt_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
            "preflight_report_internal_sha256": preflight["report_sha256"],
            "preflight_verification_internal_sha256": (
                PREFLIGHT_VERIFICATION_INTERNAL_SHA256
            ),
        },
        "candidate": dict(EXPECTED_CANDIDATE)
        | {"total_supervised_tokens": 2_339_166},
        "checks": {
            "active_data_contract_updated": False,
            "candidate_registry_promotions": 1,
            "changed_source_ids": ["deepmind_mathematics_dataset"],
            "owner_quality_sample_accepted_32_of_32": True,
            "predecessor_registry_mutated": False,
            "production_dataset_promotions": 0,
            "raw_training_text_retained": False,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "registry": {
            "predecessor_path": "configs/lumi-4b-sft-source-registry-v0.9.json",
            "predecessor_sha256": PREDECESSOR_REGISTRY_SHA256,
            "successor_path": EXPECTED_OUTPUT["successor_registry_path"],
            "successor_sha256": SUCCESSOR_REGISTRY_SHA256,
        },
        "remaining_blockers": [
            "complete_sft_mix_still_below_10m_train_token_floor",
            "deepmind_source_share_exceeds_0_35_cap",
            "cross_source_deduplication_against_complete_mix_not_complete",
            "production_dataset_seal_and_gpu_sft_not_authorized",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_after": dict(plan["source_after"]),
        "source_before": dict(source_before),
        "status": "verified_candidate_registry_v0_10_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DeepMind scale promotion receipt")
    _verify_internal_seal(receipt, "receipt_sha256")
    if receipt.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DeepMind scale promotion receipt authorization changed")
    if receipt.get("status") != "verified_candidate_registry_v0_10_nonproduction":
        raise ValueError("DeepMind scale promotion status changed")
    if receipt.get("production_eligible") is not False:
        raise ValueError("DeepMind scale promotion crossed production boundary")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("DeepMind scale promotion checks are missing")
    if (
        checks.get("candidate_registry_promotions") != 1
        or checks.get("production_dataset_promotions") != 0
        or checks.get("active_data_contract_updated") is not False
        or checks.get("predecessor_registry_mutated") is not False
        or checks.get("raw_training_text_retained") is not False
        or checks.get("changed_source_ids") != ["deepmind_mathematics_dataset"]
    ):
        raise ValueError("DeepMind scale promotion boundary changed")


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "PLAN_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "SUCCESSOR_REGISTRY_SHA256",
    "VERIFICATION_SCHEMA_VERSION",
    "build_and_validate_successor",
    "build_promotion_receipt",
    "json_file_bytes",
    "load_promotion_plan",
    "resolve_binding",
    "validate_promotion_receipt",
    "verify_promotion_prerequisites",
]
