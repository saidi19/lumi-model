"""Fail-closed candidate-registry promotion for reviewed Aya Bengali data."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_sources import (
    SFTSource,
    SFTSourceRegistry,
    build_sft_source_readiness_report,
    load_sft_source_registry,
)

PLAN_SCHEMA_VERSION = "lumi.aya-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.aya-source-promotion-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_reviewed_aya_successor_registry_only",
}
EXPECTED_LIMITS = {
    "added_sources": 0,
    "changed_predecessor_sources": 1,
    "predecessor_sources": 10,
    "removed_sources": 0,
    "successor_sources": 10,
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "receipt_contains_raw_training_text": False,
    "receipt_path": "evidence/lumi-4b-sft-aya-source-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.4.json",
}
EXPECTED_CURATED_METRICS = {
    "accepted_records": 1478,
    "by_split": {"train": {"records": 1478, "supervised_tokens": 577453}},
    "context_window": 4096,
    "owner_sample_records": 32,
    "supervised_tokens": 577453,
}
EXPECTED_REQUIRED_TASKS = [
    "coding",
    "conversation",
    "factual_qa",
    "multilingual_bengali",
    "reasoning_math_science",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
]
EXPECTED_SOURCE_BEFORE = {
    "access": "public",
    "adapter": "aya_prompt_response_v0.1",
    "blockers": [],
    "dataset_id": "CohereLabs/aya_dataset",
    "evidence_urls": ["https://huggingface.co/datasets/CohereLabs/aya_dataset"],
    "languages": ["bn", "multilingual"],
    "license_id": "apache-2.0",
    "origin": "human",
    "production_blockers": [
        "bengali_subset_size_and_annotator_concentration_not_audited",
        "language_label_and_instruction_compliance_review_not_complete",
        "attribution_manifest_not_built",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/CohereLabs/aya_dataset",
    "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "source_id": "coherelabs_aya_dataset",
    "splits": ["train"],
    "status": "canary_eligible",
    "task_categories": [
        "factual_qa",
        "multilingual_bengali",
        "writing_summarization",
    ],
}
EXPECTED_SOURCE_AFTER = {
    "access": "public",
    "adapter": "aya_bengali_repaired_4k_annotator_cap_v0.2",
    "blockers": [],
    "dataset_id": "CohereLabs/aya_dataset",
    "evidence_urls": [
        (
            "https://huggingface.co/datasets/CohereLabs/aya_dataset/tree/"
            "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
        ),
        (
            "https://huggingface.co/datasets/CohereLabs/aya_dataset/blob/"
            "f9ea04583f02a8f86404ff6c58bf75fe637df8a2/README.md"
        ),
    ],
    "languages": ["bn"],
    "license_id": "apache-2.0",
    "origin": "human",
    "production_blockers": [
        "candidate_mix_is_only_1478_records_577453_supervised_tokens",
        "frozen_validation_split_not_materialized",
        "full_mix_cross_source_deduplication_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/CohereLabs/aya_dataset",
    "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "source_id": "coherelabs_aya_dataset",
    "splits": ["train"],
    "status": "canary_eligible",
    "task_categories": ["multilingual_bengali"],
}
EXPECTED_BINDINGS = frozenset(
    {
        "attribution_manifest",
        "data_contract",
        "materialization_report",
        "owner_review_receipt",
        "predecessor_registry",
        "promotion_implementation",
        "promotion_runner",
        "source_registry_implementation",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    if sha256_value({key: item for key, item in value.items() if key != field}) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_promotion_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Aya source-promotion binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("Aya source-promotion binding escapes workspace")
    return resolved


def load_source_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya source-promotion plan")
    if plan.get("status") != "authorized_candidate_registry_successor_promotion":
        raise ValueError("Aya source-promotion status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("Aya source-promotion date changed")
    for field, expected in (
        ("authorization", EXPECTED_AUTHORIZATION),
        ("limits", EXPECTED_LIMITS),
        ("output", EXPECTED_OUTPUT),
        ("curated_metrics", EXPECTED_CURATED_METRICS),
        ("required_tasks", EXPECTED_REQUIRED_TASKS),
        ("source_before", EXPECTED_SOURCE_BEFORE),
        ("source_after", EXPECTED_SOURCE_AFTER),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"Aya source-promotion {field} changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya source-promotion binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya source-promotion binding invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_promotion_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"Aya source-promotion binding changed: {name}")
    return plan


def verify_source_promotion_prerequisites(
    *,
    attribution_rows: Sequence[Mapping[str, Any]],
    materialization_report: Mapping[str, Any],
    owner_review_receipt: Mapping[str, Any],
) -> None:
    if materialization_report.get("schema_version") != (
        "lumi.sft-aya-bengali-materialization-report.v0.1"
    ):
        raise ValueError("unsupported Aya materialization report")
    if _verify_internal_seal(materialization_report, "report_sha256") != (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    ):
        raise ValueError("Aya materialization seal is not approved")
    if materialization_report.get("candidate") != {
        "materialized": True,
        "records": 1478,
        "supervised_tokens": 577453,
    }:
        raise ValueError("Aya materialized metrics changed")
    if (
        materialization_report.get("raw_training_text_in_report") is not False
        or materialization_report.get("production", {}).get("production_eligible") is not False
    ):
        raise ValueError("Aya materialization production boundary changed")

    if owner_review_receipt.get("schema_version") != (
        "lumi.sft-aya-bengali-owner-review-receipt.v0.1"
    ):
        raise ValueError("unsupported Aya owner-review receipt")
    if _verify_internal_seal(owner_review_receipt, "receipt_sha256") != (
        "e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f"
    ):
        raise ValueError("Aya owner-review seal is not approved")
    if (
        owner_review_receipt.get("status")
        != "owner_quality_sample_passed_source_promotion_still_separate"
        or owner_review_receipt.get("sample_records") != 32
        or owner_review_receipt.get("accepted_records") != 32
        or owner_review_receipt.get("rejected_records") != 0
        or owner_review_receipt.get("all_sampled_rows_accepted") is not True
        or owner_review_receipt.get("production_eligible") is not False
        or owner_review_receipt.get("production_promotions") != 0
        or owner_review_receipt.get("raw_training_text_retained") is not False
    ):
        raise ValueError("Aya owner-review gate did not pass")

    if len(attribution_rows) != 1478:
        raise ValueError("Aya attribution manifest count changed")
    record_ids: set[str] = set()
    for row in attribution_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("Aya attribution identity changed")
        if (
            row.get("schema_version") != "lumi.sft-aya-bengali-attribution.v0.1"
            or row.get("source_id") != "coherelabs_aya_dataset"
            or row.get("dataset_id") != "CohereLabs/aya_dataset"
            or row.get("revision") != EXPECTED_SOURCE_AFTER["revision"]
            or row.get("license_id") != "apache-2.0"
        ):
            raise ValueError("Aya attribution provenance changed")
        record_ids.add(record_id)


def load_policy_and_predecessor(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[SFTPolicy, SFTSourceRegistry, dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    policy = load_sft_policy(resolve_promotion_binding(root, bindings["data_contract"]))
    if (
        policy.source_registry_path != "configs/lumi-4b-sft-source-registry-v0.1.json"
        or policy.source_registry_sha256
        != "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9"
    ):
        raise ValueError("active production data-contract registry changed")
    registry_path = resolve_promotion_binding(root, bindings["predecessor_registry"])
    predecessor = load_sft_source_registry(
        registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=bindings["predecessor_registry"]["sha256"],
    )
    return policy, predecessor, _load_json(registry_path)


def build_successor_registry(
    predecessor_value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    raw_sources = predecessor_value.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 10:
        raise ValueError("Aya predecessor source count changed")
    indexes = [
        index
        for index, source in enumerate(raw_sources)
        if isinstance(source, Mapping) and source.get("source_id") == "coherelabs_aya_dataset"
    ]
    if indexes != [1] or raw_sources[1] != plan["source_before"]:
        raise ValueError("Aya predecessor source entry changed")
    successor = copy.deepcopy(dict(predecessor_value))
    successor["sources"][1] = copy.deepcopy(plan["source_after"])
    return successor


def validate_successor_registry(
    *,
    policy: SFTPolicy,
    predecessor_value: Mapping[str, Any],
    successor_value: Mapping[str, Any],
    successor_file_sha256: str,
) -> SFTSourceRegistry:
    _require_sha256(successor_file_sha256, "successor_file_sha256")
    before = predecessor_value.get("sources")
    after = successor_value.get("sources")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Aya registry sources must be arrays")
    if len(before) != 10 or len(after) != 10:
        raise ValueError("Aya successor source count changed")
    if {key: value for key, value in predecessor_value.items() if key != "sources"} != {
        key: value for key, value in successor_value.items() if key != "sources"
    }:
        raise ValueError("Aya successor changed registry metadata")
    changed = [
        index for index, (old, new) in enumerate(zip(before, after, strict=True)) if old != new
    ]
    if changed != [1] or before[1] != EXPECTED_SOURCE_BEFORE or after[1] != EXPECTED_SOURCE_AFTER:
        raise ValueError("Aya successor changed an unauthorized source")
    parsed = tuple(
        SFTSource.from_mapping(
            item,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for item in after
    )
    return SFTSourceRegistry(
        observed_at=str(successor_value["observed_at"]),
        registry_sha256=successor_file_sha256,
        sources=parsed,
    )


def build_source_promotion_receipt(
    *,
    owner_review_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    predecessor: SFTSourceRegistry,
    successor: SFTSourceRegistry,
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    required_tasks = frozenset(plan["required_tasks"])
    before = build_sft_source_readiness_report(predecessor, required_tasks=required_tasks)
    after = build_sft_source_readiness_report(successor, required_tasks=required_tasks)
    if before["missing_canary_tasks"] or after["missing_canary_tasks"]:
        raise ValueError("Aya registry lost candidate/canary task coverage")
    if before["missing_production_tasks"] != after["missing_production_tasks"]:
        raise ValueError("Aya promotion changed production coverage")
    if set(after["missing_production_tasks"]) != required_tasks:
        raise ValueError("Aya promotion unexpectedly enabled production")
    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "attribution_manifest_file_sha256": plan["bindings"]["attribution_manifest"]["sha256"],
            "materialization_report_file_sha256": plan["bindings"]["materialization_report"][
                "sha256"
            ],
            "materialization_report_internal_sha256": (
                "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
            ),
            "owner_review_file_sha256": plan["bindings"]["owner_review_receipt"]["sha256"],
            "owner_review_internal_sha256": owner_review_receipt["receipt_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": predecessor.registry_sha256,
            "successor_registry_sha256": successor.registry_sha256,
        },
        "candidate_source_registry_promotions": 1,
        "counts": EXPECTED_LIMITS,
        "curated_metrics": EXPECTED_CURATED_METRICS,
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry_delta": {
            "adapter_after": EXPECTED_SOURCE_AFTER["adapter"],
            "adapter_before": EXPECTED_SOURCE_BEFORE["adapter"],
            "changed_source_id": "coherelabs_aya_dataset",
            "languages_after": EXPECTED_SOURCE_AFTER["languages"],
            "languages_before": EXPECTED_SOURCE_BEFORE["languages"],
            "status_after": EXPECTED_SOURCE_AFTER["status"],
            "status_before": EXPECTED_SOURCE_BEFORE["status"],
            "task_categories_after": EXPECTED_SOURCE_AFTER["task_categories"],
            "task_categories_before": EXPECTED_SOURCE_BEFORE["task_categories"],
        },
        "remaining_blockers": [
            "frozen_validation_split_not_materialized",
            "complete_10m_to_30m_learning_mix_not_ready",
            "full_mix_cross_source_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "production_dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": copy.deepcopy(EXPECTED_SOURCE_AFTER),
        "status": (
            "reviewed_aya_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def verify_source_promotion_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya source-promotion receipt")
    _verify_internal_seal(value, "receipt_sha256")
    if (
        value.get("status")
        != (
            "reviewed_aya_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        )
        or value.get("candidate_source_registry_promotions") != 1
        or value.get("production_dataset_promotions") != 0
        or value.get("production_eligible") is not False
        or value.get("active_production_data_contract_updated") is not False
        or value.get("raw_training_text_retained") is not False
        or value.get("counts") != EXPECTED_LIMITS
        or value.get("curated_metrics") != EXPECTED_CURATED_METRICS
        or value.get("source") != EXPECTED_SOURCE_AFTER
    ):
        raise ValueError("Aya source-promotion receipt boundary changed")
