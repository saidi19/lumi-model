"""Fail-closed candidate-registry promotion for reviewed OASST1 data."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
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

PLAN_SCHEMA_VERSION = "lumi.oasst1-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.oasst1-source-promotion-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_reviewed_oasst1_successor_registry_only",
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
    "receipt_path": "evidence/lumi-4b-sft-oasst1-source-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.3.json",
}
EXPECTED_CURATED_METRICS = {
    "accepted_records": 1745,
    "by_split": {
        "train": {"records": 1652, "supervised_tokens": 453981},
        "validation": {"records": 93, "supervised_tokens": 23475},
    },
    "context_window": 4096,
    "owner_sample_records": 32,
    "supervised_tokens": 477456,
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
    "adapter": "oasst1_tree_v0.1",
    "blockers": [],
    "dataset_id": "OpenAssistant/oasst1",
    "evidence_urls": [
        "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE",
    ],
    "languages": ["multilingual"],
    "license_id": "apache-2.0",
    "origin": "human",
    "production_blockers": [
        "record_level_language_and_quality_audit_not_complete",
        "thread_selection_and_rank_policy_not_sealed",
        "attribution_manifest_not_built",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
    "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
    "source_id": "openassistant_oasst1",
    "splits": ["train", "validation"],
    "status": "canary_eligible",
    "task_categories": [
        "coding",
        "conversation",
        "factual_qa",
        "writing_summarization",
    ],
}
EXPECTED_SOURCE_AFTER = {
    "access": "public",
    "adapter": "oasst1_root_direct_rank0_quality_v0.3",
    "blockers": [],
    "dataset_id": "OpenAssistant/oasst1",
    "evidence_urls": [
        (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/tree/"
            "fdf72ae0827c1cda404aff25b6603abec9e3399b"
        ),
        (
            "https://huggingface.co/datasets/OpenAssistant/oasst1/blob/"
            "fdf72ae0827c1cda404aff25b6603abec9e3399b/LICENSE"
        ),
    ],
    "languages": ["en"],
    "license_id": "apache-2.0",
    "origin": "human",
    "production_blockers": [
        "candidate_mix_is_only_1745_records_477456_supervised_tokens",
        "full_mix_cross_source_deduplication_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
    "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
    "source_id": "openassistant_oasst1",
    "splits": ["train", "validation"],
    "status": "canary_eligible",
    "task_categories": [
        "coding",
        "conversation",
        "factual_qa",
        "writing_summarization",
    ],
}
EXPECTED_BINDINGS = frozenset(
    {
        "curation_report",
        "curation_run_plan",
        "curation_verification",
        "data_contract",
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
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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
    unsealed = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsealed) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_promotion_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("OASST1 source-promotion binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("OASST1 source-promotion binding escapes repository")
    return resolved


def load_source_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 source-promotion plan")
    if plan.get("status") != "authorized_candidate_registry_successor_promotion":
        raise ValueError("OASST1 source-promotion status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OASST1 source-promotion date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST1 source-promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OASST1 source-promotion limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("OASST1 source-promotion output boundary changed")
    if plan.get("curated_metrics") != EXPECTED_CURATED_METRICS:
        raise ValueError("OASST1 source-promotion metrics changed")
    if plan.get("required_tasks") != EXPECTED_REQUIRED_TASKS:
        raise ValueError("OASST1 source-promotion task boundary changed")
    if plan.get("source_before") != EXPECTED_SOURCE_BEFORE:
        raise ValueError("OASST1 predecessor entry changed")
    if plan.get("source_after") != EXPECTED_SOURCE_AFTER:
        raise ValueError("OASST1 successor entry changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OASST1 source-promotion binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OASST1 source-promotion binding invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_promotion_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"OASST1 source-promotion binding changed: {name}")
    return plan


def verify_source_promotion_prerequisites(
    *,
    curation_report: Mapping[str, Any],
    curation_verification: Mapping[str, Any],
    owner_review_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    if curation_report.get("schema_version") != (
        "lumi.sft-oasst1-full-curation-report.v0.1"
    ):
        raise ValueError("unsupported OASST1 curation report")
    report_seal = _verify_internal_seal(curation_report, "report_sha256")
    if report_seal != "0df672f2b34a6e580fc9a577749fb03359c9628ca92a4cfe0a6ba801e25ff580":
        raise ValueError("OASST1 curation report seal is not approved")
    if curation_report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("OASST1 curation status changed")
    if curation_report.get("source") != {
        "dataset_id": "OpenAssistant/oasst1",
        "license_id": "apache-2.0",
        "revision": EXPECTED_SOURCE_AFTER["revision"],
        "source_id": "openassistant_oasst1",
    }:
        raise ValueError("OASST1 curation source binding changed")
    token_measurement = curation_report.get("curation", {}).get("token_measurement")
    if not isinstance(token_measurement, Mapping):
        raise ValueError("OASST1 token measurement is missing")
    if (
        token_measurement.get("accepted_records") != 1745
        or token_measurement.get("supervised_tokens") != 477456
        or token_measurement.get("context_window") != 4096
        or token_measurement.get("assistant_only_loss_verified") is not True
        or token_measurement.get("overlength_rejected") != 1
    ):
        raise ValueError("OASST1 token measurement changed")
    by_split = token_measurement.get("by_split")
    for split, expected in EXPECTED_CURATED_METRICS["by_split"].items():
        observed = by_split.get(split) if isinstance(by_split, Mapping) else None
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"OASST1 {split} metrics changed")
    if curation_report.get("production") != {
        "dataset_promotion_authorized": False,
        "dataset_seal_authorized": False,
        "production_eligible": False,
        "sft_training_authorized": False,
    } or curation_report.get("raw_training_text_present") is not False:
        raise ValueError("OASST1 curation production boundary changed")

    if curation_verification.get("schema_version") != (
        "lumi.sft-oasst1-full-curation-verification.v0.1"
    ):
        raise ValueError("unsupported OASST1 curation verification")
    verification_seal = _verify_internal_seal(
        curation_verification, "verification_sha256"
    )
    if verification_seal != (
        "bdd099dae98812117db028f98a4d4aa11b23ab1e08b59ae67667b1a0484b2cc3"
    ):
        raise ValueError("OASST1 curation verification seal is not approved")
    if (
        curation_verification.get("status")
        != "verified_nonproduction_owner_review_pending"
        or curation_verification.get("production_eligible") is not False
        or curation_verification.get("curated")
        != {
            "accepted_records": 1745,
            "owner_sample_records": 32,
            "supervised_tokens": 477456,
        }
        or set(curation_verification.get("checks", {}).values()) != {True}
    ):
        raise ValueError("OASST1 curation verification boundary changed")

    if owner_review_receipt.get("schema_version") != (
        "lumi.sft-oasst1-owner-review-receipt.v0.1"
    ):
        raise ValueError("unsupported OASST1 owner-review receipt")
    owner_seal = _verify_internal_seal(owner_review_receipt, "receipt_sha256")
    if owner_seal != "6f3f0232983695403f110d898ed59ab5d1bff503f587aa5360db6ec9e46439a4":
        raise ValueError("OASST1 owner-review seal is not approved")
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
        raise ValueError("OASST1 owner-review gate did not pass")


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
        raise ValueError("OASST1 predecessor source count changed")
    indexes = [
        index
        for index, source in enumerate(raw_sources)
        if isinstance(source, Mapping) and source.get("source_id") == "openassistant_oasst1"
    ]
    if indexes != [0] or raw_sources[0] != plan["source_before"]:
        raise ValueError("OASST1 predecessor source entry changed")
    successor = copy.deepcopy(dict(predecessor_value))
    successor["sources"][0] = copy.deepcopy(plan["source_after"])
    return successor


def validate_successor_registry(
    *,
    policy: SFTPolicy,
    predecessor_value: Mapping[str, Any],
    successor_value: Mapping[str, Any],
    successor_file_sha256: str,
) -> SFTSourceRegistry:
    _require_sha256(successor_file_sha256, "successor_file_sha256")
    predecessor_sources = predecessor_value.get("sources")
    successor_sources = successor_value.get("sources")
    if not isinstance(predecessor_sources, list) or not isinstance(
        successor_sources, list
    ):
        raise ValueError("OASST1 registry sources must be arrays")
    if len(predecessor_sources) != 10 or len(successor_sources) != 10:
        raise ValueError("OASST1 successor source count changed")
    if {
        key: value for key, value in predecessor_value.items() if key != "sources"
    } != {key: value for key, value in successor_value.items() if key != "sources"}:
        raise ValueError("OASST1 successor changed registry metadata")
    changed = [
        index
        for index, (before, after) in enumerate(
            zip(predecessor_sources, successor_sources, strict=True)
        )
        if before != after
    ]
    if changed != [0]:
        raise ValueError("OASST1 successor changed an unauthorized source")
    if (
        predecessor_sources[0] != EXPECTED_SOURCE_BEFORE
        or successor_sources[0] != EXPECTED_SOURCE_AFTER
    ):
        raise ValueError("OASST1 registry transition changed")
    parsed = tuple(
        SFTSource.from_mapping(
            item,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for item in successor_sources
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
    before = build_sft_source_readiness_report(
        predecessor, required_tasks=required_tasks
    )
    after = build_sft_source_readiness_report(successor, required_tasks=required_tasks)
    if before["missing_canary_tasks"] or after["missing_canary_tasks"]:
        raise ValueError("OASST1 registry lost candidate/canary task coverage")
    if before["missing_production_tasks"] != after["missing_production_tasks"]:
        raise ValueError("OASST1 promotion changed production coverage")
    if set(after["missing_production_tasks"]) != required_tasks:
        raise ValueError("OASST1 promotion unexpectedly enabled production")
    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "curation_report_file_sha256": plan["bindings"]["curation_report"][
                "sha256"
            ],
            "curation_report_internal_sha256": (
                "0df672f2b34a6e580fc9a577749fb03359c9628ca92a4cfe0a6ba801e25ff580"
            ),
            "curation_verification_file_sha256": plan["bindings"][
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": (
                "bdd099dae98812117db028f98a4d4aa11b23ab1e08b59ae67667b1a0484b2cc3"
            ),
            "owner_review_file_sha256": plan["bindings"]["owner_review_receipt"][
                "sha256"
            ],
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
            "changed_source_id": "openassistant_oasst1",
            "languages_after": EXPECTED_SOURCE_AFTER["languages"],
            "languages_before": EXPECTED_SOURCE_BEFORE["languages"],
            "status_after": EXPECTED_SOURCE_AFTER["status"],
            "status_before": EXPECTED_SOURCE_BEFORE["status"],
        },
        "remaining_blockers": [
            "complete_10m_to_30m_learning_mix_not_ready",
            "full_mix_cross_source_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "production_dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": copy.deepcopy(EXPECTED_SOURCE_AFTER),
        "status": (
            "reviewed_oasst1_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def verify_source_promotion_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 source-promotion receipt")
    _verify_internal_seal(value, "receipt_sha256")
    if (
        value.get("status")
        != (
            "reviewed_oasst1_registered_in_immutable_candidate_successor_"
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
        raise ValueError("OASST1 source-promotion receipt boundary changed")
