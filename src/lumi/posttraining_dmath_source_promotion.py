"""Fail-closed candidate-registry promotion for reviewed DMath data."""

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

PLAN_SCHEMA_VERSION = "lumi.dmath-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.dmath-source-promotion-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.dmath-source-promotion-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "dataset_seal_authorized": False,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_reviewed_dmath_successor_registry_only",
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "changed_predecessor_sources": 0,
    "predecessor_sources": 10,
    "removed_sources": 0,
    "successor_sources": 11,
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "receipt_contains_raw_training_text": False,
    "receipt_path": "evidence/lumi-4b-sft-dmath-source-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.5.json",
    "verification_path": (
        "evidence/lumi-4b-sft-dmath-source-promotion-verification-v0.1.json"
    ),
}
EXPECTED_CURATED_METRICS = {
    "accepted_records": 6943,
    "assistant_only_loss_verified": True,
    "by_split": {
        "train": {"records": 6596, "supervised_tokens": 48480},
        "validation": {"records": 347, "supervised_tokens": 2546},
    },
    "context_window": 4096,
    "owner_sample_records": 32,
    "supervised_tokens": 51026,
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
EXPECTED_SOURCE_AFTER = {
    "access": "public",
    "adapter": "dmath_question_answer_en_verified_v0.1",
    "blockers": [],
    "dataset_id": "JiwooKimAR/dmath",
    "evidence_urls": [
        (
            "https://github.com/JiwooKimAR/dmath/tree/"
            "149a7403619ac1bd6ed10f50694ad82699d5cc03"
        ),
        (
            "https://github.com/JiwooKimAR/dmath/blob/"
            "149a7403619ac1bd6ed10f50694ad82699d5cc03/LICENSE"
        ),
        (
            "https://github.com/JiwooKimAR/dmath/blob/"
            "149a7403619ac1bd6ed10f50694ad82699d5cc03/dataset/train.json"
        ),
    ],
    "languages": ["en"],
    "license_id": "mit",
    "origin": "human",
    "production_blockers": [
        "candidate_mix_is_only_6943_records_51026_supervised_tokens",
        "official_test_split_reserved_for_evaluation",
        "full_mix_cross_source_deduplication_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": "https://huggingface.co/datasets/JiwooKimAR/dmath",
    "revision": "149a7403619ac1bd6ed10f50694ad82699d5cc03",
    "source_id": "dmath",
    "splits": ["train", "validation"],
    "status": "canary_eligible",
    "task_categories": ["reasoning_math_science"],
}
EXPECTED_BINDINGS = frozenset(
    {
        "accepted_records",
        "curation_report",
        "curation_run_plan",
        "curation_verification",
        "data_contract",
        "owner_review_receipt",
        "owner_review_verification",
        "predecessor_registry",
        "promotion_implementation",
        "promotion_runner",
        "promotion_verifier",
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
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
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
        raise ValueError("DMath source-promotion binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("DMath source-promotion binding escapes workspace")
    return resolved


def load_source_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "curated_metrics": EXPECTED_CURATED_METRICS,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "required_tasks": EXPECTED_REQUIRED_TASKS,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_after": EXPECTED_SOURCE_AFTER,
        "status": "authorized_candidate_registry_successor_promotion",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"DMath source-promotion {field} changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("DMath source-promotion binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"DMath source-promotion binding invalid: {name}")
        expected_sha = _require_sha256(
            binding.get("sha256"), f"bindings.{name}.sha256"
        )
        resolved = resolve_promotion_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected_sha:
            raise ValueError(f"DMath source-promotion binding changed: {name}")
    return plan


def verify_source_promotion_prerequisites(
    *,
    accepted_records: Sequence[Mapping[str, Any]],
    curation_report: Mapping[str, Any],
    curation_verification: Mapping[str, Any],
    owner_review_receipt: Mapping[str, Any],
    owner_review_verification: Mapping[str, Any],
) -> None:
    if curation_report.get("schema_version") != "lumi.sft-dmath-curation-report.v0.1":
        raise ValueError("unsupported DMath curation report")
    if _verify_internal_seal(curation_report, "report_sha256") != (
        "c00b88c977a95f6bd8f49350a210dce0954c9255ca5b35730ab84001df3415c4"
    ):
        raise ValueError("DMath curation report seal is not approved")
    if curation_report.get("status") != "curated_nonproduction_owner_review_pending":
        raise ValueError("DMath curation status changed")
    token_measurement = curation_report.get("curation", {}).get("token_measurement")
    if not isinstance(token_measurement, Mapping):
        raise ValueError("DMath token measurement is missing")
    if (
        token_measurement.get("accepted_records") != 6943
        or token_measurement.get("supervised_tokens") != 51026
        or token_measurement.get("context_window") != 4096
        or token_measurement.get("assistant_only_loss_verified") is not True
        or token_measurement.get("overlength_rejected") != 0
    ):
        raise ValueError("DMath token measurement changed")
    for split, expected in EXPECTED_CURATED_METRICS["by_split"].items():
        observed = token_measurement.get("by_split", {}).get(split)
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"DMath {split} metrics changed")
    if curation_report.get("source") != {
        "dataset_id": "JiwooKimAR/dmath",
        "license_id": "mit",
        "object_path": "dataset/train.json",
        "object_sha256": (
            "8c18e591d4e81072107887c1297dcc5971757e548b7d79069e28f01ac85703d8"
        ),
        "revision": EXPECTED_SOURCE_AFTER["revision"],
        "source_id": "dmath",
        "source_rows": 7943,
        "test_split_accessed": False,
    }:
        raise ValueError("DMath source boundary changed")
    if curation_report.get("production") != {
        "candidate_registry_promotions": 0,
        "dataset_sealed": False,
        "production_eligible": False,
        "production_sft_authorized": False,
    } or curation_report.get("raw_training_text_present") is not False:
        raise ValueError("DMath curation production boundary changed")

    if curation_verification.get("schema_version") != (
        "lumi.sft-dmath-curation-verification.v0.1"
    ):
        raise ValueError("unsupported DMath curation verification")
    if _verify_internal_seal(curation_verification, "verification_sha256") != (
        "05a22df0cb0d84483a8bb9f0e6f395228ef3bc706c1ec57da253321a5eef2132"
    ):
        raise ValueError("DMath curation verification seal is not approved")
    if (
        curation_verification.get("status")
        != "verified_nonproduction_owner_review_pending"
        or curation_verification.get("production_eligible") is not False
        or curation_verification.get("retained_records") != 6943
        or curation_verification.get("supervised_tokens") != 51026
        or curation_verification.get("checks")
        != {
            "all_candidate_records_verifier_passed": True,
            "all_global_filters_replayed": True,
            "artifact_hashes_verified": True,
            "assistant_only_loss_verified": True,
            "benchmark_decontamination_replayed": True,
            "cross_split_exact_and_prompt_near_duplicates": 0,
            "frozen_tokenizer_measurement_replayed": True,
            "hidden_solution_fields_used_as_training_targets": False,
            "official_test_split_accessed": False,
            "owner_sample_records": 32,
            "source_registry_check_deferred_until_promotion": True,
        }
    ):
        raise ValueError("DMath curation verification boundary changed")

    if owner_review_receipt.get("schema_version") != (
        "lumi.sft-dmath-owner-review-receipt.v0.1"
    ):
        raise ValueError("unsupported DMath owner-review receipt")
    if _verify_internal_seal(owner_review_receipt, "receipt_sha256") != (
        "706787d549857bdf1136352fca3e157eb7e41fd9ac6da0a9d072d4504950597c"
    ):
        raise ValueError("DMath owner-review seal is not approved")
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
        raise ValueError("DMath owner-review gate did not pass")

    if owner_review_verification.get("schema_version") != (
        "lumi.sft-dmath-owner-review-verification.v0.1"
    ):
        raise ValueError("unsupported DMath owner-review verification")
    if _verify_internal_seal(owner_review_verification, "verification_sha256") != (
        "b475afe6c690893f085345ff28f55a94f6d5212b55e1389e15b4f03a8d72e288"
    ):
        raise ValueError("DMath owner-review verification seal is not approved")
    if (
        owner_review_verification.get("status")
        != "verified_owner_review_complete_nonproduction"
        or owner_review_verification.get("accepted_records") != 32
        or owner_review_verification.get("rejected_records") != 0
        or owner_review_verification.get("production_eligible") is not False
        or owner_review_verification.get("checks")
        != {
            "all_plan_bindings_verified": True,
            "all_sampled_rows_accepted": True,
            "decision_file_replayed": True,
            "official_test_split_accessed": False,
            "raw_training_text_retained": False,
            "receipt_seal_verified": True,
            "source_registry_promotions": 0,
        }
    ):
        raise ValueError("DMath owner-review verification boundary changed")

    if len(accepted_records) != 6943:
        raise ValueError("DMath accepted record count changed")
    record_ids: set[str] = set()
    split_counts: dict[str, int] = {"train": 0, "validation": 0}
    for row in accepted_records:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("DMath accepted record identity changed")
        source = row.get("source")
        messages = row.get("messages")
        split = row.get("split")
        if (
            row.get("schema_version") != "lumi.sft.v0.1"
            or row.get("language") != "en"
            or row.get("task_category") != "reasoning_math_science"
            or split not in split_counts
            or not isinstance(source, Mapping)
            or source.get("source_id") != "dmath"
            or source.get("dataset_id") != "JiwooKimAR/dmath"
            or source.get("license_id") != "mit"
            or source.get("revision") != EXPECTED_SOURCE_AFTER["revision"]
            or not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("DMath accepted record schema or provenance changed")
        if "solution_abst_en" in row or "solution_code_en" in row:
            raise ValueError("DMath hidden verifier fields leaked into training records")
        record_ids.add(record_id)
        split_counts[str(split)] += 1
    if split_counts != {"train": 6596, "validation": 347}:
        raise ValueError("DMath accepted record split counts changed")


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
    sources = predecessor_value.get("sources")
    if not isinstance(sources, list) or len(sources) != 10:
        raise ValueError("DMath predecessor source count changed")
    if any(
        isinstance(source, Mapping) and source.get("source_id") == "dmath"
        for source in sources
    ):
        raise ValueError("DMath already exists in the predecessor registry")
    successor = copy.deepcopy(dict(predecessor_value))
    successor["sources"].append(copy.deepcopy(plan["source_after"]))
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
        raise ValueError("DMath registry sources must be arrays")
    if len(before) != 10 or len(after) != 11:
        raise ValueError("DMath successor source count changed")
    if {key: value for key, value in predecessor_value.items() if key != "sources"} != {
        key: value for key, value in successor_value.items() if key != "sources"
    }:
        raise ValueError("DMath successor changed registry metadata")
    if after[:10] != before or after[10] != EXPECTED_SOURCE_AFTER:
        raise ValueError("DMath successor changed an unauthorized source")
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


def validate_records_against_successor(
    records: Sequence[Mapping[str, Any]],
    *,
    successor: SFTSourceRegistry,
) -> None:
    for row in records:
        source = row["source"]
        reasons = successor.record_reasons(
            dataset_id=source["dataset_id"],
            language=row["language"],
            license_id=source["license_id"],
            revision=source["revision"],
            source_id=source["source_id"],
            split=row["split"],
            task_category=row["task_category"],
        )
        if reasons:
            raise ValueError(f"DMath successor rejected an accepted record: {reasons}")


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
        raise ValueError("DMath registry lost candidate/canary task coverage")
    if before["missing_production_tasks"] != after["missing_production_tasks"]:
        raise ValueError("DMath promotion changed production coverage")
    if set(after["missing_production_tasks"]) != required_tasks:
        raise ValueError("DMath promotion unexpectedly enabled production")
    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "accepted_records_file_sha256": plan["bindings"]["accepted_records"][
                "sha256"
            ],
            "curation_report_file_sha256": plan["bindings"]["curation_report"][
                "sha256"
            ],
            "curation_report_internal_sha256": (
                "c00b88c977a95f6bd8f49350a210dce0954c9255ca5b35730ab84001df3415c4"
            ),
            "curation_verification_file_sha256": plan["bindings"][
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": (
                "05a22df0cb0d84483a8bb9f0e6f395228ef3bc706c1ec57da253321a5eef2132"
            ),
            "owner_review_file_sha256": plan["bindings"]["owner_review_receipt"][
                "sha256"
            ],
            "owner_review_internal_sha256": owner_review_receipt["receipt_sha256"],
            "owner_review_verification_file_sha256": plan["bindings"][
                "owner_review_verification"
            ]["sha256"],
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
            "added_source_id": "dmath",
            "changed_predecessor_sources": 0,
            "removed_sources": 0,
            "status_after": "canary_eligible",
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
            "reviewed_dmath_registered_in_immutable_candidate_successor_"
            "production_dataset_still_blocked"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def verify_source_promotion_receipt(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported DMath source-promotion receipt")
    _verify_internal_seal(value, "receipt_sha256")
    if (
        value.get("status")
        != (
            "reviewed_dmath_registered_in_immutable_candidate_successor_"
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
        raise ValueError("DMath source-promotion receipt boundary changed")
