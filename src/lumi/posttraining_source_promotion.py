"""Fail-closed candidate-only source-registry promotion for Lumi SFT data."""

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

PLAN_SCHEMA_VERSION = "lumi.safety-tool-source-promotion-plan.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.safety-tool-source-promotion-receipt.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_source_registry_promotion_authorized": True,
    "existing_registry_mutation_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_candidate_source_successor_registry_only",
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "changed_predecessor_sources": 0,
    "predecessor_sources": 9,
    "removed_sources": 0,
    "successor_sources": 10,
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "receipt_contains_raw_training_text": False,
    "receipt_path": "evidence/lumi-4b-safety-tool-source-promotion-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.2.json",
}
EXPECTED_SOURCE_ENTRY = {
    "access": "first_party",
    "adapter": "lumi_safety_tool_chat_v0.1",
    "blockers": [],
    "dataset_id": "Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1",
    "evidence_urls": [],
    "languages": ["bn", "en", "multilingual"],
    "license_id": "lumi-owned",
    "origin": "synthetic",
    "production_blockers": [
        "candidate_is_a_small_safety_tool_lane_not_the_complete_sft_mix",
        "full_mix_deduplication_and_benchmark_decontamination_not_complete",
        "production_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": None,
    "revision": "88765d93ed9df2effaf977ea535e901a5bf987abcfb511286f783679b7788915",
    "source_id": "lumi_safety_tool_sft_v0_1",
    "splits": ["train", "validation"],
    "status": "candidate_eligible",
    "task_categories": ["safety_helpfulness", "tool_api"],
}
EXPECTED_BINDINGS = frozenset(
    {
        "candidate_evidence",
        "data_contract",
        "owner_review_receipt",
        "predecessor_registry",
        "promotion_implementation",
        "promotion_runner",
        "source_registry_implementation",
        "token_audit",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    """Return the canonical JSON SHA-256 of a value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def json_file_bytes(value: Any) -> bytes:
    """Return the deterministic pretty-printed bytes used for JSON artifacts."""

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
    """Resolve one repository-local promotion binding without path escape."""

    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("source-promotion binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("source-promotion binding escapes repository")
    return resolved


def load_source_promotion_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load and verify the exact candidate-only promotion plan."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool source-promotion plan")
    if plan.get("status") != "authorized_candidate_only_successor_registry_promotion":
        raise ValueError("safety/tool source-promotion status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("safety/tool source-promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("safety/tool source-promotion limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("safety/tool source-promotion output boundary changed")
    if plan.get("source_entry") != EXPECTED_SOURCE_ENTRY:
        raise ValueError("safety/tool source-promotion entry changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("safety/tool source-promotion observation date changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("safety/tool source-promotion binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"source-promotion binding must be an object: {name}")
        expected_sha256 = _require_sha256(
            binding.get("sha256"), f"bindings.{name}.sha256"
        )
        resolved = resolve_promotion_binding(root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"safety/tool source-promotion binding changed: {name}")
    return plan


def verify_source_promotion_prerequisites(
    *,
    candidate_evidence: Mapping[str, Any],
    owner_review_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    token_audit: Mapping[str, Any],
) -> None:
    """Verify the machine, tokenizer, and owner gates for candidate registration."""

    if candidate_evidence.get("schema_version") != "lumi.safety-tool-sft-evidence.v0.1":
        raise ValueError("unsupported safety/tool candidate evidence")
    candidate_seal = _verify_internal_seal(candidate_evidence, "evidence_sha256")
    if candidate_seal != "62605cc07466ffc9c583655e3a85a3ad64c2bd4ef95a0c95deeb68d72823b57b":
        raise ValueError("safety/tool candidate evidence seal is not approved")
    if candidate_evidence.get("candidate_source_revision") != plan["source_entry"][
        "revision"
    ]:
        raise ValueError("safety/tool candidate source revision changed")
    counts = candidate_evidence.get("counts")
    if not isinstance(counts, Mapping) or counts.get("records") != 320:
        raise ValueError("safety/tool candidate record count changed")
    if counts.get("splits") != {"train": 240, "validation": 80}:
        raise ValueError("safety/tool candidate split counts changed")
    if counts.get("tasks") != {"safety_helpfulness": 160, "tool_api": 160}:
        raise ValueError("safety/tool candidate task counts changed")
    if (
        candidate_evidence.get("production_eligible") is not False
        or candidate_evidence.get("production_promotions") != 0
        or candidate_evidence.get("raw_training_text_retained") is not False
    ):
        raise ValueError("safety/tool candidate evidence boundary changed")

    if token_audit.get("schema_version") != (
        "lumi.safety-tool-sft-token-audit-report.v0.1"
    ):
        raise ValueError("unsupported safety/tool token audit")
    token_seal = _verify_internal_seal(token_audit, "report_sha256")
    if token_seal != "95496eaced0c6cc1e8ea53a57b7adbf6a8e40df924bfb46731651a1ba63ce658":
        raise ValueError("safety/tool token-audit seal is not approved")
    metrics = token_audit.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("safety/tool token-audit metrics are missing")
    if (
        metrics.get("accepted_records") != 320
        or metrics.get("rejected_records") != 0
        or metrics.get("overlength_records") != 0
        or metrics.get("serialized_tokens") != 39_325
        or metrics.get("supervised_tokens") != 25_024
    ):
        raise ValueError("safety/tool token-audit acceptance metrics changed")
    if metrics.get("by_split", {}).get("train", {}).get("supervised_tokens") != 18_636:
        raise ValueError("safety/tool train supervised-token count changed")
    if (
        token_audit.get("candidate_source_revision")
        != plan["source_entry"]["revision"]
        or token_audit.get("production_eligible") is not False
        or token_audit.get("production_promotions") != 0
        or token_audit.get("raw_training_text_retained") is not False
    ):
        raise ValueError("safety/tool token-audit boundary changed")

    if owner_review_receipt.get("schema_version") != (
        "lumi.safety-tool-owner-review-receipt.v0.1"
    ):
        raise ValueError("unsupported safety/tool owner-review receipt")
    owner_seal = _verify_internal_seal(owner_review_receipt, "receipt_sha256")
    if owner_seal != "369a3110a46de6189a7311eee946daa7092cf04cb9676dfb8ea67cfb41e838fb":
        raise ValueError("safety/tool owner-review seal is not approved")
    if (
        owner_review_receipt.get("status")
        != "owner_spot_check_passed_source_registry_promotion_still_separate"
        or owner_review_receipt.get("sample_records") != 40
        or owner_review_receipt.get("accepted_records") != 40
        or owner_review_receipt.get("rejected_records") != 0
        or owner_review_receipt.get("all_sampled_rows_accepted") is not True
        or owner_review_receipt.get("production_eligible") is not False
        or owner_review_receipt.get("production_promotions") != 0
        or owner_review_receipt.get("raw_training_text_retained") is not False
    ):
        raise ValueError("safety/tool owner-review gate did not pass")


def load_promotion_policy_and_predecessor(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[SFTPolicy, SFTSourceRegistry, dict[str, Any]]:
    """Load the old active contract and exact immutable predecessor registry."""

    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    policy_path = resolve_promotion_binding(root, bindings["data_contract"])
    registry_path = resolve_promotion_binding(root, bindings["predecessor_registry"])
    policy = load_sft_policy(policy_path)
    if (root / policy.source_registry_path).resolve() != registry_path:
        raise ValueError("active data contract does not bind the predecessor registry")
    if policy.source_registry_sha256 != bindings["predecessor_registry"]["sha256"]:
        raise ValueError("active data contract predecessor SHA-256 changed")
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
    """Append exactly one candidate source without changing predecessor entries."""

    raw_sources = predecessor_value.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("predecessor source list is invalid")
    if len(raw_sources) != EXPECTED_LIMITS["predecessor_sources"]:
        raise ValueError("predecessor source count changed")
    source_ids = {item.get("source_id") for item in raw_sources if isinstance(item, Mapping)}
    dataset_ids = {
        item.get("dataset_id") for item in raw_sources if isinstance(item, Mapping)
    }
    if plan["source_entry"]["source_id"] in source_ids:
        raise ValueError("successor source already exists")
    if plan["source_entry"]["dataset_id"] in dataset_ids:
        raise ValueError("successor dataset already exists")

    successor = copy.deepcopy(dict(predecessor_value))
    successor["sources"].append(copy.deepcopy(plan["source_entry"]))
    return successor


def validate_successor_registry(
    *,
    policy: SFTPolicy,
    predecessor_value: Mapping[str, Any],
    successor_value: Mapping[str, Any],
    successor_file_sha256: str,
) -> SFTSourceRegistry:
    """Validate exact append-only delta and every source under the frozen policy."""

    _require_sha256(successor_file_sha256, "successor_file_sha256")
    predecessor_without_sources = {
        key: value for key, value in predecessor_value.items() if key != "sources"
    }
    successor_without_sources = {
        key: value for key, value in successor_value.items() if key != "sources"
    }
    if predecessor_without_sources != successor_without_sources:
        raise ValueError("successor changed predecessor registry metadata")

    predecessor_sources = predecessor_value.get("sources")
    successor_sources = successor_value.get("sources")
    if not isinstance(predecessor_sources, list) or not isinstance(successor_sources, list):
        raise ValueError("source registry entries must be arrays")
    if successor_sources[:-1] != predecessor_sources:
        raise ValueError("successor changed or reordered predecessor sources")
    if len(successor_sources) != EXPECTED_LIMITS["successor_sources"]:
        raise ValueError("successor source count changed")
    if successor_sources[-1] != EXPECTED_SOURCE_ENTRY:
        raise ValueError("successor candidate entry changed")

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
    source_ids = [source.source_id for source in parsed]
    dataset_ids = [source.dataset_id for source in parsed]
    if len(source_ids) != len(set(source_ids)) or len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("successor source or dataset IDs are not unique")
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
    token_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a raw-text-free receipt for the candidate-only registry delta."""

    _require_sha256(plan_file_sha256, "plan_file_sha256")
    required_tasks = frozenset(plan["required_tasks"])
    before = build_sft_source_readiness_report(
        predecessor,
        required_tasks=required_tasks,
    )
    after = build_sft_source_readiness_report(successor, required_tasks=required_tasks)
    if before["missing_canary_tasks"] != ["safety_helpfulness"]:
        raise ValueError("predecessor canary gap changed")
    if after["missing_canary_tasks"]:
        raise ValueError("successor still lacks candidate coverage")
    if set(after["missing_production_tasks"]) != required_tasks:
        raise ValueError("candidate promotion unexpectedly changed production coverage")

    receipt: dict[str, Any] = {
        "active_production_data_contract_updated": False,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_evidence_file_sha256": plan["bindings"]["candidate_evidence"][
                "sha256"
            ],
            "candidate_evidence_internal_sha256": (
                "62605cc07466ffc9c583655e3a85a3ad64c2bd4ef95a0c95deeb68d72823b57b"
            ),
            "owner_review_file_sha256": plan["bindings"]["owner_review_receipt"][
                "sha256"
            ],
            "owner_review_internal_sha256": owner_review_receipt["receipt_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": predecessor.registry_sha256,
            "successor_registry_sha256": successor.registry_sha256,
            "token_audit_file_sha256": plan["bindings"]["token_audit"]["sha256"],
            "token_audit_internal_sha256": token_audit["report_sha256"],
        },
        "candidate_source_registry_promotions": 1,
        "counts": {
            "added_sources": len(successor.sources) - len(predecessor.sources),
            "changed_predecessor_sources": 0,
            "predecessor_sources": len(predecessor.sources),
            "removed_sources": 0,
            "successor_sources": len(successor.sources),
        },
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "readiness_delta": {
            "missing_candidate_or_canary_tasks_after": after["missing_canary_tasks"],
            "missing_candidate_or_canary_tasks_before": before["missing_canary_tasks"],
            "missing_production_tasks_after": after["missing_production_tasks"],
        },
        "remaining_blockers": [
            "complete_10m_to_30m_learning_mix_not_ready",
            "full_mix_deduplication_and_benchmark_decontamination_not_complete",
            "active_production_data_contract_not_updated",
            "production_dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": copy.deepcopy(plan["source_entry"]),
        "status": (
            "candidate_source_registered_in_immutable_successor_"
            "production_dataset_still_blocked"
        ),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def verify_source_promotion_receipt(value: Mapping[str, Any]) -> None:
    """Verify a completed raw-text-free candidate promotion receipt."""

    if value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool source-promotion receipt")
    _verify_internal_seal(value, "receipt_sha256")
    if (
        value.get("status")
        != "candidate_source_registered_in_immutable_successor_production_dataset_still_blocked"
        or value.get("candidate_source_registry_promotions") != 1
        or value.get("production_dataset_promotions") != 0
        or value.get("production_eligible") is not False
        or value.get("active_production_data_contract_updated") is not False
        or value.get("raw_training_text_retained") is not False
        or value.get("counts") != EXPECTED_LIMITS
        or value.get("source") != EXPECTED_SOURCE_ENTRY
    ):
        raise ValueError("safety/tool source-promotion receipt boundary changed")
