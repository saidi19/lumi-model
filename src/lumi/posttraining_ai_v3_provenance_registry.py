"""Immutable candidate-registry successor for the repaired Lumi AI v0.3 source."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import (
    OWNER_REVIEW_CLAIM,
    OWNER_REVIEW_SCOPE,
    SOURCE_ID,
    SOURCE_REVISION,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _verify_internal_sha,
    bytes_sha256,
    file_sha256,
    json_file_bytes,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_sources import SFTSource, SFTSourceRegistry

PLAN_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-registry-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-registry-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-registry-verification.v0.1"

PREDECESSOR_REGISTRY_SHA256 = (
    "d20c89ca34a284fc120a115d0ac9578c7e03ae3c915452d3f0ea144928e9a519"
)
SUCCESSOR_REGISTRY_SHA256 = (
    "97b7a4399810f95d9c2e39dfe236a0129bd005abac81e067b67b8d0c0fb0baac"
)
PREDECESSOR_ADAPTER = "lumi_ai_assisted_chat_v0.3"
SUCCESSOR_ADAPTER = "lumi_ai_assisted_chat_v0.3_owner_spot_check_bound_v0.1"
REMOVED_BLOCKER = "per_record_owner_review_provenance_not_materialized"
REMAINING_SOURCE_BLOCKERS = [
    "candidate_is_a_small_behavior_pilot_not_the_complete_sft_mix",
    "full_mix_cross_source_deduplication_not_complete",
    "production_dataset_seal_and_sft_authorization_not_complete",
]

EXPECTED_AUTHORIZATION = {
    "active_production_data_contract_update_authorized": False,
    "candidate_row_modification_authorized": False,
    "candidate_source_registry_successor_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "bind_verified_ai_v0_3_provenance_candidate_in_registry_v0_9_only",
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 0,
    "changed_source_entries": 1,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 13,
    "removed_sources": 0,
    "successor_sources": 13,
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "report_path": "evidence/lumi-4b-sft-ai-v3-provenance-registry-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.9.json",
    "verification_path": (
        "evidence/lumi-4b-sft-ai-v3-provenance-registry-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = {
    "data_contract",
    "implementation",
    "predecessor_registry",
    "provenance_candidate",
    "provenance_manifest",
    "provenance_plan",
    "provenance_report",
    "provenance_verification",
    "runner",
    "source_registry_implementation",
    "test",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid AI v0.3 registry binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"AI v0.3 registry binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"AI v0.3 registry binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"AI v0.3 registry binding changed: {name}")
    return resolved


def resolve_output(repository_root: str | Path, relative: str, *, name: str) -> Path:
    root = Path(repository_root).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"AI v0.3 registry output escapes repository: {name}")
    return resolved


def load_ai_v3_provenance_registry_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "expected": {
            "predecessor_adapter": PREDECESSOR_ADAPTER,
            "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
            "remaining_source_blockers": REMAINING_SOURCE_BLOCKERS,
            "removed_blocker": REMOVED_BLOCKER,
            "source_id": SOURCE_ID,
            "source_revision": SOURCE_REVISION,
            "successor_adapter": SUCCESSOR_ADAPTER,
            "successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
        },
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_ai_v0_3_provenance_registry_successor",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"AI v0.3 provenance-registry plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("AI v0.3 provenance-registry binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid AI v0.3 provenance-registry binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("AI v0.3 predecessor registry binding changed")
    for name in ("report_path", "successor_registry_path", "verification_path"):
        resolve_output(repository_root, plan["output"][name], name=name)
    return plan


def _validate_materialization_evidence(resolved: Mapping[str, Path]) -> None:
    manifest = _load_json(resolved["provenance_manifest"])
    report = _load_json(resolved["provenance_report"])
    verification = _load_json(resolved["provenance_verification"])
    _verify_internal_sha(manifest, "manifest_sha256")
    _verify_internal_sha(report, "report_sha256")
    _verify_internal_sha(verification, "verification_sha256")
    if manifest.get("status") != (
        "private_ai_v0_3_provenance_candidate_materialized_nonproduction"
    ):
        raise ValueError("AI v0.3 provenance manifest status changed")
    if report.get("status") != (
        "verified_ai_v0_3_owner_spot_check_provenance_materialized_nonproduction"
    ):
        raise ValueError("AI v0.3 provenance report status changed")
    if verification.get("status") != (
        "verified_ai_v0_3_provenance_materialization_nonproduction"
    ):
        raise ValueError("AI v0.3 provenance verification status changed")
    candidate_sha256 = file_sha256(resolved["provenance_candidate"])
    if candidate_sha256 != manifest.get("artifacts", {}).get("candidate", {}).get(
        "sha256"
    ):
        raise ValueError("AI v0.3 provenance candidate-manifest binding changed")
    if candidate_sha256 != report.get("bindings", {}).get("candidate_file_sha256"):
        raise ValueError("AI v0.3 provenance candidate-report binding changed")
    if file_sha256(resolved["provenance_manifest"]) != report.get("bindings", {}).get(
        "manifest_file_sha256"
    ):
        raise ValueError("AI v0.3 provenance manifest-report binding changed")
    if report.get("production_eligible") is not False:
        raise ValueError("AI v0.3 provenance report unexpectedly became production eligible")
    owner_review = manifest.get("owner_review")
    if owner_review != {
        "claim": OWNER_REVIEW_CLAIM,
        "individually_reviewed_all_rows_claimed": False,
        "sample_records": 32,
        "sampled_record_ids_sha256": (
            "3651ece6e6757ba89651e94266820222a2d1e8d7e907c5cbc12c4359c7034398"
        ),
        "scope": OWNER_REVIEW_SCOPE,
        "unsampled_records": 288,
    }:
        raise ValueError("AI v0.3 owner-review scope binding changed")


def _build_successor(
    predecessor: Mapping[str, Any],
    *,
    policy: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    successor = copy.deepcopy(predecessor)
    sources = successor.get("sources")
    predecessor_sources = predecessor.get("sources")
    if not isinstance(sources, list) or not isinstance(predecessor_sources, list):
        raise ValueError("AI v0.3 predecessor registry source list changed")
    if len(sources) != 13:
        raise ValueError("AI v0.3 predecessor registry source count changed")
    source_before: dict[str, Any] | None = None
    source_after: dict[str, Any] | None = None
    changed = 0
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or source.get("source_id") != SOURCE_ID:
            continue
        source_before = copy.deepcopy(predecessor_sources[index])
        if source.get("revision") != SOURCE_REVISION:
            raise ValueError("AI v0.3 predecessor source revision changed")
        if source.get("adapter") != PREDECESSOR_ADAPTER:
            raise ValueError("AI v0.3 predecessor source adapter changed")
        blockers = source.get("production_blockers")
        if not isinstance(blockers, list) or blockers != [
            REMOVED_BLOCKER,
            *REMAINING_SOURCE_BLOCKERS,
        ]:
            raise ValueError("AI v0.3 predecessor production blockers changed")
        source["adapter"] = SUCCESSOR_ADAPTER
        source["production_blockers"] = list(REMAINING_SOURCE_BLOCKERS)
        source_after = copy.deepcopy(source)
        changed += 1
    if changed != 1 or source_before is None or source_after is None:
        raise ValueError("exactly one AI v0.3 registry source must change")
    for before, after in zip(predecessor_sources, sources, strict=True):
        if before.get("source_id") == SOURCE_ID:
            changed_fields = {
                field
                for field in set(before) | set(after)
                if before.get(field) != after.get(field)
            }
            if changed_fields != {"adapter", "production_blockers"}:
                raise ValueError("AI v0.3 successor changed unauthorized source fields")
        elif before != after:
            raise ValueError("AI v0.3 successor changed an unrelated source")
    parsed = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in sources
    )
    if len(parsed) != 13:
        raise ValueError("AI v0.3 successor registry parsing changed")
    return successor, source_before, source_after


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Any,
    registry: SFTSourceRegistry,
) -> dict[str, Any]:
    if len(rows) != 320:
        raise ValueError("AI v0.3 provenance candidate row count changed")
    sampled = 0
    unsampled = 0
    for row in rows:
        source = row.get("source")
        provenance = row.get("provenance")
        if not isinstance(source, Mapping) or source.get("source_id") != SOURCE_ID:
            raise ValueError("AI v0.3 candidate source changed")
        if source.get("revision") != SOURCE_REVISION:
            raise ValueError("AI v0.3 candidate source revision changed")
        if not isinstance(provenance, Mapping):
            raise ValueError("AI v0.3 candidate provenance changed")
        expected = {
            "human_review_status": "owner_spot_check_passed",
            "owner_review_claim": OWNER_REVIEW_CLAIM,
            "owner_review_sample_records": 32,
            "owner_review_scope": OWNER_REVIEW_SCOPE,
        }
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValueError(f"AI v0.3 candidate provenance changed: {field}")
        is_sampled = provenance.get("owner_review_sampled_record")
        if not isinstance(is_sampled, bool):
            raise ValueError("AI v0.3 candidate sampled-record flag changed")
        sampled += int(is_sampled)
        unsampled += int(not is_sampled)
    accepted, generic = filter_sft_records(
        rows,
        policy=policy,
        source_registry=registry,
    )
    if len(accepted) != 320 or generic.get("rejected") != 0:
        raise ValueError("AI v0.3 candidate failed successor-registry validation")
    if sampled != 32 or unsampled != 288:
        raise ValueError("AI v0.3 candidate owner-review sample counts changed")
    return {
        "accepted": len(accepted),
        "generic_report_sha256": generic["report_sha256"],
        "records": len(rows),
        "sampled_records": sampled,
        "unsampled_records": unsampled,
    }


def build_ai_v3_provenance_registry_outputs(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    _validate_materialization_evidence(resolved)
    policy = load_sft_policy(resolved["data_contract"])
    predecessor = _load_json(resolved["predecessor_registry"])
    successor, source_before, source_after = _build_successor(
        predecessor,
        policy=policy,
    )
    successor_bytes = json_file_bytes(successor)
    successor_sha256 = bytes_sha256(successor_bytes)
    if successor_sha256 != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("AI v0.3 successor registry hash changed")
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
    candidate_metrics = _validate_candidate_rows(
        _load_jsonl(resolved["provenance_candidate"]),
        policy=policy,
        registry=registry,
    )
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "predecessor_registry_sha256": PREDECESSOR_REGISTRY_SHA256,
            "provenance_candidate_file_sha256": file_sha256(
                resolved["provenance_candidate"]
            ),
            "provenance_manifest_file_sha256": file_sha256(
                resolved["provenance_manifest"]
            ),
            "provenance_report_file_sha256": file_sha256(
                resolved["provenance_report"]
            ),
            "provenance_verification_file_sha256": file_sha256(
                resolved["provenance_verification"]
            ),
            "successor_registry_sha256": successor_sha256,
        },
        "candidate_validation": {
            **candidate_metrics,
            "candidate_file_sha256": file_sha256(resolved["provenance_candidate"]),
            "individually_reviewed_all_rows_claimed": False,
            "owner_review_claim": OWNER_REVIEW_CLAIM,
            "owner_review_scope": OWNER_REVIEW_SCOPE,
        },
        "candidate_registry_source_updates": 1,
        "observed_at": plan["observed_at"],
        "predecessor_registry_unchanged": True,
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry_delta": {
            "adapter_after": source_after["adapter"],
            "adapter_before": source_before["adapter"],
            "added_sources": 0,
            "changed_fields": ["adapter", "production_blockers"],
            "changed_source_entries": 1,
            "changed_source_id": SOURCE_ID,
            "remaining_source_blockers": source_after["production_blockers"],
            "removed_blocker": REMOVED_BLOCKER,
            "removed_sources": 0,
            "source_revision_unchanged": True,
        },
        "remaining_blockers": [
            *REMAINING_SOURCE_BLOCKERS,
            "candidate_mix_reaudit_with_registry_v0_9_not_complete",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_ai_v0_3_provenance_bound_in_registry_v0_9_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_ai_v3_provenance_registry_report(report)
    return successor_bytes, report


def validate_ai_v3_provenance_registry_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 provenance-registry report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("AI v0.3 provenance-registry report cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("AI v0.3 provenance-registry gate promoted production data")
    if report.get("predecessor_registry_unchanged") is not True:
        raise ValueError("AI v0.3 predecessor registry was not preserved")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("AI v0.3 registry report must remain raw-text-free")
    delta = report.get("registry_delta")
    if not isinstance(delta, Mapping):
        raise ValueError("AI v0.3 registry delta is missing")
    expected = {
        "adapter_after": SUCCESSOR_ADAPTER,
        "adapter_before": PREDECESSOR_ADAPTER,
        "added_sources": 0,
        "changed_fields": ["adapter", "production_blockers"],
        "changed_source_entries": 1,
        "changed_source_id": SOURCE_ID,
        "remaining_source_blockers": REMAINING_SOURCE_BLOCKERS,
        "removed_blocker": REMOVED_BLOCKER,
        "removed_sources": 0,
        "source_revision_unchanged": True,
    }
    if dict(delta) != expected:
        raise ValueError("AI v0.3 registry delta changed")
    candidate = report.get("candidate_validation")
    if not isinstance(candidate, Mapping):
        raise ValueError("AI v0.3 candidate validation is missing")
    if candidate.get("accepted") != 320 or candidate.get("records") != 320:
        raise ValueError("AI v0.3 successor-registry candidate count changed")
    if candidate.get("sampled_records") != 32 or candidate.get("unsampled_records") != 288:
        raise ValueError("AI v0.3 truthful owner-review counts changed")
    if candidate.get("individually_reviewed_all_rows_claimed") is not False:
        raise ValueError("AI v0.3 report overstated human review")


def build_ai_v3_provenance_registry_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_ai_v3_provenance_registry_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "candidate_rows_modified": False,
            "candidate_validates_against_successor_registry": True,
            "exactly_one_source_entry_changed": True,
            "only_adapter_and_production_blockers_changed": True,
            "owner_review_scope_remains_truthful_32_sampled_288_unsampled": True,
            "predecessor_registry_unchanged": True,
            "provenance_materialization_evidence_bound": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_source_revision_unchanged": True,
            "removed_only_materialized_provenance_blocker": True,
            "zero_network_api_gpu_seal_sft_or_production_promotion": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256,
            "report_file_sha256",
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_ai_v0_3_provenance_registry_v0_9_nonproduction",
        "successor_registry_sha256": SUCCESSOR_REGISTRY_SHA256,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
