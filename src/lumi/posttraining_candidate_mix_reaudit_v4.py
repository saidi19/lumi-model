"""Registry-v0.9-bound full local re-audit of the Lumi SFT candidate pool."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v3 as v3
from .posttraining_ai_v3_provenance_materialization import (
    OWNER_REVIEW_SCOPE,
    _verify_internal_sha,
)
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _targets,
    _verify_promotion_receipt,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.4"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.4"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.4"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_reaudit_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "registry_v0_9_full_local_candidate_mix_reaudit_only",
    "source_promotion_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 20_000,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.4.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.4.json"
    ),
}
EXPECTED_BINDINGS = {
    "ai_v3_provenance_candidate",
    "ai_v3_provenance_manifest",
    "ai_v3_provenance_registry_plan",
    "ai_v3_provenance_registry_report",
    "ai_v3_provenance_registry_verification",
    "implementation",
    "predecessor_plan",
    "predecessor_report",
    "predecessor_verification",
    "runner",
    "source_registry",
    "test",
    "verifier",
}
EXPECTED_REGISTRY_SHA256 = (
    "97b7a4399810f95d9c2e39dfe236a0129bd005abac81e067b67b8d0c0fb0baac"
)
EXPECTED_CANDIDATE_SHA256 = (
    "ebf019f0b417f5c1807bfa7d99a177f9c790567743932330a8e4849993b25d2f"
)
EXPECTED_RECORDS = 12_721
EXPECTED_TRAIN_TOKENS = 1_068_485
EXPECTED_VALIDATION_TOKENS = 92_657
AI_SOURCE_ID = "lumi_ai_assisted_sft_v0_3"


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.4 binding set changed")
    return {
        name: v2.resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _load_predecessor_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    path = v2.resolve_binding(
        repository_root,
        plan["bindings"]["predecessor_plan"],
        name="predecessor_plan",
    )
    return v3.load_candidate_mix_reaudit_v3_plan(path, repository_root=repository_root)


def load_candidate_mix_reaudit_v4_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_9_full_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.4 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    if plan["bindings"]["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.4 registry binding changed")
    if (
        plan["bindings"]["ai_v3_provenance_candidate"]["sha256"]
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise ValueError("candidate-mix v0.4 AI candidate binding changed")

    _load_predecessor_plan(plan, repository_root=repository_root)
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    registry_report = _load_json(resolved["ai_v3_provenance_registry_report"])
    registry_verification = _load_json(
        resolved["ai_v3_provenance_registry_verification"]
    )
    manifest = _load_json(resolved["ai_v3_provenance_manifest"])
    for value, field in (
        (predecessor_report, "report_sha256"),
        (predecessor_verification, "verification_sha256"),
        (registry_report, "report_sha256"),
        (registry_verification, "verification_sha256"),
        (manifest, "manifest_sha256"),
    ):
        _verify_internal_sha(value, field)
    if predecessor_report.get("status") != (
        "verified_registry_v0_8_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix v0.3 predecessor report changed")
    if predecessor_verification.get("status") != (
        "verified_registry_v0_8_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix v0.3 predecessor verification changed")
    if registry_report.get("status") != (
        "verified_ai_v0_3_provenance_bound_in_registry_v0_9_nonproduction"
    ):
        raise ValueError("AI v0.3 provenance-registry report changed")
    if registry_verification.get("status") != (
        "verified_ai_v0_3_provenance_registry_v0_9_nonproduction"
    ):
        raise ValueError("AI v0.3 provenance-registry verification changed")
    if manifest.get("owner_review", {}).get("scope") != OWNER_REVIEW_SCOPE:
        raise ValueError("AI v0.3 truthful owner-review scope changed")
    if manifest.get("owner_review", {}).get("sample_records") != 32:
        raise ValueError("AI v0.3 owner-review sample count changed")
    if manifest.get("owner_review", {}).get("unsampled_records") != 288:
        raise ValueError("AI v0.3 owner-review unsampled count changed")
    return plan


def _effective_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = v3._effective_plan(predecessor, repository_root=repository_root)
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"] = dict(effective["bindings"])
    effective["bindings"]["source_registry"] = dict(
        plan["bindings"]["source_registry"]
    )
    effective["bindings"]["ai_v3_candidate"] = dict(
        plan["bindings"]["ai_v3_provenance_candidate"]
    )
    ai_sources = [
        source
        for source in effective["candidate_sources"]
        if source["source_id"] == AI_SOURCE_ID
    ]
    if len(ai_sources) != 1:
        raise ValueError("candidate-mix v0.4 AI source plan changed")
    ai_sources[0]["candidate_file_sha256"] = EXPECTED_CANDIDATE_SHA256
    ai_sources[0]["expected_pending_owner_review_rows"] = 0
    return effective


def _validate_new_evidence(resolved: Mapping[str, Path]) -> None:
    registry_report = _load_json(resolved["ai_v3_provenance_registry_report"])
    registry_verification = _load_json(
        resolved["ai_v3_provenance_registry_verification"]
    )
    manifest = _load_json(resolved["ai_v3_provenance_manifest"])
    _verify_internal_sha(registry_report, "report_sha256")
    _verify_internal_sha(registry_verification, "verification_sha256")
    _verify_internal_sha(manifest, "manifest_sha256")
    if registry_report.get("bindings", {}).get("successor_registry_sha256") != (
        EXPECTED_REGISTRY_SHA256
    ):
        raise ValueError("AI v0.3 successor registry evidence changed")
    if registry_report.get("candidate_validation", {}).get("candidate_file_sha256") != (
        EXPECTED_CANDIDATE_SHA256
    ):
        raise ValueError("AI v0.3 repaired candidate evidence changed")
    if registry_report.get("candidate_validation", {}).get(
        "individually_reviewed_all_rows_claimed"
    ) is not False:
        raise ValueError("AI v0.3 owner-review evidence became overstated")


def rebuild_candidate_mix_reaudit_v4(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    predecessor_plan = _load_predecessor_plan(plan, repository_root=root)
    predecessor_report = v3.rebuild_candidate_mix_reaudit_v3(
        plan=predecessor_plan,
        plan_file_sha256=file_sha256(
            v2.resolve_binding(
                root,
                plan["bindings"]["predecessor_plan"],
                name="predecessor_plan",
            )
        ),
        repository_root=root,
    )
    bound_predecessor_report = _load_json(
        v2.resolve_binding(
            root,
            plan["bindings"]["predecessor_report"],
            name="predecessor_report",
        )
    )
    if predecessor_report != bound_predecessor_report:
        raise ValueError("candidate-mix v0.3 predecessor no longer reproduces")

    effective = _effective_plan(plan, repository_root=root)
    resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    resolved.update(_resolve_explicit_bindings(plan, root))
    v3._validate_bound_evidence(resolved)
    _validate_new_evidence(resolved)
    policy = load_sft_policy(resolved["data_contract"])
    benchmark_registry = v2.BenchmarkRegistry(policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != effective["bindings"]["benchmark_registry"]["sha256"]:
        raise ValueError("benchmark registry binding changed")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    targets = _targets(_load_json(resolved["missing_mix_proposal"]))
    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    verify_chat_tokenizer(tokenizer, template)

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    source_metrics: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    train_tokens = 0
    validation_tokens = 0
    task_tokens: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    for source_plan in sorted(
        effective["candidate_sources"],
        key=lambda item: item["source_id"],
    ):
        source_id = source_plan["source_id"]
        registry_source = registry.by_id.get(source_id)
        if registry_source is None:
            raise ValueError(f"candidate source is absent from registry: {source_id}")
        if registry_source.revision != source_plan["revision"]:
            raise ValueError(f"candidate source revision changed: {source_id}")
        if registry_source.status != source_plan["registry_status"]:
            raise ValueError(f"candidate source status changed: {source_id}")
        candidate_path = resolved[source_plan["candidate_binding"]]
        if file_sha256(candidate_path) != source_plan["candidate_file_sha256"]:
            raise ValueError(f"candidate file hash changed: {source_id}")
        rows = _load_jsonl(candidate_path)
        metrics = v3._candidate_metrics_v3(
            rows=rows,
            source_plan=source_plan,
            registry_source=registry_source,
            tokenizer=tokenizer,
            template=template,
            duplicate_index=duplicate_index,
        )
        if metrics["rows_requiring_split_normalization"] != 0:
            raise ValueError(f"{source_id} still requires split normalization")
        expected_pending = source_plan["expected_pending_owner_review_rows"]
        if metrics["rows_with_stale_pending_human_review_provenance"] != expected_pending:
            raise ValueError(f"{source_id} owner-review provenance count changed")
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        validation_tokens += metrics["by_split"]["validation"]["supervised_tokens"]
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_promotion_receipt(promotion, source_id=source_id)

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.4 row ceiling exceeded")
    accepted, generic = filter_sft_records(
        all_rows,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(accepted) != len(all_rows) or generic["rejected"] != 0:
        raise ValueError(
            "registry-v0.9 candidate pool failed generic validation: "
            f"{generic['rejection_counts']}"
        )
    for metrics in source_metrics.values():
        metrics["generic_sft_accepted_records"] = metrics["records"]
        metrics["generic_sft_rejected_records"] = 0

    exact_snapshot = _duplicate_snapshot(duplicate_index)
    if any(exact_snapshot.values()):
        raise ValueError("registry-v0.9 candidate pool has exact duplicate or split leak")
    near_snapshot = v2._near_duplicate_audit(
        all_rows,
        policy=effective["near_duplicate_policy"],
    )
    if near_snapshot["cross_source_components"] != 0:
        raise ValueError("registry-v0.9 pool has cross-source near duplicates")
    if near_snapshot["train_validation_components"] != 0:
        raise ValueError("registry-v0.9 pool has train-validation near leakage")
    if len(all_rows) != EXPECTED_RECORDS:
        raise ValueError("registry-v0.9 exact candidate record count changed")
    if train_tokens != EXPECTED_TRAIN_TOKENS:
        raise ValueError("registry-v0.9 train token count changed")
    if validation_tokens != EXPECTED_VALIDATION_TOKENS:
        raise ValueError("registry-v0.9 validation token count changed")

    source_shares = {
        source_id: round(
            metrics["by_split"]["train"]["supervised_tokens"] / train_tokens,
            8,
        )
        for source_id, metrics in sorted(source_metrics.items())
    }
    source_cap_violations = [
        source_id
        for source_id, fraction in source_shares.items()
        if fraction > targets["source_cap"]
    ]
    task_gaps = {
        task: max(0, targets["task_minimums"][task] - task_tokens[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, targets["language_minimums"][mode] - language_tokens[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }
    excluded = v2._verify_exclusions(
        plan=effective,
        resolved=resolved,
        registry=registry,
    )
    blockers = [
        "accepted_train_supervised_tokens_below_10m_floor",
        "task_token_floors_not_met",
        "language_token_floors_not_met",
        "source_concentration_cap_not_met",
        "within_split_near_duplicate_canonical_selection_not_materialized",
        "active_production_data_contract_not_updated",
        "dataset_seal_cpu_canary_and_sft_not_authorized",
    ]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_train_supervised_tokens": train_tokens,
            "candidate_records_read": len(all_rows),
            "floor_completion_fraction": round(train_tokens / targets["total"]["minimum"], 8),
            "gap_to_10m_train_supervised_tokens": targets["total"]["minimum"]
            - train_tokens,
            "registry_admissible_validation_supervised_tokens": validation_tokens,
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {task: task_tokens[task] for task in EXPECTED_TASKS},
        },
        "closed_since_v0_3": [
            "lumi_ai_assisted_sft_v0_3_all_320_rows_bind_owner_review_receipt",
            "lumi_ai_assisted_sft_v0_3_owner_review_scope_is_32_sampled_288_unsampled",
            "lumi_ai_assisted_sft_v0_3_pending_provenance_blocker_removed",
            "lumi_ai_assisted_sft_v0_3_repaired_candidate_bound_in_registry_v0_9",
        ],
        "closure": {
            "blockers": blockers,
            "language_floor_gaps": language_gaps,
            "production_dataset_ready": False,
            "source_cap": targets["source_cap"],
            "source_cap_violations": source_cap_violations,
            "task_floor_gaps": task_gaps,
        },
        "exact_duplicate_audit": exact_snapshot,
        "excluded_sources": excluded,
        "generic_sft_validation": {
            "accepted": generic["accepted"],
            "accepted_dataset_sha256": generic["accepted_dataset_sha256"],
            "benchmark_contamination_rejections": sum(
                count
                for reason, count in generic["rejection_counts"].items()
                if reason.startswith("benchmark_contamination:")
            ),
            "raw_text_retained": generic["raw_text_retained"],
            "rejected": generic["rejected"],
            "rejection_counts": generic["rejection_counts"],
            "report_sha256": generic["report_sha256"],
            "source_registry_rejections": sum(
                count
                for reason, count in generic["rejection_counts"].items()
                if reason.startswith("source_registry")
            ),
            "total": generic["total"],
        },
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "continue_rights_clear_bulk_mix_acquisition_to_close_10m_"
                "task_language_and_source_balance_gaps"
            ),
            "authorization_required": True,
            "priority_gaps": [
                "reasoning_math_science",
                "coding",
                "factual_qa",
                "writing_summarization",
                "safety_helpfulness",
                "tool_api",
                "bengali_script",
                "banglish",
                "natural_code_switching",
            ],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry": {
            "path": plan["bindings"]["source_registry"]["path"],
            "sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(registry.by_id),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "verified_registry_v0_9_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v4_report(report)
    return report


def validate_candidate_mix_reaudit_v4_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.4 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.4 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.4 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.4 report must remain raw-text-free")
    pool = report.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("candidate-mix v0.4 pool summary is missing")
    expected_pool = {
        "accepted_train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "candidate_records_read": EXPECTED_RECORDS,
        "gap_to_10m_train_supervised_tokens": 8_931_515,
        "registry_admissible_validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        "source_count": 6,
    }
    for field, value in expected_pool.items():
        if pool.get(field) != value:
            raise ValueError(f"candidate-mix v0.4 pool changed: {field}")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("candidate-mix v0.4 generic validation is missing")
    if generic.get("accepted") != EXPECTED_RECORDS or generic.get("rejected") != 0:
        raise ValueError("candidate-mix v0.4 generic validation changed")
    if generic.get("rejection_counts") != {}:
        raise ValueError("candidate-mix v0.4 generic validation has rejection reasons")
    exact = report.get("exact_duplicate_audit")
    if not isinstance(exact, Mapping) or any(exact.values()):
        raise ValueError("candidate-mix v0.4 exact duplicate audit is not clean")
    near = report.get("near_duplicate_audit")
    if not isinstance(near, Mapping):
        raise ValueError("candidate-mix v0.4 near-duplicate audit is missing")
    if near.get("cross_source_components") != 0:
        raise ValueError("candidate-mix v0.4 has cross-source near duplicates")
    if near.get("train_validation_components") != 0:
        raise ValueError("candidate-mix v0.4 has train-validation near leakage")
    ai_metrics = report.get("source_metrics", {}).get(AI_SOURCE_ID)
    if not isinstance(ai_metrics, Mapping):
        raise ValueError("candidate-mix v0.4 AI source metrics are missing")
    if ai_metrics.get("rows_with_stale_pending_human_review_provenance") != 0:
        raise ValueError("candidate-mix v0.4 AI provenance repair was not observed")
    blockers = report.get("closure", {}).get("blockers")
    if not isinstance(blockers, list) or len(blockers) != 7:
        raise ValueError("candidate-mix v0.4 blocker set changed")
    if any("owner_review_provenance" in blocker for blocker in blockers):
        raise ValueError("candidate-mix v0.4 retained the closed provenance blocker")


def build_candidate_mix_reaudit_v4_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v4_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_12721_rows_pass_schema_registry_benchmark_and_token_validation": True,
            "ai_v0_3_exact_repaired_320_rows_are_registry_admissible": True,
            "ai_v0_3_owner_review_scope_remains_32_sampled_288_unsampled": True,
            "ai_v0_3_pending_provenance_rows_are_zero": True,
            "exact_duplicate_and_train_validation_overlap_are_zero": True,
            "frozen_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "predecessor_v0_3_reaudit_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_v0_9_hash_verified": True,
            "whole_conversation_near_audit_has_no_cross_source_or_split_leakage": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256,
            "report_file_sha256",
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_9_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
