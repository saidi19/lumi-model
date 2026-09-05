"""Registry-v0.10-bound full local re-audit of the Lumi SFT candidate pool."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v3 as v3
from . import posttraining_candidate_mix_reaudit_v4 as v4
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
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
from .posttraining_deepmind_scale_owner_review import (
    validate_owner_review_receipt,
)
from .posttraining_deepmind_scale_source_promotion import (
    validate_promotion_receipt as validate_deepmind_scale_promotion,
)
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.5"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.5"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.5"

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "local_candidate_read_authorized": True,
    "local_raw_text_free_reaudit_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "registry_v0_10_full_local_candidate_mix_reaudit_only",
    "source_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 300_000,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 180,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.5.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.5.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "deepmind_scale_candidate",
        "deepmind_scale_owner_review",
        "deepmind_scale_promotion",
        "deepmind_scale_promotion_verification",
        "deepmind_scale_token_metrics",
        "implementation",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)
EXPECTED_REGISTRY_SHA256 = (
    "272b88f69ae19da25d3e979cf9d74ab0d4c8a1e6d276b9adcc148fd10738490f"
)
EXPECTED_DEEPMIND_CANDIDATE_SHA256 = (
    "5601aab088a11c2664e0e8067b509495bf3788e07b4e8e3fa548afef7ad1c7a9"
)
EXPECTED_DEEPMIND_PROMOTION_INTERNAL_SHA256 = (
    "713fb282ccb85cc1c1d1a2901ae40421c0d77a7e8cd4943e7373a94d524eee1f"
)
EXPECTED_DEEPMIND_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "c2d2cbb37554baacf8155ec1f97b8cdc37405c1d7a2ff5895887dbd3ed5a3e6e"
)
EXPECTED_DEEPMIND_OWNER_REVIEW_INTERNAL_SHA256 = (
    "97c898d6c16f55f5620ecd1ab22f8836f445c5d1ed73eb1303d073887df1d44f"
)
EXPECTED_RECORDS = 247_504
EXPECTED_TRAIN_TOKENS = 3_271_793
EXPECTED_VALIDATION_TOKENS = 209_784
DEEPMIND_SOURCE_ID = "deepmind_mathematics_dataset"


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.5 binding set changed")
    return {
        name: v2.resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _load_predecessor_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    path = v2.resolve_binding(
        repository_root,
        plan["bindings"]["predecessor_plan"],
        name="predecessor_plan",
    )
    return v4.load_candidate_mix_reaudit_v4_plan(
        path, repository_root=repository_root
    )


def load_candidate_mix_reaudit_v5_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_10_full_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.5 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    if plan["bindings"]["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.5 registry binding changed")
    if (
        plan["bindings"]["deepmind_scale_candidate"]["sha256"]
        != EXPECTED_DEEPMIND_CANDIDATE_SHA256
    ):
        raise ValueError("candidate-mix v0.5 DeepMind candidate changed")
    _load_predecessor_plan(plan, repository_root=repository_root)
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    _verify_internal_sha(predecessor_report, "report_sha256")
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if predecessor_report.get("status") != (
        "verified_registry_v0_9_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix v0.5 predecessor report changed")
    if predecessor_verification.get("status") != (
        "verified_registry_v0_9_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix v0.5 predecessor verification changed")
    _validate_deepmind_evidence(resolved)
    return plan


def _validate_deepmind_evidence(resolved: Mapping[str, Path]) -> None:
    promotion = _load_json(resolved["deepmind_scale_promotion"])
    promotion_verification = _load_json(
        resolved["deepmind_scale_promotion_verification"]
    )
    owner_review = _load_json(resolved["deepmind_scale_owner_review"])
    token_metrics = _load_json(resolved["deepmind_scale_token_metrics"])
    validate_deepmind_scale_promotion(promotion)
    validate_owner_review_receipt(owner_review)
    _verify_internal_sha(promotion_verification, "verification_sha256")
    if promotion.get("receipt_sha256") != EXPECTED_DEEPMIND_PROMOTION_INTERNAL_SHA256:
        raise ValueError("DeepMind scale promotion receipt changed")
    if promotion_verification.get("verification_sha256") != (
        EXPECTED_DEEPMIND_PROMOTION_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind scale promotion verification changed")
    if owner_review.get("receipt_sha256") != (
        EXPECTED_DEEPMIND_OWNER_REVIEW_INTERNAL_SHA256
    ):
        raise ValueError("DeepMind scale owner review changed")
    if token_metrics.get("accepted_records") != 236_698:
        raise ValueError("DeepMind scale token metrics record count changed")
    if token_metrics.get("by_split") != {
        "train": {
            "masked_tokens": 7_581_949,
            "records": 224_688,
            "serialized_tokens": 9_803_062,
            "supervised_tokens": 2_221_113,
        },
        "validation": {
            "masked_tokens": 401_757,
            "records": 12_010,
            "serialized_tokens": 519_810,
            "supervised_tokens": 118_053,
        },
    }:
        raise ValueError("DeepMind scale token metrics split totals changed")


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(
        v4._effective_plan(predecessor, repository_root=repository_root)
    )
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    deepmind = [
        source
        for source in effective["candidate_sources"]
        if source["source_id"] == DEEPMIND_SOURCE_ID
    ]
    if len(deepmind) != 1:
        raise ValueError("candidate-mix v0.5 DeepMind source plan changed")
    deepmind[0].update(
        {
            "candidate_binding": "deepmind_scale_candidate",
            "candidate_file_sha256": EXPECTED_DEEPMIND_CANDIDATE_SHA256,
            "expected_by_split": {
                "train": {"records": 224_688, "supervised_tokens": 2_221_113},
                "validation": {"records": 12_010, "supervised_tokens": 118_053},
            },
            "promotion_binding": "deepmind_scale_promotion",
            "token_evidence_binding": "deepmind_scale_token_metrics",
        }
    )
    effective["bindings"]["source_registry"] = dict(
        plan["bindings"]["source_registry"]
    )
    return effective


def _verify_candidate_promotion(
    *, source_id: str, value: Mapping[str, Any]
) -> None:
    if source_id == DEEPMIND_SOURCE_ID:
        validate_deepmind_scale_promotion(value)
        if value.get("receipt_sha256") != EXPECTED_DEEPMIND_PROMOTION_INTERNAL_SHA256:
            raise ValueError("DeepMind scale promotion receipt changed")
        return
    _verify_promotion_receipt(value, source_id=source_id)


def rebuild_candidate_mix_reaudit_v5(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    predecessor_plan = _load_predecessor_plan(plan, repository_root=root)
    predecessor_report = v4.rebuild_candidate_mix_reaudit_v4(
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
    bound_predecessor = _load_json(
        v2.resolve_binding(
            root,
            plan["bindings"]["predecessor_report"],
            name="predecessor_report",
        )
    )
    if predecessor_report != bound_predecessor:
        raise ValueError("candidate-mix v0.4 predecessor no longer reproduces")

    effective = _effective_plan(plan, repository_root=root)
    resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    resolved.update(v4._resolve_explicit_bindings(predecessor_plan, root))
    resolved["source_registry"] = v2.resolve_binding(
        root,
        plan["bindings"]["source_registry"],
        name="source_registry",
    )
    v3._validate_bound_evidence(resolved)
    v4._validate_new_evidence(resolved)
    _validate_deepmind_evidence(resolved)
    policy = load_sft_policy(resolved["data_contract"])
    benchmark_registry = v2.BenchmarkRegistry(policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != effective["bindings"]["benchmark_registry"][
        "sha256"
    ]:
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
        effective["candidate_sources"], key=lambda item: item["source_id"]
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
        if metrics["rows_with_stale_pending_human_review_provenance"] != (
            expected_pending
        ):
            raise ValueError(f"{source_id} owner-review provenance count changed")
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        validation_tokens += metrics["by_split"]["validation"][
            "supervised_tokens"
        ]
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_candidate_promotion(source_id=source_id, value=promotion)

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.5 row ceiling exceeded")
    accepted, generic = filter_sft_records(
        all_rows,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(accepted) != len(all_rows) or generic["rejected"] != 0:
        raise ValueError(
            "registry-v0.10 candidate pool failed generic validation: "
            f"{generic['rejection_counts']}"
        )
    for metrics in source_metrics.values():
        metrics["generic_sft_accepted_records"] = metrics["records"]
        metrics["generic_sft_rejected_records"] = 0

    if len(all_rows) != EXPECTED_RECORDS:
        raise ValueError("registry-v0.10 exact candidate record count changed")
    if train_tokens != EXPECTED_TRAIN_TOKENS:
        raise ValueError("registry-v0.10 train token count changed")
    if validation_tokens != EXPECTED_VALIDATION_TOKENS:
        raise ValueError("registry-v0.10 validation token count changed")
    exact_snapshot = _duplicate_snapshot(duplicate_index)
    near_snapshot = v2._near_duplicate_audit(
        all_rows, policy=effective["near_duplicate_policy"]
    )
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
    blockers = [
        "accepted_train_supervised_tokens_below_10m_floor",
        "task_token_floors_not_met",
        "language_token_floors_not_met",
        "source_concentration_cap_not_met",
        "active_production_data_contract_not_updated",
        "dataset_seal_cpu_canary_and_sft_not_authorized",
    ]
    if exact_snapshot["duplicate_groups"]:
        blockers.append("exact_duplicate_canonical_selection_not_materialized")
    if near_snapshot["multi_record_components"]:
        blockers.append("near_duplicate_canonical_selection_not_materialized")
    if near_snapshot["cross_source_components"]:
        blockers.append("cross_source_near_duplicates_not_resolved")
    if near_snapshot["train_validation_components"]:
        blockers.append("train_validation_near_leakage_not_resolved")
    excluded = v2._verify_exclusions(
        plan=effective, resolved=resolved, registry=registry
    )
    priority_gaps = [
        task for task in EXPECTED_TASKS if task_gaps[task] > 0
    ] + [mode for mode in EXPECTED_LANGUAGE_MODES if language_gaps[mode] > 0]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_train_supervised_tokens": train_tokens,
            "candidate_records_read": len(all_rows),
            "floor_completion_fraction": round(
                train_tokens / targets["total"]["minimum"], 8
            ),
            "gap_to_10m_train_supervised_tokens": (
                targets["total"]["minimum"] - train_tokens
            ),
            "registry_admissible_validation_supervised_tokens": validation_tokens,
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {
                task: task_tokens[task] for task in EXPECTED_TASKS
            },
        },
        "closed_since_v0_4": [
            "deepmind_scale_candidate_owner_sample_accepted_32_of_32",
            "deepmind_scale_candidate_registered_in_immutable_registry_v0_10",
            "deepmind_scale_candidate_replaces_the_1915_record_canary",
            "reasoning_math_science_train_token_floor_met",
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
                "acquire_diversified_rights_clear_non_math_data_to_close_"
                "remaining_10m_task_language_and_source_balance_gaps"
            ),
            "authorization_required": True,
            "priority_gaps": priority_gaps,
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(
            plan_file_sha256, "plan_file_sha256"
        ),
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
        "status": (
            "verified_registry_v0_10_candidate_mix_not_ready_for_seal_or_gpu_sft"
        ),
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v5_report(report)
    return report


def validate_candidate_mix_reaudit_v5_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.5 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.5 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.5 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.5 report must remain raw-text-free")
    pool = report.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("candidate-mix v0.5 pool summary is missing")
    expected_pool = {
        "accepted_train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "candidate_records_read": EXPECTED_RECORDS,
        "gap_to_10m_train_supervised_tokens": 6_728_207,
        "registry_admissible_validation_supervised_tokens": (
            EXPECTED_VALIDATION_TOKENS
        ),
        "source_count": 6,
    }
    for field, value in expected_pool.items():
        if pool.get(field) != value:
            raise ValueError(f"candidate-mix v0.5 pool changed: {field}")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("candidate-mix v0.5 generic validation is missing")
    if generic.get("accepted") != EXPECTED_RECORDS or generic.get("rejected") != 0:
        raise ValueError("candidate-mix v0.5 generic validation changed")
    if generic.get("rejection_counts") != {}:
        raise ValueError("candidate-mix v0.5 generic validation has rejection reasons")
    deepmind = report.get("source_metrics", {}).get(DEEPMIND_SOURCE_ID)
    if not isinstance(deepmind, Mapping):
        raise ValueError("candidate-mix v0.5 DeepMind metrics are missing")
    if deepmind.get("by_split") != {
        "train": {"records": 224_688, "supervised_tokens": 2_221_113},
        "validation": {"records": 12_010, "supervised_tokens": 118_053},
    }:
        raise ValueError("candidate-mix v0.5 DeepMind metrics changed")
    if pool.get("train_by_task", {}).get("reasoning_math_science") != 2_269_713:
        raise ValueError("candidate-mix v0.5 reasoning total changed")
    if pool.get("source_shares", {}).get(DEEPMIND_SOURCE_ID) != 0.67886721:
        raise ValueError("candidate-mix v0.5 DeepMind share changed")
    exact = report.get("exact_duplicate_audit")
    near = report.get("near_duplicate_audit")
    if not isinstance(exact, Mapping) or not isinstance(near, Mapping):
        raise ValueError("candidate-mix v0.5 duplicate audits are missing")
    if near.get("records") != EXPECTED_RECORDS:
        raise ValueError("candidate-mix v0.5 near-audit row count changed")
    if report.get("registry", {}).get("sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.5 registry evidence changed")


def build_candidate_mix_reaudit_v5_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v5_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_247504_rows_pass_schema_registry_benchmark_and_token_validation": True,
            "deepmind_scale_236698_rows_are_registry_admissible": True,
            "exact_and_near_duplicate_audits_completed": True,
            "frozen_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "predecessor_v0_4_reaudit_reproduced": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_v0_10_hash_verified": True,
        },
        "duplicate_summary": {
            "cross_source_exact_groups": report["exact_duplicate_audit"][
                "cross_source_duplicate_groups"
            ],
            "cross_source_near_components": report["near_duplicate_audit"][
                "cross_source_components"
            ],
            "train_validation_exact_groups": report["exact_duplicate_audit"][
                "train_validation_overlap_groups"
            ],
            "train_validation_near_components": report["near_duplicate_audit"][
                "train_validation_components"
            ],
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_10_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_RECORDS",
    "EXPECTED_REGISTRY_SHA256",
    "EXPECTED_TRAIN_TOKENS",
    "EXPECTED_VALIDATION_TOKENS",
    "build_candidate_mix_reaudit_v5_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v5_plan",
    "rebuild_candidate_mix_reaudit_v5",
    "validate_candidate_mix_reaudit_v5_report",
]
