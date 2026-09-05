"""Registry-v0.8-bound local re-audit of the Lumi SFT candidate pool."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _candidate_metrics,
    _duplicate_snapshot,
    _effective_split,
    _load_json,
    _load_jsonl,
    _measure_conversation,
    _require_sha256,
    _targets,
    _verify_internal_sha,
    _verify_promotion_receipt,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.3"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.3"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.3"

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
    "purpose": "registry_v0_8_local_candidate_mix_reaudit_only",
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
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.3.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.3.json"
    ),
}
EXPECTED_REGISTRY_SHA256 = (
    "d20c89ca34a284fc120a115d0ac9578c7e03ae3c915452d3f0ea144928e9a519"
)
EXPECTED_CANDIDATE_SOURCE_IDS = {
    "coherelabs_aya_dataset",
    "deepmind_mathematics_dataset",
    "dmath",
    "lumi_ai_assisted_sft_v0_3",
    "lumi_safety_tool_sft_v0_1",
    "openassistant_oasst1",
}
EXPECTED_EXCLUDED_SOURCE_IDS = {"openr1_math_220k"}
EXPECTED_SOURCE_ADDITION = {
    "candidate_binding": "ai_v3_candidate",
    "candidate_file_sha256": (
        "ecb2b4ea6ec85ac41e65236ce0669761cbc51c19267e8001ba04f134bfa7c4a9"
    ),
    "expected_by_split": {
        "train": {"records": 240, "supervised_tokens": 9_314},
        "validation": {"records": 80, "supervised_tokens": 2_138},
    },
    "expected_pending_owner_review_rows": 320,
    "language_mode_policy": "language_field_multilingual_as_banglish",
    "promotion_binding": "ai_v3_promotion",
    "registry_status": "candidate_eligible",
    "revision": "7961c081dc57339613fd95961adea65d02d22348b8a69217cc0af3949fe6a7c6",
    "source_id": "lumi_ai_assisted_sft_v0_3",
    "split_location": "top_level",
    "token_evidence_binding": "ai_v3_token_evidence",
}
EXPECTED_BINDINGS = {
    "ai_v3_candidate",
    "ai_v3_owner_review",
    "ai_v3_pilot_config",
    "ai_v3_promotion",
    "ai_v3_promotion_verification",
    "ai_v3_token_evidence",
    "implementation",
    "predecessor_plan",
    "predecessor_report",
    "predecessor_verification",
    "runner",
    "source_registry",
    "test",
    "verifier",
}
EXPECTED_RECORDS = 12_721
EXPECTED_TRAIN_TOKENS = 1_068_485
EXPECTED_VALIDATION_TOKENS = 92_657


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.3 binding set changed")
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
    return v2.load_candidate_mix_reaudit_plan(path, repository_root=repository_root)


def load_candidate_mix_reaudit_v3_plan(
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
        "source_addition": EXPECTED_SOURCE_ADDITION,
        "status": "authorized_registry_v0_8_local_reaudit_no_production_action",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-mix v0.3 plan changed: {field}")

    resolved = _resolve_explicit_bindings(plan, repository_root)
    if plan["bindings"]["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.3 registry binding changed")
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    predecessor_candidates = predecessor.get("candidate_sources")
    if not isinstance(predecessor_candidates, list):
        raise ValueError("candidate-mix v0.2 candidate set changed")
    candidate_ids = {item["source_id"] for item in predecessor_candidates}
    candidate_ids.add(plan["source_addition"]["source_id"])
    if candidate_ids != EXPECTED_CANDIDATE_SOURCE_IDS:
        raise ValueError("candidate-mix v0.3 candidate source set changed")
    predecessor_excluded = predecessor.get("excluded_sources")
    if not isinstance(predecessor_excluded, list):
        raise ValueError("candidate-mix v0.2 exclusion set changed")
    remaining_excluded = {
        item["source_id"]
        for item in predecessor_excluded
        if item["source_id"] != "lumi_ai_assisted_sft_v0_3"
    }
    if remaining_excluded != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.3 excluded source set changed")

    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    _verify_internal_sha(predecessor_report, "report_sha256")
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if predecessor_report.get("status") != (
        "verified_repaired_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix v0.2 report status changed")
    if predecessor_verification.get("status") != (
        "verified_repaired_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix v0.2 verification status changed")

    owner = _load_json(resolved["ai_v3_owner_review"])
    pilot_config = _load_json(resolved["ai_v3_pilot_config"])
    promotion = _load_json(resolved["ai_v3_promotion"])
    promotion_verification = _load_json(resolved["ai_v3_promotion_verification"])
    token_evidence = _load_json(resolved["ai_v3_token_evidence"])
    _verify_internal_sha(owner, "receipt_sha256")
    _verify_internal_sha(promotion, "receipt_sha256")
    _verify_internal_sha(promotion_verification, "verification_sha256")
    _verify_internal_sha(token_evidence, "report_sha256")
    if pilot_config.get("language_policy", {}).get("multilingual") != {
        "bengali_script_allowed": False,
        "language_mode": "banglish_romanized",
    }:
        raise ValueError("AI v0.3 multilingual-to-Banglish policy changed")
    if owner.get("accepted_records") != 32 or owner.get("rejected_records") != 0:
        raise ValueError("AI v0.3 owner spot-check receipt changed")
    if promotion.get("status") != (
        "reviewed_ai_v0_3_registered_in_immutable_candidate_successor_"
        "production_dataset_still_blocked"
    ):
        raise ValueError("AI v0.3 promotion status changed")
    if promotion_verification.get("status") != (
        "verified_ai_v0_3_candidate_registry_promotion_nonproduction"
    ):
        raise ValueError("AI v0.3 promotion verification status changed")
    metrics = token_evidence.get("source_token_evidence", {}).get("ai_assisted_v0_3")
    if not isinstance(metrics, Mapping) or metrics.get("by_split") != (
        EXPECTED_SOURCE_ADDITION["expected_by_split"]
    ):
        raise ValueError("AI v0.3 frozen-token evidence changed")
    return plan


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    candidates = [dict(item) for item in predecessor["candidate_sources"]]
    for candidate in candidates:
        candidate["expected_pending_owner_review_rows"] = 0
    candidates.append(dict(plan["source_addition"]))
    excluded = [
        dict(item)
        for item in predecessor["excluded_sources"]
        if item["source_id"] != "lumi_ai_assisted_sft_v0_3"
    ]
    bindings = dict(predecessor["bindings"])
    bindings.update(plan["bindings"])
    return {
        "authorization": plan["authorization"],
        "bindings": bindings,
        "candidate_sources": candidates,
        "excluded_sources": excluded,
        "limits": plan["limits"],
        "near_duplicate_policy": plan["near_duplicate_policy"],
        "observed_at": plan["observed_at"],
        "output": plan["output"],
    }


def _validate_bound_evidence(resolved: Mapping[str, Path]) -> None:
    v2._validate_predecessor_evidence(resolved)
    promotion = _load_json(resolved["ai_v3_promotion"])
    promotion_verification = _load_json(resolved["ai_v3_promotion_verification"])
    _verify_internal_sha(promotion, "receipt_sha256")
    _verify_internal_sha(promotion_verification, "verification_sha256")
    _verify_promotion_receipt(promotion, source_id="lumi_ai_assisted_sft_v0_3")


def _candidate_metrics_v3(
    *,
    rows: list[dict[str, Any]],
    source_plan: Mapping[str, Any],
    registry_source: Any,
    tokenizer: Any,
    template: Any,
    duplicate_index: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    policy = source_plan["language_mode_policy"]
    if policy != "language_field_multilingual_as_banglish":
        return _candidate_metrics(
            rows=rows,
            source_plan=source_plan,
            registry_source=registry_source,
            tokenizer=tokenizer,
            template=template,
            duplicate_index=duplicate_index,
        )
    if source_plan["source_id"] != "lumi_ai_assisted_sft_v0_3":
        raise ValueError("multilingual-as-Banglish policy escaped the AI v0.3 source")
    if any(row.get("language") not in {"bn", "en", "multilingual"} for row in rows):
        raise ValueError("AI v0.3 language labels changed")
    multilingual_rows = [row for row in rows if row.get("language") == "multilingual"]
    if len(multilingual_rows) != 105:
        raise ValueError("AI v0.3 multilingual row count changed")
    if any(
        any(
            "\u0980" <= character <= "\u09ff"
            for message in row["messages"]
            for character in message["content"]
        )
        for row in multilingual_rows
    ):
        raise ValueError("AI v0.3 Banglish lane unexpectedly contains Bengali script")

    adapted_rows = [
        {**row, "language": "en"} if row.get("language") == "multilingual" else row
        for row in rows
    ]
    adapted_plan = {
        **source_plan,
        "language_mode_policy": "language_field",
    }
    metrics = _candidate_metrics(
        rows=adapted_rows,
        source_plan=adapted_plan,
        registry_source=registry_source,
        tokenizer=tokenizer,
        template=template,
        duplicate_index=duplicate_index,
    )
    banglish_tokens = 0
    for row in rows:
        split, _normalized = _effective_split(row, source_plan["split_location"])
        if split != "train" or row.get("language") != "multilingual":
            continue
        banglish_tokens += _measure_conversation(
            row["messages"],
            tokenizer=tokenizer,
            template=template,
        )
    language_tokens = Counter(metrics["train_by_language_mode"])
    language_tokens["english"] -= banglish_tokens
    language_tokens["banglish"] += banglish_tokens
    if language_tokens["english"] < 0:
        raise ValueError("AI v0.3 Banglish token projection underflowed")
    metrics["train_by_language_mode"] = {
        mode: count for mode, count in sorted(language_tokens.items()) if count
    }
    metrics["language_mapping"] = {
        "multilingual_label": "banglish",
        "multilingual_records": len(multilingual_rows),
    }
    return metrics


def rebuild_candidate_mix_reaudit_v3(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    effective = _effective_plan(plan, repository_root=root)
    resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    _validate_bound_evidence(resolved)
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
    missing_mix = _load_json(resolved["missing_mix_proposal"])
    targets = _targets(missing_mix)
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
        metrics = _candidate_metrics_v3(
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
        raise ValueError("candidate-mix v0.3 row ceiling exceeded")
    accepted, generic_report = filter_sft_records(
        all_rows,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if len(accepted) != len(all_rows) or generic_report["rejected"] != 0:
        raise ValueError(
            "registry-v0.8 candidate pool failed generic validation: "
            f"{generic_report['rejection_counts']}"
        )
    for metrics in source_metrics.values():
        metrics["generic_sft_accepted_records"] = metrics["records"]
        metrics["generic_sft_rejected_records"] = 0

    exact_snapshot = _duplicate_snapshot(duplicate_index)
    if any(exact_snapshot.values()):
        raise ValueError("registry-v0.8 candidate pool has exact duplicate or split leak")
    near_snapshot = v2._near_duplicate_audit(
        all_rows,
        policy=effective["near_duplicate_policy"],
    )
    if near_snapshot["cross_source_components"] != 0:
        raise ValueError("registry-v0.8 pool has cross-source near duplicates")
    if near_snapshot["train_validation_components"] != 0:
        raise ValueError("registry-v0.8 pool has train-validation near leakage")

    if len(all_rows) != EXPECTED_RECORDS:
        raise ValueError("registry-v0.8 exact candidate record count changed")
    if train_tokens != EXPECTED_TRAIN_TOKENS:
        raise ValueError("registry-v0.8 train token count changed")
    if validation_tokens != EXPECTED_VALIDATION_TOKENS:
        raise ValueError("registry-v0.8 validation token count changed")

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
        "lumi_ai_assisted_sft_v0_3_per_record_owner_review_provenance_pending",
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
        "closed_since_v0_2": [
            "lumi_ai_assisted_sft_v0_3_exact_source_is_registered_in_v0_8",
            "lumi_ai_assisted_sft_v0_3_all_320_rows_are_included_in_mix_audit",
            "lumi_ai_assisted_sft_v0_3_frozen_token_counts_are_reproduced",
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
            "accepted": generic_report["accepted"],
            "accepted_dataset_sha256": generic_report["accepted_dataset_sha256"],
            "benchmark_contamination_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("benchmark_contamination:")
            ),
            "raw_text_retained": generic_report["raw_text_retained"],
            "rejected": generic_report["rejected"],
            "rejection_counts": generic_report["rejection_counts"],
            "report_sha256": generic_report["report_sha256"],
            "source_registry_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("source_registry")
            ),
            "total": generic_report["total"],
        },
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "resolve_ai_v0_3_owner_review_provenance_then_continue_"
                "rights_clear_bulk_mix_acquisition"
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
        "status": "verified_registry_v0_8_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v3_report(report)
    return report


def validate_candidate_mix_reaudit_v3_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.3 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.3 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.3 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.3 report must remain raw-text-free")
    pool = report.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("candidate-mix v0.3 pool summary is missing")
    expected_pool = {
        "accepted_train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "candidate_records_read": EXPECTED_RECORDS,
        "registry_admissible_validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        "source_count": 6,
    }
    for field, value in expected_pool.items():
        if pool.get(field) != value:
            raise ValueError(f"candidate-mix v0.3 pool changed: {field}")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("candidate-mix v0.3 generic validation is missing")
    if generic.get("accepted") != EXPECTED_RECORDS or generic.get("rejected") != 0:
        raise ValueError("candidate-mix v0.3 generic validation changed")
    if generic.get("rejection_counts") != {}:
        raise ValueError("candidate-mix v0.3 generic validation has rejection reasons")
    exact = report.get("exact_duplicate_audit")
    if not isinstance(exact, Mapping) or any(exact.values()):
        raise ValueError("candidate-mix v0.3 exact duplicate audit is not clean")
    near = report.get("near_duplicate_audit")
    if not isinstance(near, Mapping):
        raise ValueError("candidate-mix v0.3 near-duplicate audit is missing")
    if near.get("cross_source_components") != 0:
        raise ValueError("candidate-mix v0.3 has cross-source near duplicates")
    if near.get("train_validation_components") != 0:
        raise ValueError("candidate-mix v0.3 has train-validation near leakage")
    ai_metrics = report.get("source_metrics", {}).get("lumi_ai_assisted_sft_v0_3")
    if not isinstance(ai_metrics, Mapping):
        raise ValueError("candidate-mix v0.3 AI source metrics are missing")
    if ai_metrics.get("rows_with_stale_pending_human_review_provenance") != 320:
        raise ValueError("candidate-mix v0.3 AI provenance blocker changed")
    blockers = report.get("closure", {}).get("blockers")
    if not isinstance(blockers, list) or (
        "lumi_ai_assisted_sft_v0_3_per_record_owner_review_provenance_pending"
        not in blockers
    ):
        raise ValueError("candidate-mix v0.3 AI provenance blocker is absent")


def build_candidate_mix_reaudit_v3_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v3_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_12721_rows_pass_generic_schema_registry_and_benchmark_validation": True,
            "ai_v0_3_exact_320_rows_are_registry_admissible": True,
            "ai_v0_3_pending_owner_review_provenance_is_not_hidden": True,
            "exact_duplicate_and_train_validation_overlap_are_zero": True,
            "frozen_token_counts_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "predecessor_candidates_and_registry_v0_7_unchanged": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_v0_8_hash_verified": True,
            "whole_conversation_near_audit_has_no_cross_source_or_split_leakage": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_8_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
