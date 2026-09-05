"""Registry-v0.12-bound full local re-audit of the clean Lumi SFT pool."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v3 as v3
from . import posttraining_candidate_mix_reaudit_v4 as v4
from . import posttraining_candidate_mix_reaudit_v5 as v5
from . import posttraining_candidate_mix_reaudit_v6 as v6
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_hygiene_materialization import (
    validate_hygiene_materialization_receipt,
)
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_hygiene_registry_promotion import validate_promotion_receipt
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.7"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.7"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.7"

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
    "purpose": "registry_v0_12_full_local_candidate_mix_reaudit_only",
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
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.7.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.7.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "aya_clean_candidate",
        "benchmark_registry_v0_3",
        "hygiene_materialization_receipt",
        "hygiene_materialization_verification",
        "implementation",
        "oasst1_clean_candidate",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "registry_promotion",
        "registry_promotion_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)

EXPECTED_REGISTRY_SHA256 = (
    "8f286b096ee2e406fe4ee0a6d69ea37a8e7017e489fda6c765df46ed2d1b86b7"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "043f57d3e64ccb36c68d0d3ca19ea26cdea3580bf6541c060dc0aefed5fdbfc7"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "a3d00e24925d577b6b64329a793992f73fb0e1659d1f0e3007bdd083a2799366"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "58d5cb212566ca9292a611b053b70378863bf00c391bd94a700f7322a2a67057"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "195aa2dd152e541a04ebbcd7b5329d4bf8391970da22164d882448eee3361340"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "7ee9929ab6ec54e2e665645f1e8827607d3dfd69853fb84b0fa71b8b32fc158d"
)
EXPECTED_REGISTRY_PROMOTION_SHA256 = (
    "d46478cecd8fc5400cf8680d73f831b40c0310d424395d9ea34aa38a3d635ae6"
)
EXPECTED_REGISTRY_PROMOTION_INTERNAL_SHA256 = (
    "004811befb42be994f9bcc37eab86ae99fc037c73dd2bb08855ecc4e0b87ad1a"
)
EXPECTED_REGISTRY_PROMOTION_VERIFICATION_SHA256 = (
    "73b660ea79c43b14fafd222114f718a1ab1e00398dfad92796e70d12e5e24f66"
)
EXPECTED_REGISTRY_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "2bcf502b35b1af76ede32cd68a2dfc49e978d13b1fc557fca7b9d69ddd3212f4"
)
EXPECTED_HYGIENE_RECEIPT_SHA256 = (
    "08d9430be9b0fe8ffbf85d11bd0caa4cb81a78c9faa8ee57df45798f38371a78"
)
EXPECTED_HYGIENE_RECEIPT_INTERNAL_SHA256 = (
    "f06ba2b2c572c8e17c8c7c146bc99f6f8da7e8ae1e219f41d86a2a66ac0435ed"
)
EXPECTED_HYGIENE_VERIFICATION_SHA256 = (
    "24ec151e1e96bb159a46bbac32ab686476d8460fd15a3b6171b27fdfc6694a74"
)
EXPECTED_HYGIENE_VERIFICATION_INTERNAL_SHA256 = (
    "4c15a03a2ab50e0fc91086a34dfbe78b89c524bb584b692aa1a127cc27d438b3"
)
EXPECTED_OASST1_CANDIDATE_SHA256 = (
    "1c970f94fe009acc4d379e9c53eee5ecefe29f42db33f01c6d9c09c87e7950c2"
)
EXPECTED_AYA_CANDIDATE_SHA256 = (
    "6e601c9eee1cc3f90647cd4701a8bf24667dd36e25a42f49f7ef4059b9683558"
)
EXPECTED_RECORDS = 258_581
EXPECTED_ACCEPTED_RECORDS = 258_581
EXPECTED_TRAIN_TOKENS = 5_363_689
EXPECTED_VALIDATION_TOKENS = 318_800
EXPECTED_CODING_TOKENS = 2_153_225
EXPECTED_ENGLISH_TOKENS = 4_827_385
EXPECTED_BENGALI_TOKENS = 525_143

CLEAN_SOURCE_UPDATES = {
    "coherelabs_aya_dataset": {
        "candidate_binding": "aya_clean_candidate",
        "candidate_file_sha256": EXPECTED_AYA_CANDIDATE_SHA256,
        "expected_by_split": {
            "train": {"records": 1_313, "supervised_tokens": 513_379},
            "validation": {"records": 145, "supervised_tokens": 55_581},
        },
    },
    "openassistant_oasst1": {
        "candidate_binding": "oasst1_clean_candidate",
        "candidate_file_sha256": EXPECTED_OASST1_CANDIDATE_SHA256,
        "expected_by_split": {
            "train": {"records": 1_649, "supervised_tokens": 452_690},
            "validation": {"records": 93, "supervised_tokens": 23_475},
        },
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.7 binding set changed")
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
    return v6.load_candidate_mix_reaudit_v6_plan(
        path,
        repository_root=repository_root,
    )


def _validate_v7_evidence(resolved: Mapping[str, Path]) -> None:
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    v6.validate_candidate_mix_reaudit_v6_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256")
        != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.7 predecessor evidence changed")

    registry_promotion = _load_json(resolved["registry_promotion"])
    registry_verification = _load_json(resolved["registry_promotion_verification"])
    validate_promotion_receipt(registry_promotion)
    _verify_internal_sha(registry_verification, "verification_sha256")
    if (
        registry_promotion.get("receipt_sha256")
        != EXPECTED_REGISTRY_PROMOTION_INTERNAL_SHA256
        or registry_verification.get("verification_sha256")
        != EXPECTED_REGISTRY_PROMOTION_VERIFICATION_INTERNAL_SHA256
        or registry_verification.get("successor_registry_file_sha256")
        != EXPECTED_REGISTRY_SHA256
    ):
        raise ValueError("candidate-mix v0.7 registry-promotion evidence changed")

    hygiene_receipt = _load_json(resolved["hygiene_materialization_receipt"])
    hygiene_verification = _load_json(
        resolved["hygiene_materialization_verification"]
    )
    validate_hygiene_materialization_receipt(hygiene_receipt)
    _verify_internal_sha(hygiene_verification, "verification_sha256")
    if (
        hygiene_receipt.get("receipt_sha256")
        != EXPECTED_HYGIENE_RECEIPT_INTERNAL_SHA256
        or hygiene_verification.get("verification_sha256")
        != EXPECTED_HYGIENE_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.7 hygiene evidence changed")


def load_candidate_mix_reaudit_v7_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "benchmark_decontamination": {
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
        "clean_source_replacements": CLEAN_SOURCE_UPDATES,
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_registry_v0_12_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.7 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    exact_bindings = {
        "aya_clean_candidate": EXPECTED_AYA_CANDIDATE_SHA256,
        "benchmark_registry_v0_3": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        "hygiene_materialization_receipt": EXPECTED_HYGIENE_RECEIPT_SHA256,
        "hygiene_materialization_verification": EXPECTED_HYGIENE_VERIFICATION_SHA256,
        "oasst1_clean_candidate": EXPECTED_OASST1_CANDIDATE_SHA256,
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "registry_promotion": EXPECTED_REGISTRY_PROMOTION_SHA256,
        "registry_promotion_verification": (
            EXPECTED_REGISTRY_PROMOTION_VERIFICATION_SHA256
        ),
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact_bindings.items():
        if plan["bindings"][name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.7 binding changed: {name}")
    _load_predecessor_plan(plan, repository_root=repository_root)
    _validate_v7_evidence(resolved)
    return plan


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(v6._effective_plan(predecessor, repository_root=repository_root))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(
        plan["bindings"]["source_registry"]
    )
    effective["bindings"]["benchmark_registry"] = dict(
        plan["bindings"]["benchmark_registry_v0_3"]
    )
    seen: set[str] = set()
    for source in effective["candidate_sources"]:
        source_id = source["source_id"]
        if source_id not in CLEAN_SOURCE_UPDATES:
            continue
        source.update(copy.deepcopy(CLEAN_SOURCE_UPDATES[source_id]))
        seen.add(source_id)
    if seen != set(CLEAN_SOURCE_UPDATES):
        raise ValueError("candidate-mix v0.7 did not replace exactly two sources")
    return effective


def rebuild_candidate_mix_reaudit_v7(
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
    resolved.update(_resolve_explicit_bindings(plan, root))
    predecessor_v6 = _load_predecessor_plan(plan, repository_root=root)
    predecessor_v5 = v6._load_predecessor_plan(
        predecessor_v6,
        repository_root=root,
    )
    predecessor_v4 = v5._load_predecessor_plan(
        predecessor_v5,
        repository_root=root,
    )
    resolved.update(v4._resolve_explicit_bindings(predecessor_v4, root))
    resolved.update(_resolve_explicit_bindings(plan, root))
    resolved["source_registry"] = v2.resolve_binding(
        root, plan["bindings"]["source_registry"], name="source_registry"
    )
    v3._validate_bound_evidence(resolved)
    v4._validate_new_evidence(resolved)
    v5._validate_deepmind_evidence(resolved)
    v6._validate_opencode_evidence(resolved)
    _validate_v7_evidence(resolved)

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(
        audit_policy.benchmark_decontamination,
        root,
    )
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.7 benchmark registry changed")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
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
        if metrics["rows_with_stale_pending_human_review_provenance"] != (
            source_plan["expected_pending_owner_review_rows"]
        ):
            raise ValueError(f"{source_id} owner-review provenance count changed")
        metrics["generic_sft_accepted_records"] = metrics["records"]
        metrics["generic_sft_rejected_records"] = 0
        metrics["quarantined_by_split"] = {
            "train": {"records": 0, "supervised_tokens": 0},
            "validation": {"records": 0, "supervised_tokens": 0},
        }
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        validation_tokens += metrics["by_split"]["validation"]["supervised_tokens"]
        task_tokens.update(metrics["train_by_task"])
        language_tokens.update(metrics["train_by_language_mode"])
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        v6._verify_candidate_promotion(source_id=source_id, value=promotion)

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.7 row ceiling exceeded")
    if (
        len(all_rows) != EXPECTED_RECORDS
        or train_tokens != EXPECTED_TRAIN_TOKENS
        or validation_tokens != EXPECTED_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.12 candidate totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    if (
        len(accepted) != EXPECTED_ACCEPTED_RECORDS
        or generic["rejected"] != 0
        or generic["rejection_counts"] != {}
    ):
        raise ValueError(
            "registry-v0.12 candidate rejection set changed: "
            f"{generic['rejection_counts']}"
        )

    accepted_duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        accepted_duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(accepted_duplicate_index)
    near_snapshot = v2._near_duplicate_audit(
        accepted,
        policy=effective["near_duplicate_policy"],
    )
    if exact_snapshot["duplicate_groups"] != 0:
        raise ValueError("registry-v0.12 exact duplicates remain")
    if near_snapshot["multi_record_components"] != 0:
        raise ValueError("registry-v0.12 near duplicates remain")

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
    blockers: list[str] = []
    if train_tokens < targets["total"]["minimum"]:
        blockers.append("accepted_train_supervised_tokens_below_10m_floor")
    if any(task_gaps.values()):
        blockers.append("task_token_floors_not_met")
    if any(language_gaps.values()):
        blockers.append("language_token_floors_not_met")
    if source_cap_violations:
        blockers.append("source_concentration_cap_not_met")
    blockers.extend(
        [
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ]
    )
    excluded = v2._verify_exclusions(plan=effective, resolved=resolved, registry=registry)
    priority_gaps = [
        task for task in EXPECTED_TASKS if task_gaps[task] > 0
    ] + [mode for mode in EXPECTED_LANGUAGE_MODES if language_gaps[mode] > 0]

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "benchmark_decontamination": {
            "benchmarks": sorted(benchmark_registry.benchmarks),
            "raw_benchmark_text_retained": False,
            "registry_entries": 50_944,
            "registry_sha256": benchmark_registry.sha256,
            "unique_exact_signatures": len(benchmark_registry.exact),
        },
        "candidate_pool": {
            "accepted_records_after_benchmark_quarantine": len(accepted),
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
        "closed_since_v0_6": [
            "oasst1_three_bigcodebench_matches_removed_from_candidate",
            "aya_twenty_near_duplicate_losers_removed_from_candidate",
            "clean_candidates_bound_in_immutable_registry_v0_12",
            "full_mix_benchmark_reaudit_has_zero_rejections",
            "full_mix_exact_and_near_duplicate_reaudit_has_zero_conflicts",
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
            "benchmark_contamination_rejections": 0,
            "raw_text_retained": generic["raw_text_retained"],
            "rejected": generic["rejected"],
            "rejection_counts": generic["rejection_counts"],
            "report_sha256": generic["report_sha256"],
            "source_registry_rejections": 0,
            "total": generic["total"],
        },
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "close_remaining_data_scale_task_language_and_source_balance_"
                "gaps_before_dataset_seal"
            ),
            "authorization_required": True,
            "priority_gaps": priority_gaps,
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "quarantine": {
            "by_split": {
                "train": {"records": 0, "supervised_tokens": 0},
                "validation": {"records": 0, "supervised_tokens": 0},
            },
            "materialized_in_candidate_files": True,
            "reason_counts": {},
            "records": [],
            "total_records": 0,
        },
        "raw_training_text_retained": False,
        "registry": {
            "path": plan["bindings"]["source_registry"]["path"],
            "sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(registry.by_id),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "verified_registry_v0_12_candidate_mix_not_ready_for_seal_or_gpu_sft",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v7_report(report)
    return report


def validate_candidate_mix_reaudit_v7_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.7 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.7 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.7 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.7 report must remain raw-text-free")
    pool = report.get("candidate_pool", {})
    expected_pool = {
        "accepted_records_after_benchmark_quarantine": EXPECTED_ACCEPTED_RECORDS,
        "accepted_train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "candidate_records_read": EXPECTED_RECORDS,
        "gap_to_10m_train_supervised_tokens": 10_000_000 - EXPECTED_TRAIN_TOKENS,
        "registry_admissible_validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        "source_count": 7,
    }
    for field, expected_value in expected_pool.items():
        if pool.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.7 pool changed: {field}")
    if pool.get("train_by_task", {}).get("coding") != EXPECTED_CODING_TOKENS:
        raise ValueError("candidate-mix v0.7 coding total changed")
    if pool.get("train_by_language_mode", {}).get("english") != (
        EXPECTED_ENGLISH_TOKENS
    ):
        raise ValueError("candidate-mix v0.7 English total changed")
    if pool.get("train_by_language_mode", {}).get("bengali_script") != (
        EXPECTED_BENGALI_TOKENS
    ):
        raise ValueError("candidate-mix v0.7 Bengali total changed")
    generic = report.get("generic_sft_validation", {})
    if (
        generic.get("accepted") != EXPECTED_ACCEPTED_RECORDS
        or generic.get("rejected") != 0
        or generic.get("rejection_counts") != {}
    ):
        raise ValueError("candidate-mix v0.7 generic validation changed")
    quarantine = report.get("quarantine", {})
    if (
        quarantine.get("total_records") != 0
        or quarantine.get("reason_counts") != {}
        or quarantine.get("materialized_in_candidate_files") is not True
    ):
        raise ValueError("candidate-mix v0.7 quarantine changed")
    exact = report.get("exact_duplicate_audit", {})
    near = report.get("near_duplicate_audit", {})
    if exact.get("duplicate_groups") != 0:
        raise ValueError("candidate-mix v0.7 exact duplicate summary changed")
    if (
        near.get("records") != EXPECTED_ACCEPTED_RECORDS
        or near.get("multi_record_components") != 0
        or near.get("cross_source_components") != 0
        or near.get("train_validation_components") != 0
    ):
        raise ValueError("candidate-mix v0.7 near duplicate summary changed")
    if report.get("registry", {}).get("sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.7 registry evidence changed")
    benchmark = report.get("benchmark_decontamination", {})
    if (
        benchmark.get("registry_sha256") != EXPECTED_BENCHMARK_REGISTRY_SHA256
        or benchmark.get("registry_entries") != 50_944
    ):
        raise ValueError("candidate-mix v0.7 benchmark evidence changed")
    source_metrics = report.get("source_metrics", {})
    for source_id, update in CLEAN_SOURCE_UPDATES.items():
        metrics = source_metrics.get(source_id, {})
        if (
            metrics.get("candidate_file_sha256")
            != update["candidate_file_sha256"]
            or metrics.get("by_split") != update["expected_by_split"]
            or metrics.get("generic_sft_rejected_records") != 0
        ):
            raise ValueError(f"candidate-mix v0.7 clean source changed: {source_id}")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.7 report contains raw training text")


def build_candidate_mix_reaudit_v7_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v7_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_258581_registry_v0_12_rows_audited_with_zero_rejections": True,
            "benchmark_registry_v0_3_bigcodebench_and_livecodebench_verified": True,
            "exact_and_near_duplicate_audits_have_zero_conflicts": True,
            "frozen_4k_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "oasst1_and_aya_clean_candidates_verified": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "registry_v0_12_hash_and_promotion_evidence_verified": True,
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
        "status": "verified_registry_v0_12_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_BENCHMARK_REGISTRY_SHA256",
    "EXPECTED_RECORDS",
    "EXPECTED_REGISTRY_SHA256",
    "EXPECTED_TRAIN_TOKENS",
    "EXPECTED_VALIDATION_TOKENS",
    "build_candidate_mix_reaudit_v7_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v7_plan",
    "rebuild_candidate_mix_reaudit_v7",
    "validate_candidate_mix_reaudit_v7_report",
]
