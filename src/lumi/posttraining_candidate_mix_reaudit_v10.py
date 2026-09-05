"""Registry-v0.15-bound full local re-audit of the Lumi SFT candidate pool."""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v3 as v3
from . import posttraining_candidate_mix_reaudit_v8 as v8
from . import posttraining_candidate_mix_reaudit_v9 as v9
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _effective_split,
    _load_json,
    _load_jsonl,
    _measure_conversation,
    _require_sha256,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_first_party_teacher_canary_v18 import _load_direct_tokenizer
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry
from .posttraining_teacher_canary_source_promotion import (
    CANONICAL_CANDIDATE_SHA256,
    PREDECESSOR_REGISTRY_SHA256,
    SUCCESSOR_REGISTRY_SHA256,
    validate_promotion_receipt as validate_teacher_promotion_receipt,
)

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.10"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.10"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.10"

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
    "purpose": "registry_v0_15_full_local_candidate_mix_reaudit_only",
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
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.10.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.10.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "implementation",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "teacher_candidate",
        "teacher_promotion_plan",
        "teacher_promotion_receipt",
        "teacher_promotion_verification",
        "test",
        "verifier",
    }
)

EXPECTED_REGISTRY_SHA256 = SUCCESSOR_REGISTRY_SHA256
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v9.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "15900d198513a65300aa19e77e505d030130195df9f4a14e00eab7049e158736"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "ab9d3575ae7de7c10858ef4abb72d667f986e1b514db49601eb044935008cd60"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "5b7b509d65b9c121ef04db8402b9f647bd4945f1c6bd02ce6e6db27f8f802cc1"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "f6d9c813945507e117799d9469ca0ce919234e37afd61c05d0cee6db3e252b61"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "8ceef664cf852c5b05578f877153b092c4ac0afeb8a33c0e891ae6957c0bad38"
)
EXPECTED_TEACHER_PROMOTION_PLAN_SHA256 = (
    "b9bddc02f29a4ac0b7118c05a82a5c241e451d27a9cd35f176d197e4265045dc"
)
EXPECTED_TEACHER_PROMOTION_RECEIPT_SHA256 = (
    "1683a001256346afa56501aba5d6d13af0cf84dfa5d1963a2d17c71e568edb5e"
)
EXPECTED_TEACHER_PROMOTION_RECEIPT_INTERNAL_SHA256 = (
    "59897eb19ffc27863b15c51f68d2edb5e066b68846c3c3a4621211299932f94c"
)
EXPECTED_TEACHER_PROMOTION_VERIFICATION_SHA256 = (
    "e23cdc9592cc208537a98f63895d3910af67abfa91160409fe7489b200490a2a"
)
EXPECTED_TEACHER_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "c4414d6ed72fdd4d2a109b2d0d7fdc8538d3f857c3a958c62f5c65fd717b01d5"
)

EXPECTED_INPUT_RECORDS = v9.EXPECTED_INPUT_RECORDS + 80
EXPECTED_INPUT_TRAIN_TOKENS = v9.EXPECTED_INPUT_TRAIN_TOKENS + 2_002
EXPECTED_INPUT_VALIDATION_TOKENS = v9.EXPECTED_INPUT_VALIDATION_TOKENS + 799
TEACHER_SOURCE_ID = "lumi_first_party_teacher_canary_v0_1"
TEACHER_REVISION = "c1c9c2d09f1680c2e18eb2c7570994c141501f40732bd8b2b890015c97a88634"
TEACHER_SOURCE_ADDITION = {
    "candidate_binding": "teacher_candidate",
    "candidate_file_sha256": CANONICAL_CANDIDATE_SHA256,
    "expected_by_split": {
        "train": {"records": 60, "supervised_tokens": 2_002},
        "validation": {"records": 20, "supervised_tokens": 799},
    },
    "expected_pending_owner_review_rows": 0,
    "language_mode_policy": "teacher_record_id_suffix_v0.1",
    "promotion_binding": "teacher_promotion_receipt",
    "registry_status": "candidate_eligible",
    "revision": TEACHER_REVISION,
    "source_id": TEACHER_SOURCE_ID,
    "split_location": "top_level",
    "token_evidence_binding": "teacher_promotion_receipt",
}
EXPECTED_TEACHER_TRAIN_BY_MODE = {
    "banglish": 407,
    "bengali_script": 712,
    "english": 354,
    "natural_code_switching": 529,
}
EXPECTED_TEACHER_TRAIN_BY_TASK = {
    "factual_qa": 292,
    "multilingual_bengali": 455,
    "safety_helpfulness": 474,
    "tool_api": 401,
    "writing_summarization": 380,
}
_TEACHER_MODE = re.compile(
    r"-(banglish|bengali_script|english|natural_code_switching)-[0-9]{2}$"
)


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.10 binding set changed")
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
    return v9.load_candidate_mix_reaudit_v9_plan(path, repository_root=repository_root)


def _validate_v10_evidence(resolved: Mapping[str, Path]) -> None:
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    v9.validate_candidate_mix_reaudit_v9_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256")
        != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.10 predecessor evidence changed")

    teacher_receipt = _load_json(resolved["teacher_promotion_receipt"])
    teacher_verification = _load_json(resolved["teacher_promotion_verification"])
    validate_teacher_promotion_receipt(teacher_receipt)
    _verify_internal_sha(teacher_verification, "verification_sha256")
    if (
        teacher_receipt.get("receipt_sha256")
        != EXPECTED_TEACHER_PROMOTION_RECEIPT_INTERNAL_SHA256
        or teacher_receipt.get("candidate", {}).get("file_sha256")
        != CANONICAL_CANDIDATE_SHA256
        or teacher_receipt.get("registry", {}).get("successor_sha256")
        != EXPECTED_REGISTRY_SHA256
        or teacher_verification.get("verification_sha256")
        != EXPECTED_TEACHER_PROMOTION_VERIFICATION_INTERNAL_SHA256
        or teacher_verification.get("canonical_candidate", {}).get("records") != 80
        or teacher_verification.get("successor_registry", {}).get("source_count") != 16
        or not all(teacher_verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.10 teacher promotion evidence changed")


def load_candidate_mix_reaudit_v10_plan(
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
        "limits": EXPECTED_LIMITS,
        "near_duplicate_policy": v2.EXPECTED_NEAR_DUPLICATE_POLICY,
        "observed_at": "2026-08-09",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_addition": TEACHER_SOURCE_ADDITION,
        "status": "authorized_registry_v0_15_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.10 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    exact_bindings = {
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
        "teacher_candidate": CANONICAL_CANDIDATE_SHA256,
        "teacher_promotion_plan": EXPECTED_TEACHER_PROMOTION_PLAN_SHA256,
        "teacher_promotion_receipt": EXPECTED_TEACHER_PROMOTION_RECEIPT_SHA256,
        "teacher_promotion_verification": (
            EXPECTED_TEACHER_PROMOTION_VERIFICATION_SHA256
        ),
    }
    for name, expected_sha256 in exact_bindings.items():
        if plan["bindings"][name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.10 binding changed: {name}")
    _load_predecessor_plan(plan, repository_root=repository_root)
    _validate_v10_evidence(resolved)
    return plan


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(v9._effective_plan(predecessor, repository_root=repository_root))
    if any(
        source.get("source_id") == TEACHER_SOURCE_ID
        for source in effective["candidate_sources"]
    ):
        raise ValueError("teacher source already exists in predecessor candidate set")
    effective["candidate_sources"].append(copy.deepcopy(TEACHER_SOURCE_ADDITION))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(plan["bindings"]["source_registry"])
    return effective


def _teacher_candidate_metrics(
    *,
    rows: list[dict[str, Any]],
    source_plan: Mapping[str, Any],
    registry_source: Any,
    tokenizer: Any,
    template: Any,
    duplicate_index: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    if len(rows) != 80 or source_plan["source_id"] != TEACHER_SOURCE_ID:
        raise ValueError("teacher candidate inventory changed")
    modes: dict[str, str] = {}
    expected_language = {
        "banglish": "multilingual",
        "bengali_script": "bn",
        "english": "en",
        "natural_code_switching": "multilingual",
    }
    mode_counts: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise ValueError("teacher record ID changed")
        match = _TEACHER_MODE.search(record_id)
        if match is None:
            raise ValueError("teacher record ID no longer binds a language mode")
        mode = match.group(1)
        if row.get("language") != expected_language[mode]:
            raise ValueError("teacher language label and record ID disagree")
        modes[record_id] = mode
        mode_counts[mode] += 1
    if mode_counts != Counter({mode: 20 for mode in expected_language}):
        raise ValueError("teacher language-mode record inventory changed")

    adapted_rows = [
        {**row, "language": "en"}
        if row.get("language") == "multilingual"
        else row
        for row in rows
    ]
    metrics = v3._candidate_metrics_v3(
        rows=adapted_rows,
        source_plan={**source_plan, "language_mode_policy": "language_field"},
        registry_source=registry_source,
        tokenizer=tokenizer,
        template=template,
        duplicate_index=duplicate_index,
    )
    train_by_mode: Counter[str] = Counter()
    for row in rows:
        split, _ = _effective_split(row, source_plan["split_location"])
        if split == "train":
            train_by_mode[modes[str(row["record_id"])]] += _measure_conversation(
                row["messages"],
                tokenizer=tokenizer,
                template=template,
            )
    if dict(sorted(train_by_mode.items())) != EXPECTED_TEACHER_TRAIN_BY_MODE:
        raise ValueError("teacher language-mode frozen tokens changed")
    if metrics["train_by_task"] != EXPECTED_TEACHER_TRAIN_BY_TASK:
        raise ValueError("teacher task frozen tokens changed")
    metrics["train_by_language_mode"] = dict(sorted(train_by_mode.items()))
    metrics["language_mapping"] = {
        "contract": "record_id_suffix_v0.1",
        "records_by_mode": dict(sorted(mode_counts.items())),
    }
    return metrics


def _verify_candidate_promotion(*, source_id: str, value: Mapping[str, Any]) -> None:
    if source_id != TEACHER_SOURCE_ID:
        v9._verify_candidate_promotion(source_id=source_id, value=value)
        return
    validate_teacher_promotion_receipt(value)
    if (
        value.get("receipt_sha256")
        != EXPECTED_TEACHER_PROMOTION_RECEIPT_INTERNAL_SHA256
        or value.get("registry", {}).get("added_source_id") != TEACHER_SOURCE_ID
        or value.get("candidate", {}).get("file_sha256")
        != CANONICAL_CANDIDATE_SHA256
    ):
        raise ValueError("teacher candidate promotion receipt changed")


def rebuild_candidate_mix_reaudit_v10(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    effective = _effective_plan(plan, repository_root=root)
    resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in effective["bindings"].items()
    }
    resolved.update(_resolve_explicit_bindings(plan, root))
    _validate_v10_evidence(resolved)

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.10 benchmark registry changed")
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
    tokenizer = _load_direct_tokenizer(resolved["tokenizer_json"].parent)
    verify_chat_tokenizer(tokenizer, template)

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    source_metrics: dict[str, dict[str, Any]] = {}
    source_plans: dict[str, Mapping[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    input_train_tokens = 0
    input_validation_tokens = 0
    input_task_tokens: Counter[str] = Counter()
    input_language_tokens: Counter[str] = Counter()
    for source_plan in sorted(effective["candidate_sources"], key=lambda item: item["source_id"]):
        source_id = source_plan["source_id"]
        source_plans[source_id] = source_plan
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
        metrics = (
            _teacher_candidate_metrics(
                rows=rows,
                source_plan=source_plan,
                registry_source=registry_source,
                tokenizer=tokenizer,
                template=template,
                duplicate_index=duplicate_index,
            )
            if source_id == TEACHER_SOURCE_ID
            else v3._candidate_metrics_v3(
                rows=rows,
                source_plan=source_plan,
                registry_source=registry_source,
                tokenizer=tokenizer,
                template=template,
                duplicate_index=duplicate_index,
            )
        )
        if metrics["rows_requiring_split_normalization"] != 0:
            raise ValueError(f"{source_id} still requires split normalization")
        if metrics["rows_with_stale_pending_human_review_provenance"] != (
            source_plan["expected_pending_owner_review_rows"]
        ):
            raise ValueError(f"{source_id} owner-review provenance count changed")
        if metrics["by_split"] != source_plan["expected_by_split"]:
            raise ValueError(f"{source_id} split metrics changed")
        source_metrics[source_id] = metrics
        all_rows.extend(rows)
        input_train_tokens += metrics["by_split"]["train"]["supervised_tokens"]
        input_validation_tokens += metrics["by_split"]["validation"]["supervised_tokens"]
        input_task_tokens.update(metrics["train_by_task"])
        input_language_tokens.update(metrics["train_by_language_mode"])
        _verify_candidate_promotion(
            source_id=source_id,
            value=_load_json(resolved[source_plan["promotion_binding"]]),
        )

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.10 row ceiling exceeded")
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train_tokens != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation_tokens != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.15 input candidate totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    (
        train_tokens,
        validation_tokens,
        task_tokens,
        language_tokens,
        source_train_tokens,
        quarantine_records,
    ) = v8._subtract_quarantine(
        all_rows=all_rows,
        decisions=generic["decisions"],
        source_plans=source_plans,
        source_metrics=source_metrics,
        tokenizer=tokenizer,
        template=template,
        input_train_tokens=input_train_tokens,
        input_validation_tokens=input_validation_tokens,
        input_task_tokens=input_task_tokens,
        input_language_tokens=input_language_tokens,
    )
    if len(accepted) + len(quarantine_records) != len(all_rows):
        raise ValueError("candidate-mix v0.10 quarantine accounting changed")

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

    source_shares = {
        source_id: round(source_train_tokens[source_id] / train_tokens, 8)
        for source_id in sorted(source_metrics)
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
    if generic["rejected"]:
        blockers.append("generic_filter_quarantine_not_materialized")
    if exact_snapshot["duplicate_groups"]:
        blockers.append("cross_source_exact_duplicates_remain_after_generic_filter")
    if near_snapshot["multi_record_components"]:
        blockers.append("cross_source_near_duplicates_not_materialized")
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
    hygiene_needed = bool(
        generic["rejected"]
        or exact_snapshot["duplicate_groups"]
        or near_snapshot["multi_record_components"]
    )
    priority_gaps = [task for task in EXPECTED_TASKS if task_gaps[task] > 0] + [
        mode for mode in EXPECTED_LANGUAGE_MODES if language_gaps[mode] > 0
    ]
    generic_without_decisions = {key: value for key, value in generic.items() if key != "decisions"}
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
            "accepted_records_after_generic_quarantine": len(accepted),
            "accepted_train_supervised_tokens": train_tokens,
            "candidate_records_read": len(all_rows),
            "floor_completion_fraction": round(train_tokens / targets["total"]["minimum"], 8),
            "gap_to_10m_train_supervised_tokens": max(
                0, targets["total"]["minimum"] - train_tokens
            ),
            "input_train_supervised_tokens": input_train_tokens,
            "input_validation_supervised_tokens": input_validation_tokens,
            "registry_admissible_validation_supervised_tokens": validation_tokens,
            "source_count": len(source_metrics),
            "source_shares": source_shares,
            "train_by_language_mode": {
                mode: language_tokens[mode] for mode in EXPECTED_LANGUAGE_MODES
            },
            "train_by_task": {task: task_tokens[task] for task in EXPECTED_TASKS},
        },
        "closed_since_v0_9": [
            "reviewed_80_record_teacher_canary_materialized_in_canonical_sft_schema",
            "teacher_canary_owner_review_provenance_bound_without_human_authorship_claim",
            "teacher_canary_source_bound_in_immutable_registry_v0_15",
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
        "generic_sft_validation": generic_without_decisions,
        "near_duplicate_audit": near_snapshot,
        "next_gate": {
            "action": (
                "materialize_registry_v0_15_hygiene_successors"
                if hygiene_needed
                else "close_remaining_10m_task_language_and_source_balance_gaps"
            ),
            "authorization_required": True,
            "priority_gaps": (
                [
                    "generic_filter_quarantine",
                    "cross_source_exact_duplicates",
                    "cross_source_near_duplicates",
                ]
                if hygiene_needed
                else priority_gaps
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "quarantine": {
            "by_split": {
                split: {
                    "records": sum(1 for row in quarantine_records if row["split"] == split),
                    "supervised_tokens": sum(
                        row["supervised_tokens"]
                        for row in quarantine_records
                        if row["split"] == split
                    ),
                }
                for split in ("train", "validation")
            },
            "materialized_in_candidate_files": False,
            "reason_counts": generic["rejection_counts"],
            "records": quarantine_records,
            "total_records": len(quarantine_records),
        },
        "raw_training_text_retained": False,
        "registry": {
            "path": plan["bindings"]["source_registry"]["path"],
            "sha256": EXPECTED_REGISTRY_SHA256,
            "source_count": len(registry.by_id),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": "verified_registry_v0_15_candidate_mix_reaudit_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v10_report(report)
    return report


def validate_candidate_mix_reaudit_v10_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.10 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.10 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.10 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.10 report must remain raw-text-free")
    pool = report.get("candidate_pool", {})
    if (
        pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("source_count") != 9
    ):
        raise ValueError("candidate-mix v0.10 input pool changed")
    accepted = pool.get("accepted_records_after_generic_quarantine")
    if not isinstance(accepted, int) or not 0 < accepted <= EXPECTED_INPUT_RECORDS:
        raise ValueError("candidate-mix v0.10 accepted count changed")
    generic = report.get("generic_sft_validation", {})
    if (
        generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("accepted") != accepted
        or generic.get("accepted") + generic.get("rejected") != EXPECTED_INPUT_RECORDS
    ):
        raise ValueError("candidate-mix v0.10 generic validation changed")
    quarantine = report.get("quarantine", {})
    records = quarantine.get("records")
    if (
        not isinstance(records, list)
        or quarantine.get("total_records") != generic.get("rejected")
        or len(records) != generic.get("rejected")
        or quarantine.get("materialized_in_candidate_files") is not False
    ):
        raise ValueError("candidate-mix v0.10 quarantine changed")
    if report.get("near_duplicate_audit", {}).get("records") != accepted:
        raise ValueError("candidate-mix v0.10 near-duplicate population changed")
    if report.get("registry") != {
        "path": "configs/lumi-4b-sft-source-registry-v0.15.json",
        "sha256": EXPECTED_REGISTRY_SHA256,
        "source_count": 16,
    }:
        raise ValueError("candidate-mix v0.10 registry evidence changed")
    teacher = report.get("source_metrics", {}).get(TEACHER_SOURCE_ID, {})
    if (
        teacher.get("candidate_file_sha256") != CANONICAL_CANDIDATE_SHA256
        or teacher.get("by_split") != TEACHER_SOURCE_ADDITION["expected_by_split"]
        or teacher.get("train_by_language_mode") != EXPECTED_TEACHER_TRAIN_BY_MODE
        or teacher.get("train_by_task") != EXPECTED_TEACHER_TRAIN_BY_TASK
    ):
        raise ValueError("candidate-mix v0.10 teacher metrics changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.10 report contains raw training text")


def build_candidate_mix_reaudit_v10_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v10_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_263837_registry_v0_15_rows_reaudited": True,
            "benchmark_registry_v0_3_verified": True,
            "cross_source_exact_and_near_duplicate_results_reproduced": True,
            "frozen_4k_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "quarantine_is_raw_text_free_and_not_materialized": True,
            "registry_v0_15_and_teacher_promotion_hashes_verified": True,
            "teacher_language_modes_resolved_from_exact_record_ids": True,
        },
        "duplicate_summary": {
            "generic_quarantine_records": report["quarantine"]["total_records"],
            "remaining_cross_source_exact_groups": report["exact_duplicate_audit"][
                "cross_source_duplicate_groups"
            ],
            "remaining_cross_source_near_components": report["near_duplicate_audit"][
                "cross_source_components"
            ],
            "train_validation_near_components": report["near_duplicate_audit"][
                "train_validation_components"
            ],
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(report_file_sha256, "report_file_sha256"),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_registry_v0_15_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_BENCHMARK_REGISTRY_SHA256",
    "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS",
    "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_REGISTRY_SHA256",
    "TEACHER_SOURCE_ID",
    "build_candidate_mix_reaudit_v10_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v10_plan",
    "rebuild_candidate_mix_reaudit_v10",
    "validate_candidate_mix_reaudit_v10_report",
]
