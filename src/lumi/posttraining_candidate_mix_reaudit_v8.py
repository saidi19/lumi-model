"""Registry-v0.13-bound full local re-audit of the Lumi SFT candidate pool."""

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
from . import posttraining_candidate_mix_reaudit_v7 as v7
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _language_mode,
    _load_json,
    _load_jsonl,
    _measure_conversation,
    _require_sha256,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_oasst2_source_promotion import (
    validate_promotion_receipt as validate_oasst2_promotion,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.8"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.8"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.8"

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
    "purpose": "registry_v0_13_full_local_candidate_mix_reaudit_only",
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
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.8.json",
    "verification_path": ("evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.8.json"),
}
EXPECTED_BINDINGS = frozenset(
    {
        "implementation",
        "oasst2_candidate",
        "oasst2_promotion",
        "oasst2_promotion_verification",
        "oasst2_token_metrics",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)

EXPECTED_REGISTRY_SHA256 = "1e8f7b0d2f1bfc10118fd76a51813e942cc7d0ce558f73ebfdc0382e6ee2b077"
EXPECTED_BENCHMARK_REGISTRY_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "af75dbd29de27b97fdc9111969ca55c79bc97592f5c8ef23f228e2c2be3cea9f"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "7ba47b82bb522caa5edea3d6d87817648173246b9b8d8b7131a7dd1386d487f3"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "af1f78358d09735f828d95bb5638f3748d1670b2b19f17e2c8be7a9bb27a1f51"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "a1bda9ccf62fd274daa4fec0d84766ea4dc45c51c79dbbc2e83e3dcc38832787"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "42cebb5e8d01ab5ff9da397493dd0e1da1e60e8a3585d748763460297cc4b8b3"
)
EXPECTED_OASST2_CANDIDATE_SHA256 = (
    "1642c444f75f29c70f5182d0ff0ff94133f52b942e36fb3655be62caa4a4e6df"
)
EXPECTED_OASST2_TOKEN_METRICS_SHA256 = (
    "54a5d9581b23fe121a0bc2e5d099a34e442a25af92a9215c973df78cf61e4624"
)
EXPECTED_OASST2_PROMOTION_SHA256 = (
    "1fce13c4852f3c03da5c0063d3134d9103d8f23460786198bd56b257ee5e2bdd"
)
EXPECTED_OASST2_PROMOTION_INTERNAL_SHA256 = (
    "3f246a80e7df4577064db526125ac0e07fd2c8977eda757fc8a4974ebed55b7a"
)
EXPECTED_OASST2_PROMOTION_VERIFICATION_SHA256 = (
    "2535ca37d1b41aef15e1bf5f8ed3671359147801bbb0200a7fb45d41f0506deb"
)
EXPECTED_OASST2_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "2ada9ac684554eb736b450499f2e770cc9b8888f88b69b8aa605d3a9d13db331"
)

EXPECTED_INPUT_RECORDS = 265_007
EXPECTED_INPUT_TRAIN_TOKENS = 7_584_780
EXPECTED_INPUT_VALIDATION_TOKENS = 415_236
OASST2_SOURCE_ID = "openassistant_oasst2"
EXPECTED_SOURCE_ADDITION = {
    "candidate_binding": "oasst2_candidate",
    "candidate_file_sha256": EXPECTED_OASST2_CANDIDATE_SHA256,
    "expected_by_split": {
        "train": {"records": 6_136, "supervised_tokens": 2_221_091},
        "validation": {"records": 290, "supervised_tokens": 96_436},
    },
    "expected_pending_owner_review_rows": 0,
    "language_mode_policy": "language_field",
    "promotion_binding": "oasst2_promotion",
    "registry_status": "canary_eligible",
    "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "source_id": OASST2_SOURCE_ID,
    "split_location": "top_level",
    "token_evidence_binding": "oasst2_token_metrics",
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.8 binding set changed")
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
    return v7.load_candidate_mix_reaudit_v7_plan(path, repository_root=repository_root)


def _validate_v8_evidence(resolved: Mapping[str, Path]) -> None:
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    v7.validate_candidate_mix_reaudit_v7_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256") != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.8 predecessor evidence changed")

    promotion = _load_json(resolved["oasst2_promotion"])
    verification = _load_json(resolved["oasst2_promotion_verification"])
    token_metrics = _load_json(resolved["oasst2_token_metrics"])
    validate_oasst2_promotion(promotion)
    _verify_internal_sha(verification, "verification_sha256")
    if promotion.get("receipt_sha256") != EXPECTED_OASST2_PROMOTION_INTERNAL_SHA256:
        raise ValueError("candidate-mix v0.8 OASST2 promotion receipt changed")
    if verification.get("verification_sha256") != (
        EXPECTED_OASST2_PROMOTION_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.8 OASST2 promotion verification changed")
    if verification.get("successor_registry_file_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.8 OASST2 registry binding changed")
    if token_metrics.get("accepted_records") != 6_426:
        raise ValueError("candidate-mix v0.8 OASST2 token-metrics count changed")
    if token_metrics.get("by_split") != {
        "train": {
            "masked_tokens": 263_945,
            "records": 6_136,
            "serialized_tokens": 2_485_036,
            "supervised_tokens": 2_221_091,
        },
        "validation": {
            "masked_tokens": 11_017,
            "records": 290,
            "serialized_tokens": 107_453,
            "supervised_tokens": 96_436,
        },
    }:
        raise ValueError("candidate-mix v0.8 OASST2 split totals changed")


def load_candidate_mix_reaudit_v8_plan(
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
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_addition": EXPECTED_SOURCE_ADDITION,
        "status": "authorized_registry_v0_13_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.8 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    exact_bindings = {
        "oasst2_candidate": EXPECTED_OASST2_CANDIDATE_SHA256,
        "oasst2_promotion": EXPECTED_OASST2_PROMOTION_SHA256,
        "oasst2_promotion_verification": (EXPECTED_OASST2_PROMOTION_VERIFICATION_SHA256),
        "oasst2_token_metrics": EXPECTED_OASST2_TOKEN_METRICS_SHA256,
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact_bindings.items():
        if plan["bindings"][name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.8 binding changed: {name}")
    _load_predecessor_plan(plan, repository_root=repository_root)
    _validate_v8_evidence(resolved)
    return plan


def _effective_plan(plan: Mapping[str, Any], *, repository_root: str | Path) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(v7._effective_plan(predecessor, repository_root=repository_root))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(plan["bindings"]["source_registry"])
    if any(source["source_id"] == OASST2_SOURCE_ID for source in effective["candidate_sources"]):
        raise ValueError("candidate-mix v0.8 OASST2 source already exists")
    effective["candidate_sources"].append(copy.deepcopy(plan["source_addition"]))
    return effective


def _verify_candidate_promotion(*, source_id: str, value: Mapping[str, Any]) -> None:
    if source_id == OASST2_SOURCE_ID:
        validate_oasst2_promotion(value)
        if value.get("receipt_sha256") != EXPECTED_OASST2_PROMOTION_INTERNAL_SHA256:
            raise ValueError("OASST2 promotion receipt changed")
        return
    v6._verify_candidate_promotion(source_id=source_id, value=value)


def _row_language_mode(row: Mapping[str, Any], source_plan: Mapping[str, Any]) -> str:
    if (
        source_plan["language_mode_policy"] == "language_field_multilingual_as_banglish"
        and row.get("language") == "multilingual"
    ):
        return "banglish"
    return _language_mode(row, source_plan["language_mode_policy"])


def _subtract_quarantine(
    *,
    all_rows: list[dict[str, Any]],
    decisions: list[Mapping[str, Any]],
    source_plans: Mapping[str, Mapping[str, Any]],
    source_metrics: dict[str, dict[str, Any]],
    tokenizer: Any,
    template: Any,
    input_train_tokens: int,
    input_validation_tokens: int,
    input_task_tokens: Counter[str],
    input_language_tokens: Counter[str],
) -> tuple[
    int,
    int,
    Counter[str],
    Counter[str],
    Counter[str],
    list[dict[str, Any]],
]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in rows_by_id:
            raise ValueError("candidate-mix v0.8 global record IDs changed")
        rows_by_id[record_id] = row

    train_tokens = input_train_tokens
    validation_tokens = input_validation_tokens
    task_tokens = Counter(input_task_tokens)
    language_tokens = Counter(input_language_tokens)
    source_train_tokens = Counter(
        {
            source_id: metrics["by_split"]["train"]["supervised_tokens"]
            for source_id, metrics in source_metrics.items()
        }
    )
    rejected_by_source: Counter[str] = Counter()
    quarantine_by_source_split: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    quarantine: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("accepted") is True:
            continue
        record_id = decision.get("record_id")
        if not isinstance(record_id, str) or record_id not in rows_by_id:
            raise ValueError("candidate-mix v0.8 rejection ID changed")
        row = rows_by_id[record_id]
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("candidate-mix v0.8 rejected source metadata changed")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or source_id not in source_plans:
            raise ValueError("candidate-mix v0.8 rejected source ID changed")
        split = row.get("split")
        task = row.get("task_category")
        if split not in {"train", "validation"} or not isinstance(task, str):
            raise ValueError("candidate-mix v0.8 rejected split/task changed")
        tokens = _measure_conversation(row["messages"], tokenizer=tokenizer, template=template)
        if split == "train":
            train_tokens -= tokens
            task_tokens[task] -= tokens
            language_tokens[_row_language_mode(row, source_plans[source_id])] -= tokens
            source_train_tokens[source_id] -= tokens
        else:
            validation_tokens -= tokens
        rejected_by_source[source_id] += 1
        quarantine_by_source_split[source_id][str(split)]["records"] += 1
        quarantine_by_source_split[source_id][str(split)]["supervised_tokens"] += tokens
        reasons = decision.get("reasons")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise ValueError("candidate-mix v0.8 rejection reasons changed")
        quarantine.append(
            {
                "reasons": reasons,
                "record_id": record_id,
                "source_id": source_id,
                "split": split,
                "supervised_tokens": tokens,
            }
        )

    for source_id, metrics in source_metrics.items():
        metrics["generic_sft_accepted_records"] = metrics["records"] - rejected_by_source[source_id]
        metrics["generic_sft_rejected_records"] = rejected_by_source[source_id]
        metrics["quarantined_by_split"] = {
            split: {
                "records": quarantine_by_source_split[source_id][split]["records"],
                "supervised_tokens": quarantine_by_source_split[source_id][split][
                    "supervised_tokens"
                ],
            }
            for split in ("train", "validation")
        }
    if min([train_tokens, validation_tokens, *task_tokens.values(), *language_tokens.values()]) < 0:
        raise ValueError("candidate-mix v0.8 quarantine subtraction underflowed")
    return (
        train_tokens,
        validation_tokens,
        task_tokens,
        language_tokens,
        source_train_tokens,
        quarantine,
    )


def rebuild_candidate_mix_reaudit_v8(
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
    predecessor_v7 = _load_predecessor_plan(plan, repository_root=root)
    predecessor_v6 = v7._load_predecessor_plan(predecessor_v7, repository_root=root)
    predecessor_v5 = v6._load_predecessor_plan(predecessor_v6, repository_root=root)
    predecessor_v4 = v5._load_predecessor_plan(predecessor_v5, repository_root=root)
    predecessor_effective = v7._effective_plan(predecessor_v7, repository_root=root)
    predecessor_resolved = {
        name: v2.resolve_binding(root, binding, name=name)
        for name, binding in predecessor_effective["bindings"].items()
    }
    predecessor_resolved.update(v4._resolve_explicit_bindings(predecessor_v4, root))
    predecessor_resolved.update(v7._resolve_explicit_bindings(predecessor_v7, root))
    predecessor_resolved["source_registry"] = v2.resolve_binding(
        root,
        predecessor_v7["bindings"]["source_registry"],
        name="source_registry",
    )
    v3._validate_bound_evidence(predecessor_resolved)
    v4._validate_new_evidence(predecessor_resolved)
    v5._validate_deepmind_evidence(predecessor_resolved)
    v6._validate_opencode_evidence(predecessor_resolved)
    v7._validate_v7_evidence(predecessor_resolved)
    _validate_v8_evidence(resolved)

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.8 benchmark registry changed")
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
        resolved["tokenizer_json"].parent, local_files_only=True
    )
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
        if (
            metrics["rows_with_stale_pending_human_review_provenance"]
            != (source_plan["expected_pending_owner_review_rows"])
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
        promotion = _load_json(resolved[source_plan["promotion_binding"]])
        _verify_candidate_promotion(source_id=source_id, value=promotion)

    if len(all_rows) > effective["limits"]["maximum_candidate_records_read"]:
        raise ValueError("candidate-mix v0.8 row ceiling exceeded")
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train_tokens != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation_tokens != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.13 input candidate totals changed")

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
    ) = _subtract_quarantine(
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
        raise ValueError("candidate-mix v0.8 quarantine accounting changed")

    accepted_duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        accepted_duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(accepted_duplicate_index)
    near_snapshot = v2._near_duplicate_audit(accepted, policy=effective["near_duplicate_policy"])

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
        task: max(0, targets["task_minimums"][task] - task_tokens[task]) for task in EXPECTED_TASKS
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
        "closed_since_v0_7": [
            "oasst2_6426_record_candidate_bound_in_immutable_registry_v0_13",
            "oasst2_apache_2_0_attribution_and_owner_review_evidence_verified",
            "oasst2_records_reaudited_against_the_full_local_candidate_mix",
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
                "materialize_registry_v0_13_cross_source_hygiene_successors"
                if hygiene_needed
                else "close_remaining_data_scale_task_language_and_source_balance_gaps"
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
        "status": "verified_registry_v0_13_candidate_mix_requires_hygiene_and_closure",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v8_report(report)
    return report


def validate_candidate_mix_reaudit_v8_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.8 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.8 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.8 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.8 report must remain raw-text-free")
    pool = report.get("candidate_pool", {})
    if (
        pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens") != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("source_count") != 8
    ):
        raise ValueError("candidate-mix v0.8 input pool changed")
    accepted = pool.get("accepted_records_after_generic_quarantine")
    if not isinstance(accepted, int) or not 0 < accepted <= EXPECTED_INPUT_RECORDS:
        raise ValueError("candidate-mix v0.8 accepted count changed")
    generic = report.get("generic_sft_validation", {})
    if (
        generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("accepted") != accepted
        or generic.get("accepted") + generic.get("rejected") != EXPECTED_INPUT_RECORDS
    ):
        raise ValueError("candidate-mix v0.8 generic validation changed")
    quarantine = report.get("quarantine", {})
    records = quarantine.get("records")
    if (
        not isinstance(records, list)
        or quarantine.get("total_records") != generic.get("rejected")
        or len(records) != generic.get("rejected")
        or quarantine.get("materialized_in_candidate_files") is not False
    ):
        raise ValueError("candidate-mix v0.8 quarantine changed")
    near = report.get("near_duplicate_audit", {})
    if near.get("records") != accepted:
        raise ValueError("candidate-mix v0.8 near-duplicate population changed")
    if report.get("registry") != {
        "path": "configs/lumi-4b-sft-source-registry-v0.13.json",
        "sha256": EXPECTED_REGISTRY_SHA256,
        "source_count": 15,
    }:
        raise ValueError("candidate-mix v0.8 registry evidence changed")
    benchmark = report.get("benchmark_decontamination", {})
    if (
        benchmark.get("registry_sha256") != EXPECTED_BENCHMARK_REGISTRY_SHA256
        or benchmark.get("registry_entries") != 50_944
    ):
        raise ValueError("candidate-mix v0.8 benchmark evidence changed")
    oasst2 = report.get("source_metrics", {}).get(OASST2_SOURCE_ID, {})
    if oasst2.get("candidate_file_sha256") != EXPECTED_OASST2_CANDIDATE_SHA256 or oasst2.get(
        "by_split"
    ) != {
        "train": {"records": 6_136, "supervised_tokens": 2_221_091},
        "validation": {"records": 290, "supervised_tokens": 96_436},
    }:
        raise ValueError("candidate-mix v0.8 OASST2 source metrics changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.8 report contains raw training text")


def build_candidate_mix_reaudit_v8_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v8_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_265007_registry_v0_13_rows_reaudited": True,
            "benchmark_registry_v0_3_verified": True,
            "cross_source_exact_and_near_duplicate_results_reproduced": True,
            "frozen_4k_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "oasst2_candidate_promotion_and_token_evidence_verified": True,
            "quarantine_is_raw_text_free_and_not_materialized": True,
            "registry_v0_13_hash_verified": True,
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
        "status": "verified_registry_v0_13_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_BENCHMARK_REGISTRY_SHA256",
    "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS",
    "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_REGISTRY_SHA256",
    "build_candidate_mix_reaudit_v8_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v8_plan",
    "rebuild_candidate_mix_reaudit_v8",
    "validate_candidate_mix_reaudit_v8_report",
]
