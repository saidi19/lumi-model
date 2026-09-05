"""Registry-v0.17-bound full local re-audit of the Lumi SFT candidate pool."""

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
from . import posttraining_candidate_mix_reaudit_v11 as v11
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
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.12"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.12"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.12"

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
    "purpose": "registry_v0_17_full_local_candidate_mix_reaudit_only",
    "source_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_per_pass": 264_981,
    "maximum_full_audit_passes": 2,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 180,
    "maximum_total_candidate_row_reads": 529_962,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.12.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.12.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "gpt_oss_scale_candidate",
        "gpt_oss_scale_promotion_receipt",
        "gpt_oss_scale_promotion_verification",
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

EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "c2ef51d27dfdc31d76d308d7239bd88f8d3f6eb66b339b6639945a2c6a6104e1"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "0eb5d92c83c6f403fbd5e5f0067b3c21f1aa64db352148eee6258a966db1b1da"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "f4bbb54b0c38ad8e6c16f5ff065df2b65ca2600179fc9c961193d2a82babf0df"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "4336f0b060d3624c825e40a67ae2ba25a167d48da3dbd127db57507ec738001d"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "7f0169bb9af2d638f77f2da77ca40e81aab0a4c1bf35407453e15cd4146c40e3"
)
EXPECTED_REGISTRY_SHA256 = (
    "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
)
EXPECTED_GPT_CANDIDATE_SHA256 = (
    "d0b2af4539ae156739ffe50a72245fd1faffc27538417b214dc9864627a2e4af"
)
EXPECTED_GPT_CANDIDATE_INTERNAL_SHA256 = (
    "510e08540a7d316fd219a58623150997a1c2f532a9590b3711241da66194ecac"
)
EXPECTED_GPT_PROMOTION_RECEIPT_SHA256 = (
    "4a320c11e431877ef78aaea846e9f51957f4fadf07d9b2a86c170535a22e0702"
)
EXPECTED_GPT_PROMOTION_RECEIPT_INTERNAL_SHA256 = (
    "38109ceb4c451cd9fc764838ab36b80ce15cbb9dc69d3e92eafad8a122b90103"
)
EXPECTED_GPT_PROMOTION_VERIFICATION_SHA256 = (
    "411657aa62c2d165598dd3d31ea3d3333bc8c17e29c73bc993e78b869432efb0"
)
EXPECTED_GPT_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "fa46a17d75f967341114fb3cc76a37f92c8e49ef0b24c9495fb8a2ddcbb1f23b"
)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v11.EXPECTED_BENCHMARK_REGISTRY_SHA256

EXPECTED_INPUT_RECORDS = 264_981
EXPECTED_INPUT_TRAIN_TOKENS = 7_341_582
EXPECTED_INPUT_VALIDATION_TOKENS = 427_005
EXPECTED_INCLUDED_SOURCE_COUNT = 11
EXPECTED_REGISTRY_SOURCE_COUNT = 18

GPT_SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_6"
GPT_REVISION = "07b673b7486cfd8b6916fa19d12d0d47b0078857da8aa8bd2fe9bf6f0834008f"
GPT_SOURCE_ADDITION = {
    "candidate_binding": "gpt_oss_scale_candidate",
    "candidate_file_sha256": EXPECTED_GPT_CANDIDATE_SHA256,
    "expected_by_split": {
        "train": {"records": 787, "supervised_tokens": 75_439},
        "validation": {"records": 197, "supervised_tokens": 18_974},
    },
    "expected_pending_owner_review_rows": 0,
    "language_mode_policy": "gpt_oss_scale_record_id_language_mode_v0.17",
    "promotion_binding": "gpt_oss_scale_promotion_receipt",
    "registry_status": "candidate_eligible",
    "revision": GPT_REVISION,
    "source_id": GPT_SOURCE_ID,
    "split_location": "top_level",
    "token_evidence_binding": "gpt_oss_scale_promotion_receipt",
}
EXPECTED_GPT_TRAIN_BY_MODE = {
    "banglish": 19_490,
    "bengali_script": 20_237,
    "english": 20_585,
    "natural_code_switching": 15_127,
}
EXPECTED_GPT_TRAIN_BY_TASK = {
    "factual_qa": 3_702,
    "multilingual_bengali": 15_997,
    "safety_helpfulness": 22_123,
    "tool_api": 19_521,
    "writing_summarization": 14_096,
}
_GPT_MODE = re.compile(
    r"-(banglish|bengali_script|english|natural_code_switching)-[0-9]{4}$"
)


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.12 binding set changed")
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
    return v11.load_candidate_mix_reaudit_v11_plan(
        path, repository_root=repository_root
    )


def _validate_bound_evidence(
    *,
    plan: Mapping[str, Any],
    resolved: Mapping[str, Path],
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_plan = _load_predecessor_plan(plan, repository_root=repository_root)
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    v11.validate_candidate_mix_reaudit_v11_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256")
        != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
        or predecessor_verification.get("report_file_sha256")
        != EXPECTED_PREDECESSOR_REPORT_SHA256
        or not all(predecessor_verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.12 predecessor evidence changed")

    receipt = _load_json(resolved["gpt_oss_scale_promotion_receipt"])
    verification = _load_json(resolved["gpt_oss_scale_promotion_verification"])
    _verify_internal_sha(receipt, "receipt_sha256")
    _verify_internal_sha(verification, "verification_sha256")
    if (
        receipt.get("receipt_sha256")
        != EXPECTED_GPT_PROMOTION_RECEIPT_INTERNAL_SHA256
        or receipt.get("candidate", {}).get("file_sha256")
        != EXPECTED_GPT_CANDIDATE_SHA256
        or receipt.get("registry", {}).get("successor_sha256")
        != EXPECTED_REGISTRY_SHA256
        or receipt.get("registry", {}).get("added_source_id") != GPT_SOURCE_ID
        or receipt.get("production_eligible") is not False
        or receipt.get("checks", {}).get("sft_started") is not False
        or verification.get("verification_sha256")
        != EXPECTED_GPT_PROMOTION_VERIFICATION_INTERNAL_SHA256
        or verification.get("status")
        != "PASS_verified_gpt_oss_scale_candidate_and_registry_v0_17_nonproduction"
        or verification.get("canonical_candidate", {}).get("records") != 984
        or verification.get("successor_registry", {}).get("source_count") != 18
        or not all(verification.get("checks", {}).values())
    ):
        raise ValueError("candidate-mix v0.12 gpt-oss promotion evidence changed")

    registry_v16 = _load_json(
        v2.resolve_binding(
            repository_root,
            predecessor_plan["bindings"]["source_registry"],
            name="predecessor_source_registry",
        )
    )
    registry_v17 = _load_json(resolved["source_registry"])
    if (
        registry_v17.get("sources", [])[:-1] != registry_v16.get("sources")
        or registry_v17.get("sources", [])[-1].get("source_id") != GPT_SOURCE_ID
        or len(registry_v17.get("sources", [])) != EXPECTED_REGISTRY_SOURCE_COUNT
    ):
        raise ValueError("registry v0.17 is not the exact append-only successor")
    return predecessor_report, receipt


def load_candidate_mix_reaudit_v12_plan(
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
        "observed_at": "2026-08-12",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_addition": GPT_SOURCE_ADDITION,
        "status": "authorized_registry_v0_17_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.12 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    exact = {
        "gpt_oss_scale_candidate": EXPECTED_GPT_CANDIDATE_SHA256,
        "gpt_oss_scale_promotion_receipt": EXPECTED_GPT_PROMOTION_RECEIPT_SHA256,
        "gpt_oss_scale_promotion_verification": (
            EXPECTED_GPT_PROMOTION_VERIFICATION_SHA256
        ),
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if plan["bindings"][name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.12 exact binding changed: {name}")
    _validate_bound_evidence(
        plan=plan,
        resolved=resolved,
        repository_root=repository_root,
    )
    return plan


def _effective_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(
        v11._effective_plan(predecessor, repository_root=repository_root)
    )
    if any(
        source.get("source_id") == GPT_SOURCE_ID
        for source in effective["candidate_sources"]
    ):
        raise ValueError("gpt-oss source already exists in predecessor candidate set")
    effective["candidate_sources"].append(copy.deepcopy(GPT_SOURCE_ADDITION))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(
        plan["bindings"]["source_registry"]
    )
    return effective


def _gpt_candidate_metrics(
    *,
    rows: list[dict[str, Any]],
    source_plan: Mapping[str, Any],
    registry_source: Any,
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    if (
        len(rows) != 984
        or source_plan.get("source_id") != GPT_SOURCE_ID
        or sha256_value(rows) != EXPECTED_GPT_CANDIDATE_INTERNAL_SHA256
    ):
        raise ValueError("gpt-oss candidate inventory changed")
    expected_language = {
        "banglish": "multilingual",
        "bengali_script": "bn",
        "english": "en",
        "natural_code_switching": "multilingual",
    }
    modes: dict[str, str] = {}
    mode_counts: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise ValueError("gpt-oss record ID changed")
        match = _GPT_MODE.search(record_id)
        if match is None:
            raise ValueError("gpt-oss record ID no longer binds a language mode")
        mode = match.group(1)
        if row.get("language") != expected_language[mode]:
            raise ValueError("gpt-oss language label and record ID disagree")
        modes[record_id] = mode
        mode_counts[mode] += 1
    if mode_counts != Counter(
        {
            "banglish": 261,
            "bengali_script": 166,
            "english": 421,
            "natural_code_switching": 136,
        }
    ):
        raise ValueError("gpt-oss language-mode inventory changed")

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
        duplicate_index=defaultdict(list),
    )
    train_by_mode: Counter[str] = Counter()
    for row in rows:
        split, _ = _effective_split(row, source_plan["split_location"])
        if split == "train":
            train_by_mode[modes[str(row["record_id"])]] += _measure_conversation(
                row["messages"], tokenizer=tokenizer, template=template
            )
    if dict(sorted(train_by_mode.items())) != EXPECTED_GPT_TRAIN_BY_MODE:
        raise ValueError("gpt-oss language-mode frozen tokens changed")
    if metrics["train_by_task"] != EXPECTED_GPT_TRAIN_BY_TASK:
        raise ValueError("gpt-oss task frozen tokens changed")
    metrics["train_by_language_mode"] = dict(sorted(train_by_mode.items()))
    metrics["language_mapping"] = {
        "contract": "scale_record_id_language_mode_v0.17",
        "records_by_mode": dict(sorted(mode_counts.items())),
    }
    return metrics


def _aggregate_input_metrics(
    source_metrics: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int, Counter[str], Counter[str]]:
    train = 0
    validation = 0
    tasks: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for metrics in source_metrics.values():
        train += int(metrics["by_split"]["train"]["supervised_tokens"])
        validation += int(metrics["by_split"]["validation"]["supervised_tokens"])
        tasks.update(metrics["train_by_task"])
        languages.update(metrics["train_by_language_mode"])
    return train, validation, tasks, languages


def rebuild_candidate_mix_reaudit_v12(
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
    predecessor_report, _receipt = _validate_bound_evidence(
        plan=plan, resolved=resolved, repository_root=root
    )

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(
        audit_policy.benchmark_decontamination, root
    )
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.12 benchmark registry changed")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT:
        raise ValueError("registry v0.17 source inventory changed")
    targets = _targets(_load_json(resolved["missing_mix_proposal"]))
    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    old_metrics = predecessor_report["source_metrics"]
    source_metrics: dict[str, dict[str, Any]] = {}
    source_plans: dict[str, Mapping[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for source_plan in sorted(
        effective["candidate_sources"], key=lambda item: item["source_id"]
    ):
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
        if any(row.get("source", {}).get("source_id") != source_id for row in rows):
            raise ValueError(f"candidate row source binding changed: {source_id}")
        if source_id == GPT_SOURCE_ID:
            metrics = _gpt_candidate_metrics(
                rows=rows,
                source_plan=source_plan,
                registry_source=registry_source,
                tokenizer=tokenizer,
                template=template,
            )
        else:
            metrics = copy.deepcopy(old_metrics.get(source_id))
            if not isinstance(metrics, dict):
                raise ValueError(f"predecessor metrics missing: {source_id}")
            if (
                metrics.get("candidate_file_sha256")
                != source_plan["candidate_file_sha256"]
                or metrics.get("records") != len(rows)
                or metrics.get("by_split") != source_plan["expected_by_split"]
            ):
                raise ValueError(f"predecessor source metrics changed: {source_id}")
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

    if len(source_metrics) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.12 included source inventory changed")
    if len(all_rows) > effective["limits"]["maximum_candidate_records_per_pass"]:
        raise ValueError("candidate-mix v0.12 row ceiling exceeded")
    input_train_tokens, input_validation_tokens, input_task_tokens, input_language_tokens = (
        _aggregate_input_metrics(source_metrics)
    )
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train_tokens != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation_tokens != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.17 input candidate totals changed")

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
        raise ValueError("candidate-mix v0.12 quarantine accounting changed")

    accepted_duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        accepted_duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact_snapshot = _duplicate_snapshot(accepted_duplicate_index)
    near_snapshot = v2._near_duplicate_audit(
        accepted, policy=effective["near_duplicate_policy"]
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
    excluded = v2._verify_exclusions(
        plan=effective, resolved=resolved, registry=registry
    )
    hygiene_needed = bool(
        generic["rejected"]
        or exact_snapshot["duplicate_groups"]
        or near_snapshot["multi_record_components"]
    )
    priority_gaps = [task for task in EXPECTED_TASKS if task_gaps[task] > 0] + [
        mode for mode in EXPECTED_LANGUAGE_MODES if language_gaps[mode] > 0
    ]
    generic_without_decisions = {
        key: value for key, value in generic.items() if key != "decisions"
    }
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
            "floor_completion_fraction": round(
                train_tokens / targets["total"]["minimum"], 8
            ),
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
        "closed_since_v0_11": [
            "owner_accepted_984_record_gpt_oss_scale_candidate_materialized",
            "gpt_oss_scale_source_bound_in_immutable_registry_v0_17",
            "frozen_scale_schedule_split_bound_and_tokenized",
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
                "materialize_registry_v0_17_hygiene_successors"
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
        "plan_file_sha256": _require_sha256(
            plan_file_sha256, "plan_file_sha256"
        ),
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "quarantine": {
            "by_split": {
                split: {
                    "records": sum(
                        1 for row in quarantine_records if row["split"] == split
                    ),
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
        "status": "verified_registry_v0_17_candidate_mix_reaudit_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v12_report(report)
    return report


def validate_candidate_mix_reaudit_v12_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.12 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("candidate-mix v0.12 authorization changed")
    if (
        report.get("status")
        != "verified_registry_v0_17_candidate_mix_reaudit_nonproduction"
        or report.get("production_eligible") is not False
        or report.get("production_dataset_promotions") != 0
        or report.get("raw_training_text_retained") is not False
    ):
        raise ValueError("candidate-mix v0.12 production boundary changed")
    pool = report.get("candidate_pool", {})
    if (
        pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("source_count") != EXPECTED_INCLUDED_SOURCE_COUNT
    ):
        raise ValueError("candidate-mix v0.12 input pool changed")
    accepted = pool.get("accepted_records_after_generic_quarantine")
    if not isinstance(accepted, int) or not 0 < accepted <= EXPECTED_INPUT_RECORDS:
        raise ValueError("candidate-mix v0.12 accepted count changed")
    generic = report.get("generic_sft_validation", {})
    if (
        generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("accepted") != accepted
        or generic.get("accepted") + generic.get("rejected")
        != EXPECTED_INPUT_RECORDS
    ):
        raise ValueError("candidate-mix v0.12 generic validation changed")
    quarantine = report.get("quarantine", {})
    records = quarantine.get("records")
    if (
        not isinstance(records, list)
        or quarantine.get("total_records") != generic.get("rejected")
        or len(records) != generic.get("rejected")
        or quarantine.get("materialized_in_candidate_files") is not False
    ):
        raise ValueError("candidate-mix v0.12 quarantine changed")
    if report.get("near_duplicate_audit", {}).get("records") != accepted:
        raise ValueError("candidate-mix v0.12 near-duplicate population changed")
    if report.get("registry") != {
        "path": "configs/lumi-4b-sft-source-registry-v0.17.json",
        "sha256": EXPECTED_REGISTRY_SHA256,
        "source_count": EXPECTED_REGISTRY_SOURCE_COUNT,
    }:
        raise ValueError("candidate-mix v0.12 registry evidence changed")
    gpt = report.get("source_metrics", {}).get(GPT_SOURCE_ID, {})
    if (
        gpt.get("candidate_file_sha256") != EXPECTED_GPT_CANDIDATE_SHA256
        or gpt.get("by_split") != GPT_SOURCE_ADDITION["expected_by_split"]
        or gpt.get("train_by_language_mode") != EXPECTED_GPT_TRAIN_BY_MODE
        or gpt.get("train_by_task") != EXPECTED_GPT_TRAIN_BY_TASK
    ):
        raise ValueError("candidate-mix v0.12 gpt-oss metrics changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        '"messages"',
        '"content"',
        '"assistant_response"',
        '"user_prompt"',
    ):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.12 report contains raw training text")


def build_candidate_mix_reaudit_v12_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v12_report(report)
    verification: dict[str, Any] = {
        "candidate_pool": {
            "accepted_records": report["candidate_pool"][
                "accepted_records_after_generic_quarantine"
            ],
            "accepted_train_supervised_tokens": report["candidate_pool"][
                "accepted_train_supervised_tokens"
            ],
            "input_records": EXPECTED_INPUT_RECORDS,
            "input_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "input_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
        },
        "checks": {
            "all_264981_registry_v0_17_rows_reaudited": True,
            "benchmark_registry_v0_3_verified": True,
            "cross_source_exact_and_near_duplicate_results_reproduced": True,
            "frozen_4k_token_task_language_and_source_totals_reproduced": True,
            "gpt_oss_promotion_candidate_and_language_modes_verified": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "quarantine_is_raw_text_free_and_not_materialized": True,
            "registry_v0_17_append_only_identity_verified": True,
        },
        "duplicate_summary": {
            "generic_quarantine_records": report["quarantine"]["total_records"],
            "remaining_cross_source_exact_groups": report[
                "exact_duplicate_audit"
            ]["cross_source_duplicate_groups"],
            "remaining_cross_source_near_components": report[
                "near_duplicate_audit"
            ]["cross_source_components"],
            "train_validation_near_components": report["near_duplicate_audit"][
                "train_validation_components"
            ],
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "registry_file_sha256": EXPECTED_REGISTRY_SHA256,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_registry_v0_17_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS",
    "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_REGISTRY_SHA256",
    "GPT_SOURCE_ID",
    "build_candidate_mix_reaudit_v12_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v12_plan",
    "rebuild_candidate_mix_reaudit_v12",
    "validate_candidate_mix_reaudit_v12_report",
]
