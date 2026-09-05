"""Registry-v0.14-bound full local re-audit of the Lumi SFT candidate pool."""

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
from . import posttraining_candidate_mix_reaudit_v8 as v8
from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_hygiene_materialization_v2 import (
    validate_hygiene_materialization_v2_manifest,
    validate_hygiene_materialization_v2_receipt,
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
from .posttraining_hygiene_registry_promotion_v2 import (
    validate_registry_v14_promotion_receipt,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.9"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.9"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.9"

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
    "purpose": "registry_v0_14_full_local_candidate_mix_reaudit_only",
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
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.9.json",
    "verification_path": ("evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.9.json"),
}
EXPECTED_BINDINGS = frozenset(
    {
        "hygiene_materialization_manifest",
        "hygiene_materialization_receipt",
        "hygiene_materialization_verification",
        "implementation",
        "oasst1_clean_candidate",
        "oasst2_clean_candidate",
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

EXPECTED_REGISTRY_SHA256 = "a7310e469abb06cf4fbd12a72b2dcf2fbbebeac064e7b9182b994a9a77254e16"
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v8.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_PREDECESSOR_PLAN_SHA256 = (
    "83eb5b2186069d172aa679c26fc8a7f3161acb0df32bf4a0f502e03e263b4a48"
)
EXPECTED_PREDECESSOR_REPORT_SHA256 = (
    "cf6246e42b5d4a78a1e0bff461b05eaf71def13e3a3938b1ccdc95c0a965fb90"
)
EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256 = (
    "c59536268128799d08e83d2a022566b76bcd4c11ffa8795bf225d464d68683b6"
)
EXPECTED_PREDECESSOR_VERIFICATION_SHA256 = (
    "bcc6e91c94ee8a99eaaf55e1cf2ea9cf9417188ade428a08b6a3088fae926aa8"
)
EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256 = (
    "89de48a1be6d6d6c49a4031c5fa53b4fcec574930079c52ed4faff4cf17b8f0e"
)
EXPECTED_MATERIALIZATION_MANIFEST_SHA256 = (
    "ba4dcd6302678d5deb2029514db667ed7563a0ebfb0921e475e3d363cf9d948d"
)
EXPECTED_MATERIALIZATION_MANIFEST_INTERNAL_SHA256 = (
    "fec0e85da3b187de02fea3675d0859ec4028b15ebeda1b1981f9e5017c7d1f55"
)
EXPECTED_MATERIALIZATION_RECEIPT_SHA256 = (
    "600c0878f8400ce7ce166e7bbbfb4cdcb7a17252b195377470625236fae95062"
)
EXPECTED_MATERIALIZATION_RECEIPT_INTERNAL_SHA256 = (
    "5082f5d04a9eeea861ea9350cbe8dc1bb6612926fdce517252893fd915884ce4"
)
EXPECTED_MATERIALIZATION_VERIFICATION_SHA256 = (
    "3cd8e9642894f7657164df82a200119115960745d2d6786c9276fbb32de709af"
)
EXPECTED_MATERIALIZATION_VERIFICATION_INTERNAL_SHA256 = (
    "7c899c4c0157dc46dc3f677440331bdc233feae9f626432e2bbe1be1ccc080bb"
)
EXPECTED_REGISTRY_PROMOTION_SHA256 = (
    "04003f81b8006db1f21003076b3b18cfbc9077abbd69897021587ad248803da9"
)
EXPECTED_REGISTRY_PROMOTION_INTERNAL_SHA256 = (
    "84abaea23a8bb22f0d06974ac99ac99c0f8639c591d383f6d9dc6e4276f88538"
)
EXPECTED_REGISTRY_PROMOTION_VERIFICATION_SHA256 = (
    "6a526c0652169c699bc595cd73de691c372a60e0f0f45180909025ed7b34e613"
)
EXPECTED_REGISTRY_PROMOTION_VERIFICATION_INTERNAL_SHA256 = (
    "d2a229f65c8ce6de4519947a21d734c1d2366eeaf32bd11b8043711aa1c09d1f"
)
EXPECTED_OASST1_CLEAN_SHA256 = "6a0828f05bcf737921e9388d00705e7dd0a07cf447ba6b35fec57124a29bc369"
EXPECTED_OASST2_CLEAN_SHA256 = "57ba5ea18e2f621b9bcc79d18340df92ba7bf3a0e378d3273498d5e497870459"

EXPECTED_INPUT_RECORDS = 263_757
EXPECTED_INPUT_TRAIN_TOKENS = 7_254_668
EXPECTED_INPUT_VALIDATION_TOKENS = 405_326
OASST1_SOURCE_ID = "openassistant_oasst1"
OASST2_SOURCE_ID = "openassistant_oasst2"
SOURCE_REPLACEMENTS = {
    OASST1_SOURCE_ID: {
        "candidate_binding": "oasst1_clean_candidate",
        "candidate_file_sha256": EXPECTED_OASST1_CLEAN_SHA256,
        "expected_by_split": {
            "train": {"records": 1_627, "supervised_tokens": 447_746},
            "validation": {"records": 93, "supervised_tokens": 23_475},
        },
        "expected_pending_owner_review_rows": 0,
        "language_mode_policy": "language_field",
        "promotion_binding": "registry_promotion",
        "registry_status": "canary_eligible",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "source_id": OASST1_SOURCE_ID,
        "split_location": "top_level",
        "token_evidence_binding": "hygiene_materialization_receipt",
    },
    OASST2_SOURCE_ID: {
        "candidate_binding": "oasst2_clean_candidate",
        "candidate_file_sha256": EXPECTED_OASST2_CLEAN_SHA256,
        "expected_by_split": {
            "train": {"records": 4_944, "supervised_tokens": 1_895_923},
            "validation": {"records": 254, "supervised_tokens": 86_526},
        },
        "expected_pending_owner_review_rows": 0,
        "language_mode_policy": "language_field",
        "promotion_binding": "registry_promotion",
        "registry_status": "canary_eligible",
        "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
        "source_id": OASST2_SOURCE_ID,
        "split_location": "top_level",
        "token_evidence_binding": "hygiene_materialization_receipt",
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _resolve_explicit_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate-mix v0.9 binding set changed")
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
    return v8.load_candidate_mix_reaudit_v8_plan(path, repository_root=repository_root)


def _validate_v9_evidence(resolved: Mapping[str, Path]) -> None:
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    v8.validate_candidate_mix_reaudit_v8_report(predecessor_report)
    _verify_internal_sha(predecessor_verification, "verification_sha256")
    if (
        predecessor_report.get("report_sha256") != EXPECTED_PREDECESSOR_REPORT_INTERNAL_SHA256
        or predecessor_verification.get("verification_sha256")
        != EXPECTED_PREDECESSOR_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.9 predecessor evidence changed")

    manifest = _load_json(resolved["hygiene_materialization_manifest"])
    receipt = _load_json(resolved["hygiene_materialization_receipt"])
    verification = _load_json(resolved["hygiene_materialization_verification"])
    validate_hygiene_materialization_v2_manifest(manifest)
    validate_hygiene_materialization_v2_receipt(receipt)
    _verify_internal_sha(verification, "verification_sha256")
    if (
        manifest.get("manifest_sha256") != EXPECTED_MATERIALIZATION_MANIFEST_INTERNAL_SHA256
        or receipt.get("receipt_sha256") != EXPECTED_MATERIALIZATION_RECEIPT_INTERNAL_SHA256
        or verification.get("verification_sha256")
        != EXPECTED_MATERIALIZATION_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("candidate-mix v0.9 materialization evidence changed")

    promotion = _load_json(resolved["registry_promotion"])
    promotion_verification = _load_json(resolved["registry_promotion_verification"])
    validate_registry_v14_promotion_receipt(promotion)
    _verify_internal_sha(promotion_verification, "verification_sha256")
    if (
        promotion.get("receipt_sha256") != EXPECTED_REGISTRY_PROMOTION_INTERNAL_SHA256
        or promotion_verification.get("verification_sha256")
        != EXPECTED_REGISTRY_PROMOTION_VERIFICATION_INTERNAL_SHA256
        or promotion_verification.get("successor_registry_file_sha256") != EXPECTED_REGISTRY_SHA256
    ):
        raise ValueError("candidate-mix v0.9 registry promotion evidence changed")


def load_candidate_mix_reaudit_v9_plan(
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
        "source_replacements": [
            SOURCE_REPLACEMENTS[source_id] for source_id in sorted(SOURCE_REPLACEMENTS)
        ],
        "status": "authorized_registry_v0_14_full_local_reaudit_no_production_action",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate-mix v0.9 plan changed: {field}")
    resolved = _resolve_explicit_bindings(plan, repository_root)
    exact_bindings = {
        "hygiene_materialization_manifest": (EXPECTED_MATERIALIZATION_MANIFEST_SHA256),
        "hygiene_materialization_receipt": (EXPECTED_MATERIALIZATION_RECEIPT_SHA256),
        "hygiene_materialization_verification": (EXPECTED_MATERIALIZATION_VERIFICATION_SHA256),
        "oasst1_clean_candidate": EXPECTED_OASST1_CLEAN_SHA256,
        "oasst2_clean_candidate": EXPECTED_OASST2_CLEAN_SHA256,
        "predecessor_plan": EXPECTED_PREDECESSOR_PLAN_SHA256,
        "predecessor_report": EXPECTED_PREDECESSOR_REPORT_SHA256,
        "predecessor_verification": EXPECTED_PREDECESSOR_VERIFICATION_SHA256,
        "registry_promotion": EXPECTED_REGISTRY_PROMOTION_SHA256,
        "registry_promotion_verification": (EXPECTED_REGISTRY_PROMOTION_VERIFICATION_SHA256),
        "source_registry": EXPECTED_REGISTRY_SHA256,
    }
    for name, expected_sha256 in exact_bindings.items():
        if plan["bindings"][name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate-mix v0.9 binding changed: {name}")
    _load_predecessor_plan(plan, repository_root=repository_root)
    _validate_v9_evidence(resolved)
    return plan


def _effective_plan(plan: Mapping[str, Any], *, repository_root: str | Path) -> dict[str, Any]:
    predecessor = _load_predecessor_plan(plan, repository_root=repository_root)
    effective = copy.deepcopy(v8._effective_plan(predecessor, repository_root=repository_root))
    effective["authorization"] = dict(plan["authorization"])
    effective["limits"] = dict(plan["limits"])
    effective["near_duplicate_policy"] = dict(plan["near_duplicate_policy"])
    effective["observed_at"] = plan["observed_at"]
    effective["output"] = dict(plan["output"])
    effective["bindings"].update(plan["bindings"])
    effective["bindings"]["source_registry"] = dict(plan["bindings"]["source_registry"])
    replaced: set[str] = set()
    for index, source in enumerate(effective["candidate_sources"]):
        source_id = source["source_id"]
        if source_id in SOURCE_REPLACEMENTS:
            effective["candidate_sources"][index] = copy.deepcopy(SOURCE_REPLACEMENTS[source_id])
            replaced.add(source_id)
    if replaced != set(SOURCE_REPLACEMENTS):
        raise ValueError("candidate-mix v0.9 did not replace exactly OASST1/OASST2")
    return effective


def _verify_candidate_promotion(*, source_id: str, value: Mapping[str, Any]) -> None:
    if source_id in SOURCE_REPLACEMENTS:
        validate_registry_v14_promotion_receipt(value)
        if value.get("receipt_sha256") != EXPECTED_REGISTRY_PROMOTION_INTERNAL_SHA256:
            raise ValueError("registry-v0.14 promotion receipt changed")
        changed = [item.get("source_id") for item in value.get("changed_sources", [])]
        if source_id not in changed:
            raise ValueError(f"registry-v0.14 promotion omitted {source_id}")
        return
    v8._verify_candidate_promotion(source_id=source_id, value=value)


def rebuild_candidate_mix_reaudit_v9(
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
    _validate_v9_evidence(resolved)

    base_policy = load_sft_policy(resolved["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination=dict(plan["benchmark_decontamination"]),
    )
    benchmark_registry = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.9 benchmark registry changed")
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
        raise ValueError("candidate-mix v0.9 row ceiling exceeded")
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train_tokens != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation_tokens != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("registry-v0.14 input candidate totals changed")

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
        raise ValueError("candidate-mix v0.9 quarantine accounting changed")

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
        "closed_since_v0_8": [
            "oasst1_oasst2_1250_cross_source_hygiene_losers_materialized",
            "clean_6918_record_oasst1_oasst2_successors_hash_verified",
            "clean_successor_metadata_bound_in_immutable_registry_v0_14",
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
                "materialize_registry_v0_14_remaining_hygiene_successors"
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
        "status": "verified_registry_v0_14_candidate_mix_reaudit_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_mix_reaudit_v9_report(report)
    return report


def validate_candidate_mix_reaudit_v9_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.9 report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-mix v0.9 cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-mix v0.9 cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-mix v0.9 report must remain raw-text-free")
    pool = report.get("candidate_pool", {})
    if (
        pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens") != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("source_count") != 8
    ):
        raise ValueError("candidate-mix v0.9 input pool changed")
    accepted = pool.get("accepted_records_after_generic_quarantine")
    if not isinstance(accepted, int) or not 0 < accepted <= EXPECTED_INPUT_RECORDS:
        raise ValueError("candidate-mix v0.9 accepted count changed")
    generic = report.get("generic_sft_validation", {})
    if (
        generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("accepted") != accepted
        or generic.get("accepted") + generic.get("rejected") != EXPECTED_INPUT_RECORDS
    ):
        raise ValueError("candidate-mix v0.9 generic validation changed")
    quarantine = report.get("quarantine", {})
    records = quarantine.get("records")
    if (
        not isinstance(records, list)
        or quarantine.get("total_records") != generic.get("rejected")
        or len(records) != generic.get("rejected")
        or quarantine.get("materialized_in_candidate_files") is not False
    ):
        raise ValueError("candidate-mix v0.9 quarantine changed")
    if report.get("near_duplicate_audit", {}).get("records") != accepted:
        raise ValueError("candidate-mix v0.9 near-duplicate population changed")
    if report.get("registry") != {
        "path": "configs/lumi-4b-sft-source-registry-v0.14.json",
        "sha256": EXPECTED_REGISTRY_SHA256,
        "source_count": 15,
    }:
        raise ValueError("candidate-mix v0.9 registry evidence changed")
    benchmark = report.get("benchmark_decontamination", {})
    if (
        benchmark.get("registry_sha256") != EXPECTED_BENCHMARK_REGISTRY_SHA256
        or benchmark.get("registry_entries") != 50_944
    ):
        raise ValueError("candidate-mix v0.9 benchmark evidence changed")
    expected_oasst = {
        OASST1_SOURCE_ID: {
            "candidate_file_sha256": EXPECTED_OASST1_CLEAN_SHA256,
            "by_split": SOURCE_REPLACEMENTS[OASST1_SOURCE_ID]["expected_by_split"],
        },
        OASST2_SOURCE_ID: {
            "candidate_file_sha256": EXPECTED_OASST2_CLEAN_SHA256,
            "by_split": SOURCE_REPLACEMENTS[OASST2_SOURCE_ID]["expected_by_split"],
        },
    }
    for source_id, expected in expected_oasst.items():
        observed = report.get("source_metrics", {}).get(source_id, {})
        if (
            observed.get("candidate_file_sha256") != expected["candidate_file_sha256"]
            or observed.get("by_split") != expected["by_split"]
        ):
            raise ValueError(f"candidate-mix v0.9 source metrics changed: {source_id}")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate-mix v0.9 report contains raw training text")


def build_candidate_mix_reaudit_v9_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_candidate_mix_reaudit_v9_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_263757_registry_v0_14_rows_reaudited": True,
            "benchmark_registry_v0_3_verified": True,
            "cross_source_exact_and_near_duplicate_results_reproduced": True,
            "frozen_4k_token_task_language_and_source_totals_reproduced": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "oasst1_oasst2_clean_candidate_evidence_verified": True,
            "quarantine_is_raw_text_free_and_not_materialized": True,
            "registry_v0_14_and_promotion_hashes_verified": True,
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
        "status": "verified_registry_v0_14_candidate_mix_reaudit_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_BENCHMARK_REGISTRY_SHA256",
    "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS",
    "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_REGISTRY_SHA256",
    "build_candidate_mix_reaudit_v9_verification",
    "json_file_bytes",
    "load_candidate_mix_reaudit_v9_plan",
    "rebuild_candidate_mix_reaudit_v9",
    "validate_candidate_mix_reaudit_v9_report",
]
