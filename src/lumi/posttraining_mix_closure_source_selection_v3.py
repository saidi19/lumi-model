"""Registry-v0.15-bound source selection for closing the Lumi 4B 10M SFT mix."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_mix_closure_source_selection_v2 as v2
from .posttraining_candidate_mix_preflight import (
    _load_json,
    _verify_internal_sha,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding
from .posttraining_teacher_canary_source_promotion import (
    validate_promotion_receipt as validate_teacher_promotion_receipt,
)

PLAN_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-plan.v0.3"
REPORT_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-report.v0.3"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-10m-mix-closure-source-selection-verification.v0.3"
)

EXPECTED_AUTHORIZATION = {
    "bulk_dataset_download_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "official_metadata_lookup_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "lumi_sft_10m_v0_15_mix_closure_update_only",
    "raw_text_free_plan_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "bulk_dataset_payload_bytes": 0,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 120,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.3.json",
    "verification_path": (
        "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.3.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "candidate_mix_reaudit",
        "candidate_mix_reaudit_verification",
        "data_contract",
        "implementation",
        "missing_mix_proposal",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "teacher_promotion_receipt",
        "teacher_promotion_verification",
        "test",
        "verifier",
    }
)

EXPECTED_FILE_SHA256 = {
    "candidate_mix_reaudit": (
        "bc204200a01450cdb522f9b4890be9f4da3e063ae2f3fdbfcae06c32077a88be"
    ),
    "candidate_mix_reaudit_verification": (
        "00b624ffa6a02775de745f88f206e32dda0cba2c5f3a9e6cc2dafb7888feaf4b"
    ),
    "data_contract": (
        "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06"
    ),
    "missing_mix_proposal": (
        "810c39e81ba36160f06c7da8938bcbe5ebf349af333930dc248d197accfc5b22"
    ),
    "predecessor_plan": (
        "672e1faf1e558af7fc6c39afccda3dbe0eafeca77e0b6e6324215a8c2f3bb0c7"
    ),
    "predecessor_report": (
        "3013e3d09ada6dec55911606f5586b59a972027906fb273457afd867e9fad071"
    ),
    "predecessor_verification": (
        "a190c0a7edbfae169b5d6245126919d423159beeba14e193f3c2429a35639148"
    ),
    "source_registry": (
        "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
    ),
    "teacher_promotion_receipt": (
        "1683a001256346afa56501aba5d6d13af0cf84dfa5d1963a2d17c71e568edb5e"
    ),
    "teacher_promotion_verification": (
        "e23cdc9592cc208537a98f63895d3910af67abfa91160409fe7489b200490a2a"
    ),
}
EXPECTED_INTERNAL_SHA256 = {
    "candidate_mix_reaudit": (
        "1249e69b538351453c9c186de18c846eaf616264e94400892e902fa6421cd15f"
    ),
    "candidate_mix_reaudit_verification": (
        "8c1723dd8949fb3592182141a736b0caf3e167437611262f012e92bb6ddd61ec"
    ),
    "predecessor_report": (
        "6f259cf080e376e566b68c1fe54949c2bde982592ee8f70c2fea5e083141909a"
    ),
    "predecessor_verification": (
        "785d50d91005cda91ef600b8b2cd3a521bc853269c4cfb3cb48e7e5ae583d8c1"
    ),
    "teacher_promotion_receipt": (
        "59897eb19ffc27863b15c51f68d2edb5e066b68846c3c3a4621211299932f94c"
    ),
    "teacher_promotion_verification": (
        "c4414d6ed72fdd4d2a109b2d0d7fdc8538d3f857c3a958c62f5c65fd717b01d5"
    ),
}

EXPECTED_CURRENT_TRAIN_TOKENS = 7_256_670
EXPECTED_GAP_TO_10M = 2_743_330
EXPECTED_TASK_GAPS = {
    "coding": 0,
    "conversation": 0,
    "factual_qa": 100_131,
    "multilingual_bengali": 481_837,
    "reasoning_math_science": 0,
    "safety_helpfulness": 789_105,
    "tool_api": 790_186,
    "writing_summarization": 805_413,
}
EXPECTED_LANGUAGE_GAPS = {
    "banglish": 793_248,
    "bengali_script": 474_145,
    "english": 1_081_282,
    "natural_code_switching": 394_655,
}
EXPECTED_MINIMUM_TASK_CLOSURE = 2_966_672
EXPECTED_TARGET_TOKENS = 3_200_000
EXPECTED_PROJECTED_TOKENS = 10_456_670
SOURCE_CAP = 0.35
TARGET_SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_3"
TEACHER_CANARY_SOURCE_ID = "lumi_first_party_teacher_canary_v0_1"

EXPECTED_NEXT_GATE = {
    "action": "provider_bound_first_party_scale_canary",
    "authorization_required": True,
    "bulk_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_or_cloud_gpu_requires_separate_authorization": True,
    "maximum_candidate_records": 1_000,
    "maximum_retained_train_supervised_tokens": 120_000,
    "paid_gpu_required": False,
    "production_sft_authorized": False,
    "provider_decision_required": True,
    "provider_terms_receipt_required": True,
    "required_checks": [
        "exact_provider_model_deployment_terms_prompt_and_revision_binding",
        "truthful_ai_assisted_provenance_and_no_human_authorship_claim",
        "task_and_language_stratified_generation_against_the_3_2m_allocation",
        "factual_grounding_safety_policy_and_tool_result_verification",
        "native_bengali_banglish_code_switch_and_punctuation_quality_review",
        "assistant_and_tool_only_loss_masking_and_chat_template_validation",
        "semantic_train_validation_isolation_exact_and_near_deduplication",
        "benchmark_decontamination_four_k_fit_and_frozen_token_measurement",
        "acceptance_rate_latency_throughput_and_cost_measurement",
        "deterministic_owner_review_and_fail_closed_promotion_thresholds",
    ],
    "target_source_id": TARGET_SOURCE_ID,
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_value(unsigned)


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("v0.15 mix-closure binding set changed")
    return {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _verify_evidence(resolved: Mapping[str, Path]) -> dict[str, Any]:
    reaudit = _load_json(resolved["candidate_mix_reaudit"])
    reaudit_verification = _load_json(resolved["candidate_mix_reaudit_verification"])
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    teacher_receipt = _load_json(resolved["teacher_promotion_receipt"])
    teacher_verification = _load_json(resolved["teacher_promotion_verification"])

    for value, field in (
        (reaudit, "report_sha256"),
        (reaudit_verification, "verification_sha256"),
        (predecessor_report, "report_sha256"),
        (predecessor_verification, "verification_sha256"),
        (teacher_verification, "verification_sha256"),
    ):
        _verify_internal_sha(value, field)
    validate_teacher_promotion_receipt(teacher_receipt)

    observed = {
        "candidate_mix_reaudit": reaudit.get("report_sha256"),
        "candidate_mix_reaudit_verification": reaudit_verification.get(
            "verification_sha256"
        ),
        "predecessor_report": predecessor_report.get("report_sha256"),
        "predecessor_verification": predecessor_verification.get(
            "verification_sha256"
        ),
        "teacher_promotion_receipt": teacher_receipt.get("receipt_sha256"),
        "teacher_promotion_verification": teacher_verification.get(
            "verification_sha256"
        ),
    }
    if observed != EXPECTED_INTERNAL_SHA256:
        raise ValueError("v0.15 mix-closure evidence changed")
    if not all(reaudit_verification.get("checks", {}).values()):
        raise ValueError("v0.15 candidate re-audit verification is not clean")
    if not all(teacher_verification.get("checks", {}).values()):
        raise ValueError("teacher canary promotion verification is not clean")
    return {
        "reaudit": reaudit,
        "predecessor_report": predecessor_report,
        "teacher_receipt": teacher_receipt,
        "teacher_verification": teacher_verification,
    }


def _verify_selected_lane(plan: Mapping[str, Any], predecessor_plan: Mapping[str, Any]) -> None:
    lanes = plan.get("selected_lanes")
    if lanes != predecessor_plan.get("selected_lanes"):
        raise ValueError("v0.15 selected closure lane changed from verified v0.14 decision")
    if not isinstance(lanes, list) or len(lanes) != 1:
        raise ValueError("v0.15 closure must contain exactly one selected lane")
    lane = lanes[0]
    if (
        lane.get("source_id") != TARGET_SOURCE_ID
        or lane.get("target_retained_train_supervised_tokens")
        != EXPECTED_TARGET_TOKENS
        or sum(lane.get("task_allocation", {}).values()) != EXPECTED_TARGET_TOKENS
        or sum(lane.get("language_allocation", {}).values()) != EXPECTED_TARGET_TOKENS
    ):
        raise ValueError("v0.15 selected closure allocation changed")


def load_mix_closure_source_selection_plan_v3(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": "2026-08-09",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_local_evidence_only_v0_15_10m_mix_closure_update",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"v0.15 mix-closure plan changed: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    for name, expected_sha256 in EXPECTED_FILE_SHA256.items():
        if plan["bindings"][name].get("sha256") != expected_sha256:
            raise ValueError(f"v0.15 mix-closure binding changed: {name}")

    predecessor_plan = v2.load_mix_closure_source_selection_plan_v2(
        resolved["predecessor_plan"], repository_root=repository_root
    )
    evidence = _verify_evidence(resolved)
    reaudit = evidence["reaudit"]
    pool = reaudit.get("candidate_pool", {})
    closure = reaudit.get("closure", {})
    if (
        pool.get("accepted_train_supervised_tokens") != EXPECTED_CURRENT_TRAIN_TOKENS
        or pool.get("gap_to_10m_train_supervised_tokens") != EXPECTED_GAP_TO_10M
        or closure.get("task_floor_gaps") != EXPECTED_TASK_GAPS
        or closure.get("language_floor_gaps") != EXPECTED_LANGUAGE_GAPS
    ):
        raise ValueError("v0.15 mix-closure current gap binding changed")
    if sum(EXPECTED_TASK_GAPS.values()) != EXPECTED_MINIMUM_TASK_CLOSURE:
        raise ValueError("v0.15 minimum task closure constant changed")

    minimum = plan.get("minimum_accepted_closure")
    if minimum != {
        "language_floor_gap_total": EXPECTED_GAP_TO_10M,
        "language_floor_gaps": EXPECTED_LANGUAGE_GAPS,
        "task_floor_gap_total": EXPECTED_MINIMUM_TASK_CLOSURE,
        "task_floor_gaps": EXPECTED_TASK_GAPS,
        "task_minus_language_gap_headroom": (
            EXPECTED_MINIMUM_TASK_CLOSURE - EXPECTED_GAP_TO_10M
        ),
    }:
        raise ValueError("v0.15 minimum closure summary changed")
    _verify_selected_lane(plan, predecessor_plan)

    current_sources = set(reaudit.get("source_metrics", {}))
    groups = plan.get("source_family_groups")
    if not isinstance(groups, Mapping):
        raise ValueError("v0.15 source family groups missing")
    flattened = [source_id for source_ids in groups.values() for source_id in source_ids]
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != current_sources | {TARGET_SOURCE_ID}
    ):
        raise ValueError("v0.15 source family groups changed")
    return plan


def rebuild_mix_closure_source_selection_v3(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    evidence = _verify_evidence(resolved)
    reaudit = evidence["reaudit"]
    predecessor_report = evidence["predecessor_report"]
    teacher_receipt = evidence["teacher_receipt"]
    proposal = _load_json(resolved["missing_mix_proposal"])
    task_floors, language_floors = v2._proposal_floors(proposal)
    pool = reaudit["candidate_pool"]
    lanes = [dict(row) for row in plan["selected_lanes"]]

    projected_tasks = v2._sum_allocations(
        pool["train_by_task"], lanes, key="task_allocation"
    )
    projected_languages = v2._sum_allocations(
        pool["train_by_language_mode"], lanes, key="language_allocation"
    )
    current_source_tokens = {
        source_id: metrics["by_split"]["train"]["supervised_tokens"]
        for source_id, metrics in reaudit["source_metrics"].items()
    }
    source_tokens, source_shares, projected_total = v2._project_source_tokens(
        current_source_tokens, lanes
    )
    family_tokens, family_shares = v2._project_family_shares(
        source_tokens,
        plan["source_family_groups"],
        total=projected_total,
    )
    task_gaps = {
        key: max(0, floor - projected_tasks.get(key, 0))
        for key, floor in task_floors.items()
    }
    language_gaps = {
        key: max(0, floor - projected_languages.get(key, 0))
        for key, floor in language_floors.items()
    }
    source_violations = [key for key, share in source_shares.items() if share > SOURCE_CAP]
    family_violations = [key for key, share in family_shares.items() if share > SOURCE_CAP]

    teacher_source = reaudit["source_metrics"][TEACHER_CANARY_SOURCE_ID]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(plan["bindings"]),
        "current_mix": {
            "accepted_train_supervised_tokens": pool["accepted_train_supervised_tokens"],
            "gap_to_10m_train_supervised_tokens": pool[
                "gap_to_10m_train_supervised_tokens"
            ],
            "language_floor_gaps": dict(reaudit["closure"]["language_floor_gaps"]),
            "source_cap_violations": list(
                reaudit["closure"]["source_cap_violations"]
            ),
            "task_floor_gaps": dict(reaudit["closure"]["task_floor_gaps"]),
        },
        "decision_inheritance": {
            "conditional_reserve_count": len(
                predecessor_report["decisions"]["conditional_reserves"]
            ),
            "excluded_source_count": len(
                predecessor_report["decisions"]["exclusions"]
            ),
            "predecessor_decisions_sha256": sha256_value(
                predecessor_report["decisions"]
            ),
            "selected_lane_unchanged": True,
        },
        "minimum_accepted_closure": dict(plan["minimum_accepted_closure"]),
        "network_accounting": {
            "bulk_dataset_payload_bytes": 0,
            "maximum_network_bytes": 0,
            "official_metadata_lookups": 0,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "dataset_payload_downloaded": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "gpu_used": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "projected_mix": {
            "added_train_supervised_tokens": EXPECTED_TARGET_TOKENS,
            "all_language_floors_met": not any(language_gaps.values()),
            "all_source_caps_met": not source_violations,
            "all_source_family_caps_met": not family_violations,
            "all_task_floors_met": not any(task_gaps.values()),
            "buffer_above_minimum_task_closure": (
                EXPECTED_TARGET_TOKENS - EXPECTED_MINIMUM_TASK_CLOSURE
            ),
            "buffer_above_10m_floor": projected_total - 10_000_000,
            "language_floor_gaps": language_gaps,
            "source_cap": SOURCE_CAP,
            "source_cap_violations": source_violations,
            "source_family_cap_violations": family_violations,
            "source_family_shares": family_shares,
            "source_family_train_supervised_tokens": family_tokens,
            "source_shares": source_shares,
            "source_train_supervised_tokens": source_tokens,
            "task_floor_gaps": task_gaps,
            "train_by_language_mode": projected_languages,
            "train_by_task": projected_tasks,
            "train_supervised_tokens": projected_total,
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "provider_bound_1000_row_scale_canary_not_authorized_or_run",
            "first_party_3_2m_generation_and_curation_not_authorized_or_run",
            "candidate_registry_successor_not_materialized",
            "full_mix_cross_source_reaudit_not_complete",
            "frozen_validation_mix_not_sealed",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_gpu_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "selected_lanes": lanes,
        "status": "verified_v0_15_10m_mix_closure_update_no_payload_or_training",
        "teacher_canary_gate": {
            "candidate_file_sha256": teacher_receipt["candidate"]["file_sha256"],
            "cross_source_exact_duplicate_groups": reaudit["exact_duplicate_audit"][
                "cross_source_duplicate_groups"
            ],
            "cross_source_near_duplicate_components": reaudit[
                "near_duplicate_audit"
            ]["cross_source_components"],
            "records": teacher_source["records"],
            "source_id": TEACHER_CANARY_SOURCE_ID,
            "train_records": teacher_source["by_split"]["train"]["records"],
            "train_supervised_tokens": teacher_source["by_split"]["train"][
                "supervised_tokens"
            ],
            "validation_records": teacher_source["by_split"]["validation"][
                "records"
            ],
            "validation_supervised_tokens": teacher_source["by_split"][
                "validation"
            ]["supervised_tokens"],
        },
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_mix_closure_source_selection_report_v3(report)
    return report


def validate_mix_closure_source_selection_report_v3(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("v0.15 mix-closure report schema changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("v0.15 mix-closure report retained raw text")
    if any(report.get("production", {}).values()):
        raise ValueError("v0.15 mix-closure report authorized production")
    projected = report.get("projected_mix", {})
    if (
        projected.get("added_train_supervised_tokens") != EXPECTED_TARGET_TOKENS
        or projected.get("train_supervised_tokens") != EXPECTED_PROJECTED_TOKENS
    ):
        raise ValueError("v0.15 mix-closure projection changed")
    for field in (
        "all_language_floors_met",
        "all_source_caps_met",
        "all_source_family_caps_met",
        "all_task_floors_met",
    ):
        if projected.get(field) is not True:
            raise ValueError(f"v0.15 mix-closure projection failed: {field}")
    if report.get("network_accounting") != {
        "bulk_dataset_payload_bytes": 0,
        "maximum_network_bytes": 0,
        "official_metadata_lookups": 0,
    }:
        raise ValueError("v0.15 mix-closure performed an external action")
    teacher = report.get("teacher_canary_gate", {})
    if (
        teacher.get("records") != 80
        or teacher.get("train_supervised_tokens") != 2_002
        or teacher.get("validation_supervised_tokens") != 799
    ):
        raise ValueError("v0.15 teacher canary proof changed")


def build_mix_closure_source_selection_verification_v3(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_mix_closure_source_selection_report_v3(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_task_and_language_floors_close_with_buffer": True,
            "current_v0_15_reaudit_and_registry_bindings_verified": True,
            "no_payload_network_api_gpu_registry_seal_or_sft_action": True,
            "per_source_and_conservative_family_caps_below_0_35": True,
            "prior_rights_decisions_inherited_by_exact_hash": True,
            "raw_training_text_absent": True,
            "scale_canary_required_before_3_2m_generation": True,
            "teacher_canary_promotion_and_full_mix_dedup_verified": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_train_supervised_tokens": report["projected_mix"][
            "train_supervised_tokens"
        ],
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_v0_15_10m_mix_closure_update_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
