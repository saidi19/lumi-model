"""Registry-v0.17-bound plan for closing the Lumi 4B 10M SFT mix.

This module is deliberately local-only and raw-text-free.  It reconciles the
latest verified candidate mix, proves that the retained 3.2M-token allocation
still closes every frozen task/language floor, and records the authorization
boundary before any new provider or generation work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_mix_closure_source_selection_v2 as v2
from . import posttraining_mix_closure_source_selection_v3 as v3
from .posttraining_candidate_mix_preflight import (
    _load_json,
    _verify_internal_sha,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-plan.v0.4"
REPORT_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-report.v0.4"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-10m-mix-closure-source-selection-verification.v0.4"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "generation_authorized": False,
    "local_raw_text_free_reconciliation_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "registry_v0_17_10m_mix_closure_reconciliation_only",
    "remote_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.4.json",
    "verification_path": (
        "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.4.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "candidate_mix_reaudit",
        "candidate_mix_reaudit_verification",
        "closure_retirement",
        "closure_retirement_verification",
        "data_contract",
        "implementation",
        "missing_mix_proposal",
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)

EXPECTED_FILE_SHA256 = {
    "candidate_mix_reaudit": (
        "2389e61c66fa0000690fe119745e963b9cc2cab1eb26e61224bef87ef9522dcb"
    ),
    "candidate_mix_reaudit_verification": (
        "d12d28e192cf31e33d3c034293868a720fd74d9934ae506c0e1d5c2d1d3c18d2"
    ),
    "closure_retirement": (
        "369ed092ce7a4cfc8d13fde4d5fa0750166c59fe4a9102a5848eb65cd4667deb"
    ),
    "closure_retirement_verification": (
        "fac9637ca42d6afb1d71d405fe785d4ee6efa797b53ac37d76370f1455a33812"
    ),
    "data_contract": (
        "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06"
    ),
    "missing_mix_proposal": (
        "810c39e81ba36160f06c7da8938bcbe5ebf349af333930dc248d197accfc5b22"
    ),
    "predecessor_plan": (
        "d95b61d68a2f3491deae537f8370c17665490ae71c50c893f4470d081cee0fad"
    ),
    "predecessor_report": (
        "629b1115feda72fe9c32a38eb18bb5b868b885c0e20ad801406a5e38c90f66f1"
    ),
    "predecessor_verification": (
        "9933a0e5900709ed81900de7b040a93685252dccc856d3c24ead0cbe372ed4b0"
    ),
    "source_registry": (
        "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
    ),
}
EXPECTED_INTERNAL_SHA256 = {
    "candidate_mix_reaudit": (
        "f41fcf1cddcf14f071e8bee9b06477232829b961f5ef57d20c5f4c4af8e80de2"
    ),
    "candidate_mix_reaudit_verification": (
        "71aeaef05d2797f73869215a7bab63b6737211d8f0164d1b363b87648e93d52f"
    ),
    "predecessor_report": (
        "54d53dda2caa62ee552e1453b7b2df21a1f2148e787d47237ae27ea045c720a5"
    ),
    "predecessor_verification": (
        "5e7dc7632935414cb352c703ff56b8ea70fe5f825c6c7d36813f78a0ca8f9337"
    ),
}

EXPECTED_CURRENT_TRAIN_TOKENS = 7_341_582
EXPECTED_GAP_TO_10M = 2_658_418
EXPECTED_TASK_GAPS = {
    "coding": 0,
    "conversation": 0,
    "factual_qa": 95_894,
    "multilingual_bengali": 462_758,
    "reasoning_math_science": 0,
    "safety_helpfulness": 763_745,
    "tool_api": 770_665,
    "writing_summarization": 788_698,
}
EXPECTED_LANGUAGE_GAPS = {
    "banglish": 772_110,
    "bengali_script": 450_348,
    "english": 1_059_243,
    "natural_code_switching": 376_717,
}
EXPECTED_MINIMUM_TASK_CLOSURE = 2_881_760
EXPECTED_TARGET_TOKENS = 3_200_000
EXPECTED_PROJECTED_TOKENS = 10_541_582
EXPECTED_CURRENT_FIRST_PARTY_TOKENS = 114_864
EXPECTED_PROJECTED_FIRST_PARTY_TOKENS = 3_314_864
SOURCE_CAP = 0.35
TARGET_SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_7"
RETIRED_CANDIDATE_SHA256 = (
    "e8c0a360c898e6c8f78701c27f534b49f8bbe43c557bec9d15f0ec2c4eb74b1e"
)

EXPECTED_TASK_ALLOCATION = {
    "factual_qa": 150_000,
    "multilingual_bengali": 520_000,
    "safety_helpfulness": 850_000,
    "tool_api": 850_000,
    "writing_summarization": 830_000,
}
EXPECTED_LANGUAGE_ALLOCATION = {
    "banglish": 850_000,
    "bengali_script": 550_000,
    "english": 1_350_000,
    "natural_code_switching": 450_000,
}
EXPECTED_CELL_ALLOCATION = {
    "factual_qa": {
        "banglish": 40_000,
        "bengali_script": 26_000,
        "english": 63_000,
        "natural_code_switching": 21_000,
    },
    "multilingual_bengali": {
        "banglish": 138_000,
        "bengali_script": 89_000,
        "english": 219_000,
        "natural_code_switching": 74_000,
    },
    "safety_helpfulness": {
        "banglish": 226_000,
        "bengali_script": 146_000,
        "english": 359_000,
        "natural_code_switching": 119_000,
    },
    "tool_api": {
        "banglish": 226_000,
        "bengali_script": 146_000,
        "english": 359_000,
        "natural_code_switching": 119_000,
    },
    "writing_summarization": {
        "banglish": 220_000,
        "bengali_script": 143_000,
        "english": 350_000,
        "natural_code_switching": 117_000,
    },
}

EXPECTED_NEXT_GATE = {
    "action": "provider_and_terms_bound_100_row_response_only_canary",
    "authorization_required": True,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "generation_authorized": False,
    "legacy_gpt_oss_host_or_candidate_reuse_permitted": False,
    "local_qwen_reinstall_permitted": False,
    "maximum_candidate_records": 100,
    "maximum_inference_calls": 120,
    "maximum_retry_calls": 20,
    "maximum_retained_train_supervised_tokens": 12_000,
    "minimax_cloud_use_authorized": False,
    "network_or_provider_use_requires_separate_authorization": True,
    "production_sft_authorized": False,
    "provider_decision_required": True,
    "provider_terms_receipt_required": True,
    "required_checks": [
        "exact_provider_model_deployment_terms_endpoint_and_revision_binding",
        "response_only_generation_with_frozen_prompts_and_metadata",
        "deterministic_tool_result_rendering_without_model_generated_tool_results",
        "truthful_ai_assisted_provenance_and_no_human_authorship_claim",
        "five_rows_per_task_language_cell",
        "factual_grounding_safety_timed_plan_and_tool_result_verification",
        "native_bengali_banglish_code_switch_and_punctuation_quality_review",
        "assistant_and_tool_only_loss_masking_and_chat_template_validation",
        "semantic_train_validation_isolation_exact_and_near_deduplication",
        "benchmark_decontamination_four_k_fit_and_frozen_token_measurement",
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
        raise ValueError("registry-v0.17 closure binding set changed")
    return {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _verify_evidence(resolved: Mapping[str, Path]) -> dict[str, Any]:
    reaudit = _load_json(resolved["candidate_mix_reaudit"])
    reaudit_verification = _load_json(resolved["candidate_mix_reaudit_verification"])
    predecessor_report = _load_json(resolved["predecessor_report"])
    predecessor_verification = _load_json(resolved["predecessor_verification"])
    retirement = _load_json(resolved["closure_retirement"])
    retirement_verification = _load_json(resolved["closure_retirement_verification"])
    registry = _load_json(resolved["source_registry"])

    for value, field in (
        (reaudit, "report_sha256"),
        (reaudit_verification, "verification_sha256"),
        (predecessor_report, "report_sha256"),
        (predecessor_verification, "verification_sha256"),
    ):
        _verify_internal_sha(value, field)
    v3.validate_mix_closure_source_selection_report_v3(predecessor_report)
    observed = {
        "candidate_mix_reaudit": reaudit.get("report_sha256"),
        "candidate_mix_reaudit_verification": reaudit_verification.get(
            "verification_sha256"
        ),
        "predecessor_report": predecessor_report.get("report_sha256"),
        "predecessor_verification": predecessor_verification.get(
            "verification_sha256"
        ),
    }
    if observed != EXPECTED_INTERNAL_SHA256:
        raise ValueError("registry-v0.17 closure evidence changed")
    if not all(reaudit_verification.get("checks", {}).values()):
        raise ValueError("candidate-mix v0.12 verification is not clean")
    if not all(retirement_verification.get("checks", {}).values()):
        raise ValueError("closure retirement verification is not clean")
    if (
        retirement.get("status")
        != "PASS_local_evidence_preserved_remote_termination_ready"
        or retirement.get("closure_generation", {}).get("status")
        != "retired_after_fail_closed_transport_instability"
        or retirement.get("local_evidence", {}).get("candidate", {}).get("sha256")
        != RETIRED_CANDIDATE_SHA256
        or retirement.get("local_ollama_cleanup", {}).get(
            "remaining_local_model_bytes"
        )
        != 0
        or retirement.get("local_ollama_cleanup", {}).get("minimax_cloud_touched")
        is not False
    ):
        raise ValueError("legacy closure retirement evidence changed")
    registered_source_ids = {
        source.get("source_id")
        for source in registry.get("sources", [])
        if isinstance(source, Mapping)
    }
    reaudit_candidate_hashes = {
        metrics.get("candidate_file_sha256")
        for metrics in reaudit.get("source_metrics", {}).values()
        if isinstance(metrics, Mapping)
    }
    if (
        len(registered_source_ids) != 18
        or RETIRED_CANDIDATE_SHA256 in reaudit_candidate_hashes
    ):
        raise ValueError("retired closure candidate unexpectedly entered registry-v0.17")
    return {
        "reaudit": reaudit,
        "predecessor_report": predecessor_report,
        "retirement": retirement,
    }


def _verify_selected_lane(plan: Mapping[str, Any]) -> None:
    lanes = plan.get("selected_lanes")
    if not isinstance(lanes, list) or len(lanes) != 1:
        raise ValueError("registry-v0.17 closure must contain one selected lane")
    lane = lanes[0]
    if (
        lane.get("source_id") != TARGET_SOURCE_ID
        or lane.get("target_retained_train_supervised_tokens")
        != EXPECTED_TARGET_TOKENS
        or lane.get("task_allocation") != EXPECTED_TASK_ALLOCATION
        or lane.get("language_allocation") != EXPECTED_LANGUAGE_ALLOCATION
        or lane.get("task_language_cell_allocation") != EXPECTED_CELL_ALLOCATION
    ):
        raise ValueError("registry-v0.17 closure allocation changed")
    cell_task_totals = {
        task: sum(by_language.values())
        for task, by_language in EXPECTED_CELL_ALLOCATION.items()
    }
    cell_language_totals = {
        language: sum(
            by_language[language]
            for by_language in EXPECTED_CELL_ALLOCATION.values()
        )
        for language in EXPECTED_LANGUAGE_ALLOCATION
    }
    if (
        cell_task_totals != EXPECTED_TASK_ALLOCATION
        or cell_language_totals != EXPECTED_LANGUAGE_ALLOCATION
    ):
        raise ValueError("closure cell allocation does not reconcile")


def load_mix_closure_source_selection_plan_v4(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": "2026-08-13",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_local_registry_v0_17_closure_reconciliation_only",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"registry-v0.17 closure plan changed: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    for name, expected_sha256 in EXPECTED_FILE_SHA256.items():
        if plan["bindings"][name].get("sha256") != expected_sha256:
            raise ValueError(f"registry-v0.17 closure binding changed: {name}")
    v3.load_mix_closure_source_selection_plan_v3(
        resolved["predecessor_plan"], repository_root=repository_root
    )
    evidence = _verify_evidence(resolved)
    reaudit = evidence["reaudit"]
    pool = reaudit.get("candidate_pool", {})
    closure = reaudit.get("closure", {})
    if (
        pool.get("accepted_train_supervised_tokens")
        != EXPECTED_CURRENT_TRAIN_TOKENS
        or pool.get("gap_to_10m_train_supervised_tokens") != EXPECTED_GAP_TO_10M
        or closure.get("task_floor_gaps") != EXPECTED_TASK_GAPS
        or closure.get("language_floor_gaps") != EXPECTED_LANGUAGE_GAPS
        or sum(EXPECTED_TASK_GAPS.values()) != EXPECTED_MINIMUM_TASK_CLOSURE
    ):
        raise ValueError("registry-v0.17 current gap binding changed")
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
        raise ValueError("registry-v0.17 minimum closure summary changed")
    _verify_selected_lane(plan)
    current_sources = set(reaudit.get("source_metrics", {}))
    groups = plan.get("source_family_groups")
    if not isinstance(groups, Mapping):
        raise ValueError("registry-v0.17 source family groups missing")
    flattened = [source_id for source_ids in groups.values() for source_id in source_ids]
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != current_sources | {TARGET_SOURCE_ID}
    ):
        raise ValueError("registry-v0.17 source family groups changed")
    return plan


def rebuild_mix_closure_source_selection_v4(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    evidence = _verify_evidence(resolved)
    reaudit = evidence["reaudit"]
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

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(plan["bindings"]),
        "current_mix": {
            "accepted_train_supervised_tokens": pool["accepted_train_supervised_tokens"],
            "gap_to_10m_train_supervised_tokens": pool[
                "gap_to_10m_train_supervised_tokens"
            ],
            "language_floor_gaps": dict(reaudit["closure"]["language_floor_gaps"]),
            "source_cap_violations": list(reaudit["closure"]["source_cap_violations"]),
            "task_floor_gaps": dict(reaudit["closure"]["task_floor_gaps"]),
        },
        "legacy_closure_retirement": {
            "candidate_excluded_from_registry_v0_17": True,
            "candidate_sha256": RETIRED_CANDIDATE_SHA256,
            "local_qwen_model_bytes": 0,
            "minimax_cloud_qualified_or_used": False,
            "retirement_evidence_observed_at": evidence["retirement"]["timestamp_utc"],
            "retirement_status": evidence["retirement"]["closure_generation"]["status"],
        },
        "minimum_accepted_closure": dict(plan["minimum_accepted_closure"]),
        "network_accounting": {
            "external_calls": 0,
            "maximum_network_bytes": 0,
            "provider_metadata_lookups": 0,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "generation_started": False,
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
            "buffer_above_10m_floor": projected_total - 10_000_000,
            "buffer_above_minimum_task_closure": (
                EXPECTED_TARGET_TOKENS - EXPECTED_MINIMUM_TASK_CLOSURE
            ),
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
            "teacher_provider_and_output_terms_not_selected_or_bound",
            "provider_bound_100_row_response_only_canary_not_authorized_or_run",
            "first_party_3_2m_generation_and_curation_not_authorized_or_run",
            "candidate_registry_v0_18_not_materialized",
            "full_mix_cross_source_reaudit_not_complete",
            "production_dataset_seal_and_encoded_shards_not_materialized",
            "active_production_data_contract_not_updated",
            "seal_bound_cpu_canary_gpu_smoke_lr_screen_and_full_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "selected_lanes": lanes,
        "status": "verified_registry_v0_17_10m_closure_reconciliation_nonproduction",
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_mix_closure_source_selection_report_v4(report)
    return report


def validate_mix_closure_source_selection_report_v4(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("registry-v0.17 closure report schema changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("registry-v0.17 closure report retained raw text")
    if any(report.get("production", {}).values()):
        raise ValueError("registry-v0.17 closure report performed an unauthorized action")
    projected = report.get("projected_mix", {})
    if (
        projected.get("added_train_supervised_tokens") != EXPECTED_TARGET_TOKENS
        or projected.get("train_supervised_tokens") != EXPECTED_PROJECTED_TOKENS
        or projected.get("source_family_train_supervised_tokens", {}).get(
            "lumi_first_party_family"
        )
        != EXPECTED_PROJECTED_FIRST_PARTY_TOKENS
    ):
        raise ValueError("registry-v0.17 closure projection changed")
    for field in (
        "all_language_floors_met",
        "all_source_caps_met",
        "all_source_family_caps_met",
        "all_task_floors_met",
    ):
        if projected.get(field) is not True:
            raise ValueError(f"registry-v0.17 closure projection failed: {field}")
    if report.get("network_accounting") != {
        "external_calls": 0,
        "maximum_network_bytes": 0,
        "provider_metadata_lookups": 0,
    }:
        raise ValueError("registry-v0.17 closure used network or a provider")
    retirement = report.get("legacy_closure_retirement", {})
    if (
        retirement.get("candidate_excluded_from_registry_v0_17") is not True
        or retirement.get("candidate_sha256") != RETIRED_CANDIDATE_SHA256
        or retirement.get("local_qwen_model_bytes") != 0
        or retirement.get("minimax_cloud_qualified_or_used") is not False
    ):
        raise ValueError("legacy closure exclusion changed")


def build_mix_closure_source_selection_verification_v4(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_mix_closure_source_selection_report_v4(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_task_and_language_floors_close_with_buffer": True,
            "current_registry_v0_17_reaudit_and_registry_bindings_verified": True,
            "legacy_gpt_oss_candidate_excluded_and_local_qwen_absent": True,
            "no_generation_network_api_gpu_registry_seal_or_sft_action": True,
            "per_source_and_conservative_family_caps_below_0_35": True,
            "provider_and_terms_bound_canary_required_before_generation": True,
            "raw_training_text_absent": True,
            "task_language_cell_allocation_reconciles_exactly": True,
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
        "status": "PASS_registry_v0_17_10m_closure_reconciliation_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
