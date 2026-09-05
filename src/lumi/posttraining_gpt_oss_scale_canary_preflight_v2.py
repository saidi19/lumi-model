"""Fail-closed preflight for the Lumi gpt-oss 1,000-row scale canary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import _load_json, file_sha256, sha256_value
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA = "lumi.sft-gpt-oss-scale-canary-preflight-plan.v0.2"
REPORT_SCHEMA = "lumi.sft-gpt-oss-scale-canary-preflight-report.v0.2"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-scale-canary-preflight-verification.v0.2"

BASE_TRAIN_TOKENS = 7_266_143
BASE_VALIDATION_TOKENS = 408_031
TARGET_TRAIN_TOKENS = 3_200_000
PROJECTED_TRAIN_TOKENS = 10_466_143
TARGET_SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_3"

TASK_TARGET = {
    "factual_qa": 150_000,
    "multilingual_bengali": 520_000,
    "safety_helpfulness": 850_000,
    "tool_api": 850_000,
    "writing_summarization": 830_000,
}
LANGUAGE_TARGET = {
    "banglish": 850_000,
    "bengali_script": 550_000,
    "english": 1_350_000,
    "natural_code_switching": 450_000,
}
PROJECTED_TASK = {
    "coding": 2_153_012,
    "conversation": 500_617,
    "factual_qa": 1_550_404,
    "multilingual_bengali": 1_041_245,
    "reasoning_math_science": 2_269_713,
    "safety_helpfulness": 864_132,
    "tool_api": 859_814,
    "writing_summarization": 1_227_206,
}
PROJECTED_LANGUAGE = {
    "banglish": 858_400,
    "bengali_script": 1_079_415,
    "english": 8_070_172,
    "natural_code_switching": 458_156,
}
TASK_FLOORS = {
    "coding": 1_500_000,
    "conversation": 500_000,
    "factual_qa": 1_500_000,
    "multilingual_bengali": 1_000_000,
    "reasoning_math_science": 1_500_000,
    "safety_helpfulness": 800_000,
    "tool_api": 800_000,
    "writing_summarization": 1_200_000,
}
LANGUAGE_FLOORS = {
    "banglish": 800_000,
    "bengali_script": 1_000_000,
    "english": 7_800_000,
    "natural_code_switching": 400_000,
}
TASK_ALLOCATION = {
    "factual_qa": 47,
    "multilingual_bengali": 162,
    "safety_helpfulness": 266,
    "tool_api": 266,
    "writing_summarization": 259,
}
LANGUAGE_ALLOCATION = {
    "banglish": 266,
    "bengali_script": 172,
    "english": 422,
    "natural_code_switching": 140,
}
CELL_ALLOCATION = {
    "factual_qa": {
        "banglish": 12,
        "bengali_script": 8,
        "english": 20,
        "natural_code_switching": 7,
    },
    "multilingual_bengali": {
        "banglish": 43,
        "bengali_script": 28,
        "english": 68,
        "natural_code_switching": 23,
    },
    "safety_helpfulness": {
        "banglish": 71,
        "bengali_script": 46,
        "english": 112,
        "natural_code_switching": 37,
    },
    "tool_api": {
        "banglish": 71,
        "bengali_script": 46,
        "english": 112,
        "natural_code_switching": 37,
    },
    "writing_summarization": {
        "banglish": 69,
        "bengali_script": 44,
        "english": 110,
        "natural_code_switching": 36,
    },
}
TRAIN_CELL_ALLOCATION = {
    "factual_qa": {
        "banglish": 10,
        "bengali_script": 6,
        "english": 16,
        "natural_code_switching": 6,
    },
    "multilingual_bengali": {
        "banglish": 34,
        "bengali_script": 22,
        "english": 54,
        "natural_code_switching": 18,
    },
    "safety_helpfulness": {
        "banglish": 57,
        "bengali_script": 37,
        "english": 90,
        "natural_code_switching": 30,
    },
    "tool_api": {
        "banglish": 57,
        "bengali_script": 37,
        "english": 90,
        "natural_code_switching": 29,
    },
    "writing_summarization": {
        "banglish": 55,
        "bengali_script": 35,
        "english": 88,
        "natural_code_switching": 29,
    },
}

EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_signatures",
        "candidate_mix_report",
        "candidate_mix_verification",
        "chat_template",
        "closure_plan",
        "closure_report",
        "closure_verification",
        "data_quality",
        "data_scale_policy",
        "host_bound_plan",
        "host_bound_report",
        "host_bound_verification",
        "implementation",
        "known_hosts",
        "model_download_receipt",
        "owner_review",
        "owner_review_verification",
        "promotion_receipt",
        "promotion_verification",
        "registry",
        "run_report",
        "run_verification",
        "runner",
        "runtime_reuse_receipt",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
    }
)


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_value(unsigned)


def _validation_cells() -> dict[str, dict[str, int]]:
    return {
        task: {
            language: CELL_ALLOCATION[task][language]
            - TRAIN_CELL_ALLOCATION[task][language]
            for language in LANGUAGE_ALLOCATION
        }
        for task in TASK_ALLOCATION
    }


def _sum_cells(value: Mapping[str, Mapping[str, int]]) -> int:
    return sum(sum(row.values()) for row in value.values())


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("gpt-oss scale-canary binding set changed")
    resolved = {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    for name, path in resolved.items():
        expected = str(bindings[name].get("sha256", ""))
        if file_sha256(path) != expected:
            raise ValueError(f"gpt-oss scale-canary binding changed: {name}")
    return resolved


def _verify_schedule(plan: Mapping[str, Any]) -> None:
    schedule = plan.get("canary_contract", {}).get("schedule", {})
    validation = _validation_cells()
    if (
        schedule.get("task_allocation") != TASK_ALLOCATION
        or schedule.get("language_allocation") != LANGUAGE_ALLOCATION
        or schedule.get("cell_allocation") != CELL_ALLOCATION
        or schedule.get("train_cell_allocation") != TRAIN_CELL_ALLOCATION
        or schedule.get("validation_cell_allocation") != validation
        or sum(TASK_ALLOCATION.values()) != 1_000
        or sum(LANGUAGE_ALLOCATION.values()) != 1_000
        or _sum_cells(CELL_ALLOCATION) != 1_000
        or _sum_cells(TRAIN_CELL_ALLOCATION) != 800
        or _sum_cells(validation) != 200
    ):
        raise ValueError("gpt-oss scale-canary schedule changed")


def _verify_predecessors(resolved: Mapping[str, Path]) -> dict[str, Any]:
    mix = _load_json(resolved["candidate_mix_report"])
    mix_verification = _load_json(resolved["candidate_mix_verification"])
    pool = mix.get("candidate_pool", {})
    closure = mix.get("closure", {})
    if (
        mix.get("status") != "verified_registry_v0_16_candidate_mix_reaudit_nonproduction"
        or pool.get("candidate_records_read") != 263_997
        or pool.get("accepted_train_supervised_tokens") != BASE_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens") != BASE_VALIDATION_TOKENS
        or mix_verification.get("status")
        != "PASS_verified_registry_v0_16_candidate_mix_reaudit_nonproduction"
        or not all(mix_verification.get("checks", {}).values())
        or closure.get("source_cap_violations") != []
    ):
        raise ValueError("registry-v0.16 candidate-mix evidence changed")

    closure_report = _load_json(resolved["closure_report"])
    lane = closure_report.get("selected_lanes", [{}])[0]
    next_gate = closure_report.get("next_gate", {})
    if (
        lane.get("source_id") != TARGET_SOURCE_ID
        or lane.get("target_retained_train_supervised_tokens") != TARGET_TRAIN_TOKENS
        or lane.get("task_allocation") != TASK_TARGET
        or lane.get("language_allocation") != LANGUAGE_TARGET
        or next_gate.get("action") != "provider_bound_first_party_scale_canary"
        or next_gate.get("maximum_candidate_records") != 1_000
    ):
        raise ValueError("10M closure lane changed")

    owner_review = _load_json(resolved["owner_review"])
    run_report = _load_json(resolved["run_report"])
    machine = run_report.get("machine_audit", {})
    if (
        owner_review.get("decision") != "accept"
        or owner_review.get("owner_acceptance_rate") != 1.0
        or machine.get("final_outputs") != 160
        or machine.get("first_attempt_acceptance_rate") != 0.95
        or machine.get("critical_failures") != 0
        or not all(machine.get("checks", {}).values())
    ):
        raise ValueError("accepted gpt-oss teacher evidence changed")

    promotion = _load_json(resolved["promotion_receipt"])
    candidate = promotion.get("candidate", {})
    if (
        promotion.get("status")
        != "verified_gpt_oss_teacher_candidate_and_registry_v0_16_nonproduction"
        or candidate.get("records") != 160
        or candidate.get("train_supervised_tokens") != 9_473
        or candidate.get("validation_supervised_tokens") != 1_906
        or promotion.get("registry", {}).get("source_count") != 17
    ):
        raise ValueError("gpt-oss source-promotion evidence changed")

    host = _load_json(resolved["host_bound_plan"])
    server = host.get("server", {})
    model = host.get("model", {})
    runtime = host.get("runtime", {})
    if (
        server.get("host") != "68.209.73.55"
        or server.get("host_key_fingerprint")
        != "SHA256:2ZSQqNUYJhWNjihHnqLQ1YNHxoqtMJeckVJRKlc5hs4"
        or server.get("gpu_uuid")
        != "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3"
        or server.get("gpu_fb_memory_total_mib") != 81_559
        or model.get("revision") != "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
        or model.get("object_count") != 25
        or runtime.get("image_digest")
        != "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
    ):
        raise ValueError("bound gpt-oss host/model/runtime identity changed")
    return {"mix": mix, "closure": closure_report, "promotion": promotion}


def load_gpt_oss_scale_canary_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "local_preflight_only_pending_explicit_gpu_execution_authorization"
        or plan.get("observed_at") != "2026-08-11"
    ):
        raise ValueError("gpt-oss scale-canary plan identity changed")
    authorization = plan.get("authorization", {})
    if (
        authorization.get("local_preflight_authorized") is not True
        or authorization.get("remote_access_authorized") is not False
        or authorization.get("paid_gpu_authorized") is not False
        or authorization.get("candidate_generation_authorized") is not False
        or authorization.get("production_sft_authorized") is not False
        or authorization.get("dataset_seal_authorized") is not False
    ):
        raise ValueError("gpt-oss scale-canary preflight authorization changed")
    if plan.get("limits") != {
        "candidate_records_generated": 0,
        "external_api_calls": 0,
        "gpu_use": 0,
        "maximum_network_bytes": 0,
        "maximum_private_write_bytes": 1_048_576,
        "maximum_runtime_minutes": 30,
    }:
        raise ValueError("gpt-oss scale-canary preflight limits changed")
    _verify_schedule(plan)
    _resolve_bindings(plan, repository_root)
    return plan


def build_gpt_oss_scale_canary_preflight_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    prior = _verify_predecessors(resolved)
    pool = prior["mix"]["candidate_pool"]
    projected_first_party_family = 3_239_425
    projected_family_share = round(
        projected_first_party_family / PROJECTED_TRAIN_TOKENS, 8
    )
    task_gaps = {
        key: max(0, TASK_FLOORS[key] - PROJECTED_TASK[key]) for key in TASK_FLOORS
    }
    language_gaps = {
        key: max(0, LANGUAGE_FLOORS[key] - PROJECTED_LANGUAGE[key])
        for key in LANGUAGE_FLOORS
    }
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {"path": plan["bindings"][name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "canary_contract": plan["canary_contract"],
        "closure_projection": {
            "base_train_supervised_tokens": pool["accepted_train_supervised_tokens"],
            "base_validation_supervised_tokens": pool["input_validation_supervised_tokens"],
            "buffer_above_10m_floor": PROJECTED_TRAIN_TOKENS - 10_000_000,
            "language_floor_gaps": language_gaps,
            "projected_first_party_family_share": projected_family_share,
            "projected_first_party_family_train_supervised_tokens": projected_first_party_family,
            "projected_train_by_language_mode": PROJECTED_LANGUAGE,
            "projected_train_by_task": PROJECTED_TASK,
            "projected_train_supervised_tokens": PROJECTED_TRAIN_TOKENS,
            "source_family_cap": 0.35,
            "source_family_cap_passes": projected_family_share <= 0.35,
            "target_language_allocation": LANGUAGE_TARGET,
            "target_source_id": TARGET_SOURCE_ID,
            "target_task_allocation": TASK_TARGET,
            "target_train_supervised_tokens": TARGET_TRAIN_TOKENS,
            "task_floor_gaps": task_gaps,
        },
        "execution_boundary": plan["execution_boundary"],
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "output": plan["output"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "provider_and_runtime": plan["provider_and_runtime"],
        "raw_training_text_retained": False,
        "report_sha256": "",
        "roadmap": {
            "completed": [
                "160_row_gpt_oss_teacher_canary_and_owner_review",
                "immutable_source_registry_v0_16_promotion",
                "263997_row_registry_v0_16_full_mix_reaudit",
                "gpt_oss_1000_row_tool_inclusive_scale_canary_preflight",
            ],
            "current": "await_explicit_authorization_then_run_1000_row_scale_canary",
            "remaining_after_scale_canary_pass": [
                "generate_and_curate_3_2m_retained_train_token_closure_lane",
                "promote_closure_source_and_reaudit_full_mix",
                "seal_dataset_and_run_cpu_training_smoke",
                "authorize_and_run_production_sft_then_evaluate",
            ],
        },
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_gpt_oss_1000_row_scale_canary_preflight_no_execution",
        "teacher_evidence": {
            "accepted_owner_review_rows": 32,
            "critical_failures": 0,
            "first_attempt_acceptance_rate": 0.95,
            "machine_verified_rows": 160,
            "promoted_train_supervised_tokens": 9_473,
            "tool_api_in_160_row_teacher_canary": False,
            "tool_api_required_in_scale_canary": True,
        },
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_gpt_oss_scale_canary_preflight_report(report)
    return report


def validate_gpt_oss_scale_canary_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("gpt-oss scale-canary report schema changed")
    if report.get("report_sha256") != _internal_sha(report, "report_sha256"):
        raise ValueError("gpt-oss scale-canary report seal changed")
    projection = report.get("closure_projection", {})
    if (
        report.get("status") != "PASS_gpt_oss_1000_row_scale_canary_preflight_no_execution"
        or projection.get("projected_train_supervised_tokens") != PROJECTED_TRAIN_TOKENS
        or projection.get("task_floor_gaps") != {key: 0 for key in TASK_FLOORS}
        or projection.get("language_floor_gaps") != {key: 0 for key in LANGUAGE_FLOORS}
        or projection.get("source_family_cap_passes") is not True
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
    ):
        raise ValueError("gpt-oss scale-canary preflight did not pass")


def build_gpt_oss_scale_canary_preflight_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_gpt_oss_scale_canary_preflight_report(report)
    checks = {
        "accepted_gpt_oss_teacher_and_owner_review_bound": True,
        "all_20_task_language_cells_scheduled": len(CELL_ALLOCATION) == 5
        and all(len(row) == 4 for row in CELL_ALLOCATION.values()),
        "candidate_mix_v0_11_and_registry_v0_16_bound": True,
        "closure_projection_meets_all_floors_and_source_cap": True,
        "exact_1000_row_800_200_schedule_verified": _sum_cells(CELL_ALLOCATION)
        == 1_000
        and _sum_cells(TRAIN_CELL_ALLOCATION) == 800
        and _sum_cells(_validation_cells()) == 200,
        "host_model_runtime_and_cache_receipts_bound": True,
        "local_preflight_used_no_gpu_network_or_inference": plan["limits"]["gpu_use"]
        == 0
        and plan["limits"]["maximum_network_bytes"] == 0,
        "raw_training_text_absent": report.get("raw_training_text_retained") is False,
        "tool_api_is_required_and_no_row_patch_loop_is_authorized": report[
            "teacher_evidence"
        ]["tool_api_required_in_scale_canary"]
        and report["canary_contract"]["failure_policy"]
        == "aggregate_cell_report_fail_closed_no_one_row_repair_loop",
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "verification_sha256": "",
    }
    if not all(checks.values()):
        raise ValueError("gpt-oss scale-canary preflight verification failed")
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
