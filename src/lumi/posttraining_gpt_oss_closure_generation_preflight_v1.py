"""Bound the 3.2M-token gpt-oss closure lane without remote execution."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value

ARTIFACT_ROOT = (
    "artifacts/lumi-gpt-oss-deterministic-tool-repair-v0.6/"
    "79db468b8c61/attempt-001"
)
CANONICAL_CANDIDATE_PATH = f"{ARTIFACT_ROOT}/private/canonical-candidate-v0.17.jsonl"
TELEMETRY_PATH = f"{ARTIFACT_ROOT}/private/generation-telemetry.jsonl"
RUN_REPORT_PATH = f"{ARTIFACT_ROOT}/evidence/run-report.json"
RUN_VERIFICATION_PATH = f"{ARTIFACT_ROOT}/evidence/run-verification.json"
OWNER_REVIEW_PATH = f"{ARTIFACT_ROOT}/evidence/owner-review.json"
OWNER_VERIFICATION_PATH = f"{ARTIFACT_ROOT}/evidence/owner-review-verification.json"

PLAN_SCHEMA = "lumi.sft-gpt-oss-closure-generation-preflight-plan.v0.1"
REPORT_SCHEMA = "lumi.sft-gpt-oss-closure-generation-preflight-report.v0.1"
VERIFICATION_SCHEMA = (
    "lumi.sft-gpt-oss-closure-generation-preflight-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_only_3p2m_closure_generation_projection",
    "remote_access_authorized": False,
    "s3_action_authorized": False,
    "termination_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_local_runtime_minutes": 30,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_remote_actions": 0,
}
EXPECTED_OUTPUT = {
    "preflight_report_path": (
        "evidence/lumi-4b-sft-gpt-oss-closure-generation-preflight-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-gpt-oss-closure-generation-preflight-"
        "verification-v0.1.json"
    ),
}

EXPECTED_BINDINGS = frozenset(
    {
        "canonical_candidate",
        "candidate_mix_plan",
        "candidate_mix_report",
        "candidate_mix_verification",
        "implementation",
        "owner_review",
        "owner_review_verification",
        "promotion_receipt",
        "promotion_verification",
        "run_plan",
        "run_report",
        "run_verification",
        "runner",
        "source_registry",
        "telemetry",
        "test",
        "verifier",
    }
)

TASK_TARGETS = {
    "factual_qa": 150_000,
    "multilingual_bengali": 520_000,
    "safety_helpfulness": 850_000,
    "tool_api": 850_000,
    "writing_summarization": 830_000,
}
LANGUAGE_TARGETS = {
    "banglish": 850_000,
    "bengali_script": 550_000,
    "english": 1_350_000,
    "natural_code_switching": 450_000,
}
CELL_TARGETS = {
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
SCHEDULE_SPEC = {
    "cell_targets": CELL_TARGETS,
    "checkpoint_retained_train_token_shards": [
        400_000,
        800_000,
        1_200_000,
        1_600_000,
        2_000_000,
        2_400_000,
        2_800_000,
        3_200_000,
    ],
    "contract": "lumi.gpt-oss-teacher.closure-3p2m.v0.1",
    "language_targets": LANGUAGE_TARGETS,
    "maximum_supervised_tokens_per_accepted_train_row": 512,
    "minimum_supervised_tokens_per_accepted_train_row": 64,
    "schedule_seed": 3_200_017,
    "target_retained_train_supervised_tokens": 3_200_000,
    "task_targets": TASK_TARGETS,
    "validation_scenario_family_fraction": 0.05,
}
EXPECTED_SCHEDULE_SPEC_SHA256 = (
    "8ce8e97d39ada5271f5e4eb6057b5fb31b4aa1018f58347b66a14451ed9a4bdd"
)

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
EXPECTED_CALIBRATION = {
    "acceptance_rate": 0.984,
    "accepted_rows": 984,
    "deterministic_rows": 216,
    "inference_attempts": 803,
    "model_rows": 768,
    "observed_final_attempt_latency_seconds": {
        "maximum": 11.033109,
        "mean": 6.190187,
        "median": 5.88448,
        "sum": 4_754.063918,
    },
    "retry_calls": 35,
    "train_rows": 787,
    "train_supervised_tokens": 75_439,
    "train_tokens_by_cell": {
        "factual_qa": {
            "banglish": {"records": 10, "supervised_tokens": 1_069},
            "bengali_script": {"records": 6, "supervised_tokens": 756},
            "english": {"records": 16, "supervised_tokens": 1_020},
            "natural_code_switching": {"records": 6, "supervised_tokens": 857},
        },
        "multilingual_bengali": {
            "banglish": {"records": 34, "supervised_tokens": 4_172},
            "bengali_script": {"records": 19, "supervised_tokens": 4_334},
            "english": {"records": 54, "supervised_tokens": 4_014},
            "natural_code_switching": {"records": 18, "supervised_tokens": 3_477},
        },
        "safety_helpfulness": {
            "banglish": {"records": 53, "supervised_tokens": 5_434},
            "bengali_script": {"records": 35, "supervised_tokens": 6_376},
            "english": {"records": 90, "supervised_tokens": 5_503},
            "natural_code_switching": {"records": 27, "supervised_tokens": 4_810},
        },
        "tool_api": {
            "banglish": {"records": 57, "supervised_tokens": 5_140},
            "bengali_script": {"records": 37, "supervised_tokens": 5_398},
            "english": {"records": 90, "supervised_tokens": 5_958},
            "natural_code_switching": {"records": 29, "supervised_tokens": 3_025},
        },
        "writing_summarization": {
            "banglish": {"records": 55, "supervised_tokens": 3_675},
            "bengali_script": {"records": 35, "supervised_tokens": 3_373},
            "english": {"records": 88, "supervised_tokens": 4_090},
            "natural_code_switching": {"records": 28, "supervised_tokens": 2_958},
        },
    },
}
EXPECTED_PROJECTED_ROWS_BY_CELL = {
    "factual_qa": {
        "banglish": 375,
        "bengali_script": 207,
        "english": 989,
        "natural_code_switching": 148,
    },
    "multilingual_bengali": {
        "banglish": 1_125,
        "bengali_script": 391,
        "english": 2_947,
        "natural_code_switching": 384,
    },
    "safety_helpfulness": {
        "banglish": 2_205,
        "bengali_script": 802,
        "english": 5_872,
        "natural_code_switching": 668,
    },
    "tool_api": {
        "banglish": 2_507,
        "bengali_script": 1_001,
        "english": 5_423,
        "natural_code_switching": 1_141,
    },
    "writing_summarization": {
        "banglish": 3_293,
        "bengali_script": 1_484,
        "english": 7_531,
        "natural_code_switching": 1_108,
    },
}

EXPECTED_REMOTE_BOUNDS = {
    "checkpoint_shards": 8,
    "expected_accepted_train_rows": 39_601,
    "expected_accepted_validation_rows": 2_085,
    "expected_inference_calls": 32_476,
    "expected_instance_minutes_with_overhead": 3_914,
    "expected_usd_before_tax_at_4_29_per_hour": 279.83,
    "maximum_accepted_outputs": 53_000,
    "maximum_additional_network_bytes": 1_073_741_824,
    "maximum_additional_private_write_bytes": 1_073_741_824,
    "maximum_inference_calls": 45_000,
    "maximum_instance_minutes": 5_760,
    "maximum_retry_calls": 6_000,
    "maximum_usd_before_tax_at_4_29_per_hour": 411.84,
    "owner_review_rows": 200,
}

_MODE = re.compile(
    r"-(banglish|bengali_script|english|natural_code_switching)-[0-9]{4}$"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at line {number}: {path}")
        values.append(value)
    return values


def resolve_binding(root: Path, specification: Mapping[str, Any], *, name: str) -> Path:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid closure preflight binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing closure preflight binding: {name}")
    if file_sha256(path) != expected:
        raise ValueError(f"changed closure preflight binding: {name}")
    return path


def _calibration(rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_id = {str(row["record_id"]): row for row in rows}
    if len(row_by_id) != 984 or len(telemetry) != 984:
        raise ValueError("closure calibration inventory changed")
    cells: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    model_latencies: list[float] = []
    train_rows = 0
    train_tokens = 0
    for item in telemetry:
        record_id = str(item["record_id"])
        row = row_by_id.get(record_id)
        if row is None:
            raise ValueError("closure calibration telemetry row changed")
        latency = float(item["elapsed_seconds_final_attempt"])
        if latency > 0:
            model_latencies.append(latency)
        if row["split"] != "train":
            continue
        match = _MODE.search(record_id)
        if match is None:
            raise ValueError("closure calibration language mode changed")
        task = str(row["task_category"])
        mode = match.group(1)
        tokens = int(item["token_counts"]["supervised_tokens"])
        cells[task][mode]["records"] += 1
        cells[task][mode]["supervised_tokens"] += tokens
        train_rows += 1
        train_tokens += tokens
    ordered_cells = {
        task: {mode: dict(cells[task][mode]) for mode in LANGUAGE_TARGETS}
        for task in TASK_TARGETS
    }
    sorted_latency = sorted(model_latencies)
    midpoint = len(sorted_latency) // 2
    median = (sorted_latency[midpoint - 1] + sorted_latency[midpoint]) / 2
    result = {
        "acceptance_rate": 0.984,
        "accepted_rows": len(telemetry),
        "deterministic_rows": sum(
            1 for item in telemetry if item["accepted_on_attempt"] == 0
        ),
        "inference_attempts": sum(int(item["attempts"]) for item in telemetry),
        "model_rows": len(model_latencies),
        "observed_final_attempt_latency_seconds": {
            "maximum": round(max(model_latencies), 6),
            "mean": round(sum(model_latencies) / len(model_latencies), 6),
            "median": round(median, 6),
            "sum": round(sum(model_latencies), 6),
        },
        "retry_calls": sum(int(item["attempts"]) for item in telemetry)
        - len(model_latencies),
        "train_rows": train_rows,
        "train_supervised_tokens": train_tokens,
        "train_tokens_by_cell": ordered_cells,
    }
    if result != EXPECTED_CALIBRATION:
        raise ValueError("closure calibration changed")
    return result


def _projected_rows(calibration: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    projected: dict[str, dict[str, int]] = {}
    for task, modes in CELL_TARGETS.items():
        projected[task] = {}
        for mode, target in modes.items():
            observed = calibration["train_tokens_by_cell"][task][mode]
            projected[task][mode] = math.ceil(
                target * observed["records"] / observed["supervised_tokens"]
            )
    if projected != EXPECTED_PROJECTED_ROWS_BY_CELL:
        raise ValueError("closure projected row counts changed")
    return projected


def load_preflight_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _read_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-12",
        "output": EXPECTED_OUTPUT,
        "remote_bounds_if_authorized": EXPECTED_REMOTE_BOUNDS,
        "schedule_spec": SCHEDULE_SPEC,
        "schema_version": PLAN_SCHEMA,
        "status": "local_projection_only_no_remote_action",
    }
    for name, value in expected.items():
        if plan.get(name) != value:
            raise ValueError(f"closure generation preflight changed: {name}")
    if sha256_value(plan["schedule_spec"]) != EXPECTED_SCHEDULE_SPEC_SHA256:
        raise ValueError("closure schedule specification changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("closure preflight binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid closure preflight binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    next_gate = plan.get("next_gate")
    if not isinstance(next_gate, Mapping) or not next_gate.get("authorization_required"):
        raise ValueError("closure generation next gate changed")
    return plan


def build_preflight_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: Path
) -> dict[str, Any]:
    bindings = plan["bindings"]
    resolved = {
        name: resolve_binding(repository_root, spec, name=name)
        for name, spec in bindings.items()
    }
    mix_report = _read_json(resolved["candidate_mix_report"])
    mix_verification = _read_json(resolved["candidate_mix_verification"])
    _verify_internal_sha(mix_report, "report_sha256")
    _verify_internal_sha(mix_verification, "verification_sha256")
    closure = mix_report.get("closure", {})
    pool = mix_report.get("candidate_pool", {})
    if (
        mix_report.get("status")
        != "verified_registry_v0_17_candidate_mix_reaudit_nonproduction"
        or mix_verification.get("status")
        != "PASS_verified_registry_v0_17_candidate_mix_reaudit_nonproduction"
        or pool.get("accepted_records_after_generic_quarantine") != 264_981
        or pool.get("accepted_train_supervised_tokens") != 7_341_582
        or pool.get("gap_to_10m_train_supervised_tokens") != 2_658_418
        or closure.get("task_floor_gaps") != EXPECTED_TASK_GAPS
        or closure.get("language_floor_gaps") != EXPECTED_LANGUAGE_GAPS
    ):
        raise ValueError("closure preflight candidate-mix evidence changed")
    run_report = _read_json(resolved["run_report"])
    run_verification = _read_json(resolved["run_verification"])
    owner = _read_json(resolved["owner_review"])
    owner_verification = _read_json(resolved["owner_review_verification"])
    promotion = _read_json(resolved["promotion_receipt"])
    promotion_verification = _read_json(resolved["promotion_verification"])
    if (
        run_report.get("machine_audit", {}).get("accepted_records") != 984
        or run_verification.get("status") != "PASS"
        or owner.get("status") != "ACCEPT_deterministic_100_row_owner_review_packet"
        or owner_verification.get("status") != "PASS_owner_acceptance_verified"
        or promotion.get("production_eligible") is not False
        or promotion_verification.get("status")
        != "PASS_verified_gpt_oss_scale_candidate_and_registry_v0_17_nonproduction"
    ):
        raise ValueError("closure preflight scale evidence changed")
    rows = _read_jsonl(resolved["canonical_candidate"])
    telemetry = _read_jsonl(resolved["telemetry"])
    calibration = _calibration(rows, telemetry)
    projected_rows = _projected_rows(calibration)
    task_gap_total = sum(EXPECTED_TASK_GAPS.values())
    language_gap_total = sum(EXPECTED_LANGUAGE_GAPS.values())
    projected_train_rows = sum(
        value for modes in projected_rows.values() for value in modes.values()
    )
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": {
            name: {"path": bindings[name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "calibration": calibration,
        "checks": {
            "all_local_bindings_verified": True,
            "candidate_mix_v0_12_is_clean": True,
            "gpu_network_remote_and_s3_actions_absent": True,
            "owner_accepted_scale_canary_verified": True,
            "projected_source_share_below_0_35": True,
            "raw_training_text_absent": True,
            "schedule_closes_every_task_and_language_floor": True,
            "schedule_has_318240_task_gap_headroom": True,
        },
        "closure": {
            "current_train_supervised_tokens": 7_341_582,
            "language_floor_gap_total": language_gap_total,
            "language_floor_gaps": dict(EXPECTED_LANGUAGE_GAPS),
            "projected_new_source_share": round(3_200_000 / 10_541_582, 8),
            "projected_post_closure_train_supervised_tokens": 10_541_582,
            "selected_retained_train_token_target": 3_200_000,
            "task_floor_gap_total": task_gap_total,
            "task_floor_gaps": dict(EXPECTED_TASK_GAPS),
            "task_gap_headroom": 3_200_000 - task_gap_total,
        },
        "next_gate": copy.deepcopy(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "pricing": {
            "currency": "USD",
            "lambda_1x_h100_sxm_hourly_before_tax": 4.29,
            "official_source": "https://lambda.ai/pricing",
            "verified_at": "2026-08-12",
        },
        "production": {
            "candidate_generated": False,
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_started": False,
        },
        "projection": {
            "expected_accepted_train_rows": projected_train_rows,
            "expected_accepted_validation_rows": 2_085,
            "expected_inference_calls": 32_476,
            "expected_instance_minutes_with_overhead": 3_914,
            "expected_usd_before_tax": 279.83,
            "projected_rows_by_cell": projected_rows,
            "remote_bounds": dict(EXPECTED_REMOTE_BOUNDS),
        },
        "schedule_spec_sha256": EXPECTED_SCHEDULE_SPEC_SHA256,
        "schema_version": REPORT_SCHEMA,
        "status": "verified_local_3p2m_closure_projection_no_remote_action",
    }
    report["report_sha256"] = sha256_value(report)
    validate_preflight_report(report)
    return report


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unsupported closure preflight report")
    _verify_internal_sha(report, "report_sha256")
    if report.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("closure preflight authorization changed")
    checks = report.get("checks")
    production = report.get("production")
    if not isinstance(checks, Mapping) or not all(checks.values()):
        raise ValueError("closure preflight checks failed")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("closure preflight crossed a production boundary")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
        if forbidden in serialized:
            raise ValueError("closure preflight retained raw training text")


def build_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    validate_preflight_report(report)
    validate_preflight_report(replay)
    if report != replay:
        raise ValueError("closure preflight replay changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_bindings_reverified": True,
            "budget_and_cost_projection_reproduced": True,
            "candidate_mix_gaps_reproduced": True,
            "no_remote_gpu_network_s3_registry_seal_or_sft_action": True,
            "raw_training_text_absent": True,
            "schedule_matrix_and_hash_reproduced": True,
            "scale_canary_calibration_reproduced": True,
        },
        "observed_at": plan["observed_at"],
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_internal_sha256": report["report_sha256"],
        "schedule_spec_sha256": EXPECTED_SCHEDULE_SPEC_SHA256,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS_verified_local_3p2m_closure_projection_no_remote_action",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_REMOTE_BOUNDS",
    "EXPECTED_SCHEDULE_SPEC_SHA256",
    "SCHEDULE_SPEC",
    "build_preflight_report",
    "build_preflight_verification",
    "load_preflight_plan",
    "validate_preflight_report",
]
