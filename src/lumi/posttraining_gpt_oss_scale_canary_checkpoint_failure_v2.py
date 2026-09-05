"""Finalize a mathematically ineligible gpt-oss scale-canary checkpoint."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_gpt_oss_scale_canary_v2 as canary
from .posttraining_candidate_mix_preflight import file_sha256

REPORT_SCHEMA = "lumi.sft-gpt-oss-scale-canary-checkpoint-failure-report.v0.2"
VERIFICATION_SCHEMA = (
    "lumi.sft-gpt-oss-scale-canary-checkpoint-failure-verification.v0.2"
)
TARGET_ACCEPTED = 850

BOUND_FILES = {
    "authorization.json": "536800f101e943b3e72d7d4bdfa4a1457f18367c66e4d0786adae69a332e4a63",
    "runtime-resume-receipt.json": (
        "54a85c58f2a0f668f763c465e042993ec3111566dd1de230bc671427f2c0a3d3"
    ),
    "private/attempt-metadata.jsonl": (
        "0bcc57a0f36601ea9794320b0f0a7c7275da14285cedf600c924ac0e1893ec6a"
    ),
    "private/candidate-rows.partial.jsonl": (
        "0c81cec793e04fc1574ff3403fc50c1a037c4b7423ca73ac7167ac92953ce605"
    ),
    "private/generation-telemetry.partial.jsonl": (
        "ef18c82d609494d6689804bd4b51d4696b112a9ca180beab9611cdb0c481721b"
    ),
}

BOUND_RUNTIME_OBSERVATION = {
    "container": {
        "exit_code": 0,
        "finished_at": "2026-08-11T11:44:40.975535698Z",
        "id": canary.CONTAINER_ID,
        "oom_killed": False,
        "running": False,
        "status": "exited",
    },
    "gpu": {
        "memory_total_mib": 81_559,
        "memory_used_mib": 0,
        "name": "NVIDIA H100 80GB HBM3",
        "utilization_percent": 0,
        "uuid": "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3",
    },
    "host": "68.209.73.55",
    "observed_at_utc": "2026-08-11T11:51:54Z",
    "observation_mode": "pinned_ssh_read_only",
}


def _sha256_value(value: Mapping[str, Any]) -> str:
    return canary.scale.sha256_value(value)


def _assert_bound_files(artifact_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in BOUND_FILES.items():
        path = artifact_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        if digest != expected_sha256:
            raise ValueError(f"checkpoint binding changed: {relative_path}")
        observed[relative_path] = digest
    return observed


def _cell_summary(
    schedule: Sequence[canary.scale.Slot],
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int | float]]:
    attempted_ids = {str(item["record_id"]) for item in attempts}
    accepted_ids = {str(row["record_id"]) for row in rows}
    cells: dict[str, dict[str, int | float]] = {}
    tasks = sorted({slot.task for slot in schedule})
    languages = sorted({slot.language_mode for slot in schedule})
    for task in tasks:
        for language in languages:
            ids = {
                slot.record_id
                for slot in schedule
                if slot.task == task and slot.language_mode == language
            }
            attempted = len(ids & attempted_ids)
            accepted = len(ids & accepted_ids)
            remaining = len(ids) - attempted
            maximum = accepted + remaining
            cells[f"{task}/{language}"] = {
                "accepted": accepted,
                "attempted": attempted,
                "failed_final": attempted - accepted,
                "maximum_final_accepted": maximum,
                "maximum_final_rate": round(maximum / len(ids), 6),
                "remaining": remaining,
                "scheduled": len(ids),
            }
    return cells


def build_report(artifact_root: str | Path) -> dict[str, Any]:
    artifact = Path(artifact_root).resolve()
    bindings = _assert_bound_files(artifact)
    schedule = canary.build_schedule()
    rows = canary._read_jsonl(artifact / "private/candidate-rows.partial.jsonl")
    telemetry = canary._read_jsonl(
        artifact / "private/generation-telemetry.partial.jsonl"
    )
    attempts = canary._read_jsonl(artifact / "private/attempt-metadata.jsonl")
    canary._validate_partial_state(schedule, rows, telemetry, attempts)

    attempted_ids = {str(item["record_id"]) for item in attempts}
    accepted_ids = {str(row["record_id"]) for row in rows}
    retry_calls = sum(1 for item in attempts if item.get("attempt_number") == 2)
    remaining = len(schedule) - len(attempted_ids)
    maximum_final_accepted = len(accepted_ids) + remaining
    rejection_counts = Counter(
        str(item.get("validation_error", "unknown"))
        for item in attempts
        if item.get("accepted") is not True
    )
    cells = _cell_summary(schedule, rows, attempts)
    impossible_cells = sorted(
        key for key, value in cells.items() if value["maximum_final_rate"] < 0.75
    )
    checks = {
        "accepted_and_telemetry_membership_match": {
            str(row["record_id"]) for row in rows
        }
        == {str(item["record_id"]) for item in telemetry},
        "all_checkpoint_bindings_match": bindings == BOUND_FILES,
        "container_stopped": BOUND_RUNTIME_OBSERVATION["container"]["running"]
        is False,
        "gpu_idle": BOUND_RUNTIME_OBSERVATION["gpu"]["memory_used_mib"] == 0
        and BOUND_RUNTIME_OBSERVATION["gpu"]["utilization_percent"] == 0,
        "maximum_possible_below_850": maximum_final_accepted < TARGET_ACCEPTED,
        "retry_ceiling_reached": retry_calls == canary.MAX_RETRIES,
        "schedule_exactly_1000": len(schedule) == 1_000,
    }
    if not all(checks.values()):
        raise ValueError("checkpoint failure evidence did not reproduce")

    report: dict[str, Any] = {
        "accepted_rows": len(accepted_ids),
        "attempted_records": len(attempted_ids),
        "bindings": bindings,
        "calls": len(attempts),
        "cell_summary": cells,
        "checks": checks,
        "failed_final_records": len(attempted_ids) - len(accepted_ids),
        "failure_policy": "aggregate_cell_report_fail_closed_no_one_row_repair_loop",
        "impossible_75_percent_cells": impossible_cells,
        "maximum_possible_final_accepted": maximum_final_accepted,
        "maximum_possible_final_rate": round(
            maximum_final_accepted / len(schedule), 6
        ),
        "production_eligible": False,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "remaining_records": remaining,
        "retry_calls": retry_calls,
        "runtime_observation": BOUND_RUNTIME_OBSERVATION,
        "schema_version": REPORT_SCHEMA,
        "status": "FAIL_CLOSED_checkpoint_mathematically_ineligible",
        "target_accepted": TARGET_ACCEPTED,
    }
    report["report_sha256"] = _sha256_value(report)
    return report


def finalize_checkpoint(artifact_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = Path(artifact_root).resolve()
    evidence = artifact / "evidence"
    report_path = evidence / "checkpoint-aggregate-failure-report.json"
    verification_path = evidence / "checkpoint-aggregate-failure-verification.json"
    if report_path.exists() or verification_path.exists():
        raise FileExistsError("refusing to overwrite checkpoint failure evidence")
    report = build_report(artifact)
    canary._write_json_atomic(report_path, report)

    reproduced = build_report(artifact)
    loaded = canary._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256")
    serialized_report = json.dumps(report, ensure_ascii=False)
    checks = {
        "independent_report_reproduced": reproduced == report,
        "internal_report_hash": internal == _sha256_value(unsigned),
        "raw_candidate_text_absent": all(
            marker not in serialized_report
            for marker in ('"messages"', '"user_prompt"', '"assistant_response"')
        ),
        "report_file_round_trip": loaded == report,
        "status_fail_closed": report["status"]
        == "FAIL_CLOSED_checkpoint_mathematically_ineligible",
    }
    verification: dict[str, Any] = {
        "checks": checks,
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = _sha256_value(verification)
    canary._write_json_atomic(verification_path, verification)
    if verification["status"] != "PASS":
        raise ValueError("checkpoint failure verification failed")
    return report, verification
