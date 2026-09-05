from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary import (
    EXACT_AUTHORIZATION_TEXT,
    FACTUAL_SCENARIOS,
    LIMITS,
    PROHIBITIONS,
    QUALITY_GATES,
    SERVER,
    _machine_audit,
    _parse_final_response,
    build_host_bound_report,
    build_host_bound_verification,
    expected_record_ids,
    load_host_bound_plan,
    prompt_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-canary-host-bound-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-host-bound-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-gpt-oss-teacher-canary-host-bound-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_host_gpu_memory_and_prohibitions_are_exact() -> None:
    assert SERVER["gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert SERVER["gpu_uuid"] == "GPU-c7e68b23-95b4-37bd-b40b-c41a7ec716f3"
    assert SERVER["gpu_nominal_memory_gb"] == 80
    assert SERVER["gpu_fb_memory_total_mib"] < SERVER["legacy_preflight_minimum_mib"]
    assert SERVER["legacy_numeric_gate_passed"] is False
    assert (
        SERVER["gpu_fb_memory_total_mib"]
        >= SERVER["successor_minimum_reported_fb_mib"]
    )
    assert all(PROHIBITIONS.values())


def test_prompt_manifest_is_exactly_16_cells_and_160_outputs() -> None:
    manifest = prompt_manifest()
    assert len(manifest) == LIMITS["exact_final_outputs"] == 160
    assert len(expected_record_ids()) == 160
    assert len({item["record_id"] for item in manifest}) == 160
    cells = {(item["task"], item["language"]) for item in manifest}
    assert len(cells) == LIMITS["exact_prompt_cells"] == 16
    assert all(
        sum(
            item["task"] == task and item["language"] == language
            for item in manifest
        )
        == LIMITS["prompts_per_cell"]
        for task, language in cells
    )


def test_host_bound_plan_is_nonexecuting_and_reproduces() -> None:
    plan = load_host_bound_plan(PLAN, repository_root=ROOT)
    assert not any(
        value for value in plan["authorization"].values() if isinstance(value, bool)
    )
    assert plan["next_gate"]["exact_authorization_text"] == EXACT_AUTHORIZATION_TEXT
    rebuilt = build_host_bound_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == _load(REPORT)


def test_host_bound_verification_reproduces() -> None:
    report = _load(REPORT)
    expected = build_host_bound_verification(
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    assert expected["status"] == "PASS"
    assert QUALITY_GATES["critical_factual_safety_or_tool_failures"] == 0


def test_machine_audit_passes_only_the_exact_160_record_membership() -> None:
    manifest = prompt_manifest()
    rows = [
        {
            "language_mode": item["language"],
            "record_id": item["record_id"],
            "task": item["task"],
        }
        for item in manifest
    ]
    telemetry = [
        {
            "accepted_on_attempt": 1,
            "attempts": 1,
            "critical_failures": 0,
            "record_id": item["record_id"],
        }
        for item in manifest
    ]
    audit = _machine_audit(rows, telemetry)
    assert all(audit["checks"].values())
    assert audit["final_outputs"] == 160
    assert audit["inference_calls"] == 160
    assert audit["runtime_retries"] == 0


def test_response_parser_discards_reasoning_and_validates_final_json() -> None:
    scenario = FACTUAL_SCENARIOS[0]
    final = {
        "assistant_response": "There are exactly 2500 metres in 2.5 kilometres.",
        "safety_label": "safe",
        "scenario_id": scenario.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": "How many metres are in 2.5 kilometres?",
    }
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(final),
                    "reasoning_content": "must never be retained",
                },
            }
        ]
    }
    messages = _parse_final_response(response, scenario=scenario, language="english")
    assert messages == [
        {"role": "user", "content": final["user_prompt"]},
        {"role": "assistant", "content": final["assistant_response"]},
    ]
    assert "reasoning_content" not in json.dumps(messages)
