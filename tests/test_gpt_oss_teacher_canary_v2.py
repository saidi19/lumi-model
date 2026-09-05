from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi import posttraining_gpt_oss_teacher_canary as predecessor
from lumi.posttraining_gpt_oss_teacher_canary_v2 import (
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    MAXIMUM_ATTEMPTS_PER_RECORD,
    PREDECESSOR,
    PROHIBITIONS,
    PROMPT_CONTRACT,
    RUNTIME_REUSE,
    _append_rejection,
    build_successor_report,
    build_successor_verification,
    chat_completion_payload,
    load_successor_plan,
    prompt_manifest,
    successor_seed,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.2.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.2.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-verification-v0.2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_successor_contract_is_zero_network_and_nonexecuting() -> None:
    assert LIMITS["maximum_network_bytes"] == 0
    assert LIMITS["maximum_inference_calls"] == 200
    assert LIMITS["maximum_runtime_retries"] == 40
    assert LIMITS["maximum_attempts_per_record"] == 3
    assert RUNTIME_REUSE["additional_model_or_runtime_downloads"] == 0
    assert PREDECESSOR["accepted_final_outputs"] == 0
    assert all(PROHIBITIONS.values())


def test_attempt_bound_seeds_and_manifest_are_exact() -> None:
    manifest = prompt_manifest()
    assert len(manifest) == 160
    assert len({item["record_id"] for item in manifest}) == 160
    for item in manifest:
        attempts = item["attempts"]
        assert len(attempts) == MAXIMUM_ATTEMPTS_PER_RECORD
        assert len({attempt["seed"] for attempt in attempts}) == 3
        assert len({attempt["request_sha256"] for attempt in attempts}) == 3
        assert [attempt["attempt_number"] for attempt in attempts] == [1, 2, 3]


def test_banglish_repair_payload_is_field_and_label_bound() -> None:
    scenario = predecessor.SAFETY_SCENARIOS[0]
    record_id = "lumi-gptoss-safety_helpfulness-banglish-00"
    payload = chat_completion_payload(
        record_id, scenario, "banglish", attempt_number=1
    )
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert "user_prompt must contain" in text
    assert "assistant_response must contain" in text
    assert "parbo na" in text
    assert "English-only text is forbidden" in text
    assert payload["seed"] == successor_seed(record_id, 1)
    assert payload["seed"] != successor_seed(record_id, 2)


def test_predecessor_parser_accepts_bound_banglish_refusal() -> None:
    scenario = predecessor.SAFETY_SCENARIOS[0]
    final = {
        "assistant_response": (
            "Ami onno karo password dite parbo na. Safe account recovery niye help korte pari."
        ),
        "safety_label": "refuse",
        "scenario_id": scenario.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": "Tumi ki onno karo password dite parba?",
    }
    response = {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(final)}}
        ]
    }
    messages = predecessor._parse_final_response(
        response, scenario=scenario, language="banglish"
    )
    assert messages[0]["content"].startswith("Tumi")
    assert messages[1]["content"].startswith("Ami")


def test_rejection_ledger_retains_hashes_not_raw_text() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rejections.jsonl"
        ledger: list[dict] = []
        _append_rejection(
            path,
            ledger,
            record_id="record",
            attempt_number=1,
            request_sha256="a" * 64,
            response={"private": "must-not-be-retained"},
            error=ValueError("language gate"),
            critical=False,
        )
        stored = path.read_text(encoding="utf-8")
        assert "must-not-be-retained" not in stored
        assert "response_envelope_sha256" in stored
        assert "language gate" in stored


def test_successor_plan_and_verification_reproduce() -> None:
    plan = load_successor_plan(PLAN, repository_root=ROOT)
    assert plan["next_gate"]["exact_authorization_text"] == EXACT_AUTHORIZATION_TEXT
    assert plan["prompt_contract"]["id"] == PROMPT_CONTRACT
    report = build_successor_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert report == _load(REPORT)
    verification = build_successor_verification(
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
