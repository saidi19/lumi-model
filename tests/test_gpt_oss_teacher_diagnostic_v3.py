from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_diagnostic_v3 import (
    ATTEMPT_TOKEN_BUDGETS,
    DIAGNOSTIC_CONTRACT,
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    PRIOR_FAILURE,
    REASONING_EFFORT,
    RECORD_ID,
    RETENTION,
    _response_metadata,
    build_plan_report,
    build_plan_verification,
    diagnostic_manifest,
    diagnostic_payload,
    load_diagnostic_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-diagnostic-v0.3.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-diagnostic-v0.3.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-diagnostic-verification-v0.3.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_diagnostic_is_one_row_and_tightly_bounded() -> None:
    assert LIMITS["exact_diagnostic_records"] == 1
    assert LIMITS["maximum_inference_calls"] == 3
    assert LIMITS["maximum_runtime_retries"] == 2
    assert LIMITS["maximum_network_bytes"] == 0
    assert LIMITS["maximum_generated_tokens_by_attempt"] == [1024, 1536, 2048]
    assert PRIOR_FAILURE["accepted_final_outputs"] == 0


def test_manifest_is_exact_and_attempt_bound() -> None:
    manifest = diagnostic_manifest()
    assert manifest["record_id"] == RECORD_ID
    assert len(manifest["attempts"]) == 3
    assert [item["max_tokens"] for item in manifest["attempts"]] == list(ATTEMPT_TOKEN_BUDGETS)
    assert {item["reasoning_effort"] for item in manifest["attempts"]} == {REASONING_EFFORT}
    assert len({item["seed"] for item in manifest["attempts"]}) == 3
    assert len({item["request_sha256"] for item in manifest["attempts"]}) == 3


def test_payload_keeps_v2_prompt_and_changes_only_diagnostic_controls() -> None:
    payload = diagnostic_payload(2)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert "user_prompt must contain" in text
    assert "assistant_response must contain" in text
    assert "parbo na" in text
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536


def test_response_metadata_is_non_sensitive() -> None:
    response = {
        "choices": [{"finish_reason": "length", "message": {"content": "must-not-leak"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1024, "total_tokens": 1074},
    }
    metadata = _response_metadata(response)
    assert metadata == {
        "finish_reason": "length",
        "usage": {"prompt_tokens": 50, "completion_tokens": 1024, "total_tokens": 1074},
    }
    assert "must-not-leak" not in json.dumps(metadata)
    assert RETENTION["rejected_raw_text"] is False


def test_exact_authorization_forbids_scale_and_download() -> None:
    assert "Run only record " + RECORD_ID in EXACT_AUTHORIZATION_TEXT
    assert "This diagnostic cannot satisfy the 16-cell gate" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port, model or runtime download" in EXACT_AUTHORIZATION_TEXT


def test_diagnostic_plan_and_verification_reproduce() -> None:
    plan = load_diagnostic_plan(PLAN, repository_root=ROOT)
    assert plan["diagnostic_contract"]["id"] == DIAGNOSTIC_CONTRACT
    assert plan["next_gate"]["exact_authorization_text"] == EXACT_AUTHORIZATION_TEXT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
