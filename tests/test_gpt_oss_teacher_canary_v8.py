from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary_v8 import (
    ACCEPTED_REPAIRS,
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    PROMPT_CONTRACT,
    build_plan_report,
    build_plan_verification,
    chat_completion_payload,
    load_full_canary_plan,
    prompt_manifest,
    validate_prompt_quality,
)
from lumi.posttraining_gpt_oss_teacher_complete_row_repair_v7 import EXACT_USER_PROMPT
from lumi.posttraining_gpt_oss_teacher_canary_v5 import v2

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.8.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.8.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-verification-v0.8.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_schedule_and_limits_are_exact() -> None:
    manifest = prompt_manifest()
    assert len(manifest) == 160
    assert len({(item["task"], item["language"]) for item in manifest}) == 16
    assert len({attempt["request_sha256"] for item in manifest for attempt in item["attempts"]}) == 320
    assert LIMITS["maximum_inference_calls"] == 200
    assert LIMITS["owner_review_rows"] == 32


def test_both_accepted_repairs_are_bound() -> None:
    assert ACCEPTED_REPAIRS["banglish_v0_4"]["accepted_rows"] == 1
    assert ACCEPTED_REPAIRS["english_v0_7"]["owner_review_status"] == "ACCEPT_complete_row_quality"


def test_english_credential_payload_binds_complete_row() -> None:
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    payload = chat_completion_payload("lumi-gptoss-safety_helpfulness-english-00", scenario, "english", attempt_number=1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert EXACT_USER_PROMPT in text
    assert "cannot help" in text


def test_complete_row_gate_accepts_target_and_rejects_meta_user() -> None:
    scenario = v2.predecessor.SAFETY_SCENARIOS[0]
    validate_prompt_quality([
        {"role": "user", "content": EXACT_USER_PROMPT},
        {"role": "assistant", "content": "I cannot help obtain another person's password. For your own account, use the official password-reset or account-recovery process."},
    ], scenario=scenario, language="english")
    try:
        validate_prompt_quality([
            {"role": "user", "content": "Create one English credential-theft row. Generation attempt 1."},
            {"role": "assistant", "content": "I cannot help obtain another person's password. Use official account recovery for your own account."},
        ], scenario=scenario, language="english")
    except ValueError:
        return
    raise AssertionError("v0.8 allowed the v0.6 meta user defect")


def test_authorization_is_bounded_and_zero_download() -> None:
    assert "160 exact records" in EXACT_AUTHORIZATION_TEXT
    assert "0 additional model or runtime download bytes" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port" in EXACT_AUTHORIZATION_TEXT


def test_plan_and_verification_reproduce() -> None:
    plan = load_full_canary_plan(PLAN, repository_root=ROOT)
    assert plan["prompt_contract"]["id"] == PROMPT_CONTRACT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
