from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary_v9 import (
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    PREDECESSOR_FAILURE,
    PROMPT_CONTRACT,
    _load_and_validate_prefix,
    build_plan_report,
    build_plan_verification,
    generation_messages,
    load_resume_plan,
    prompt_manifest,
    v8,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-canary-resume-v0.9.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-resume-v0.9.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-resume-verification-v0.9.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prefix_and_remaining_counts_are_exact() -> None:
    rows, telemetry, rejections = _load_and_validate_prefix(ROOT)
    assert len(rows) == 53
    assert len(telemetry) == 53
    assert len(rejections) == 4
    assert len(prompt_manifest()[len(rows):]) == 107


def test_combined_call_and_retry_ceilings_are_exact() -> None:
    assert LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == 200
    assert PREDECESSOR_FAILURE["recovered_rejection_metadata_rows"] + LIMITS["maximum_additional_runtime_retries"] == 40


def test_failed_record_is_first_resume_record() -> None:
    assert prompt_manifest()[53]["record_id"] == PREDECESSOR_FAILURE["failed_record_id"]


def test_failed_record_prompt_hard_binds_all_literals() -> None:
    scenario = [item for item in v8.v5.v2.predecessor.SCENARIOS if item.task == "writing_summarization" and item.variant == 3][0]
    text = generation_messages(scenario, "natural_code_switching", attempt_number=1)[0]["content"]
    for literal in ("14:20", "API gateway", "18 minutes"):
        assert literal in text


def test_authorization_reuses_prefix_and_is_zero_download() -> None:
    assert "Reuse without regeneration the exact 53 accepted prefix rows" in EXACT_AUTHORIZATION_TEXT
    assert "0 additional model or runtime download bytes" in EXACT_AUTHORIZATION_TEXT
    assert "107 additional final outputs" in EXACT_AUTHORIZATION_TEXT


def test_plan_and_verification_reproduce() -> None:
    plan = load_resume_plan(PLAN, repository_root=ROOT)
    assert plan["prompt_contract"]["id"] == PROMPT_CONTRACT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
