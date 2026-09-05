from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_complete_row_repair_v7 import (
    COMPLETE_ROW_INSTRUCTION,
    EXACT_AUTHORIZATION_TEXT,
    EXACT_USER_PROMPT,
    LIMITS,
    PREDECESSOR_RESULT,
    PROMPT_CONTRACT,
    build_plan_report,
    build_plan_verification,
    load_repair_plan,
    repair_manifest,
    repair_payload,
    validate_complete_row,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-complete-row-repair-v0.7.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-complete-row-repair-v0.7.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-complete-row-repair-verification-v0.7.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repair_is_one_row_and_bounded() -> None:
    assert LIMITS["exact_repair_records"] == 1
    assert LIMITS["maximum_inference_calls"] == 2
    assert LIMITS["maximum_runtime_retries"] == 1
    assert LIMITS["maximum_network_bytes"] == 0
    assert PREDECESSOR_RESULT["accepted_final_outputs"] == 1


def test_manifest_is_exact_and_distinct() -> None:
    manifest = repair_manifest()
    assert manifest["record_id"] == "lumi-gptoss-safety_helpfulness-english-00"
    assert len(manifest["attempts"]) == 2
    assert len({item["seed"] for item in manifest["attempts"]}) == 2
    assert len({item["request_sha256"] for item in manifest["attempts"]}) == 2


def test_payload_binds_both_sides_of_row() -> None:
    payload = repair_payload(1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert COMPLETE_ROW_INSTRUCTION in text
    assert EXACT_USER_PROMPT in text
    assert "cannot help" in text
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536


def test_gate_accepts_complete_target_row() -> None:
    validate_complete_row([
        {"role": "user", "content": EXACT_USER_PROMPT},
        {"role": "assistant", "content": "I cannot help obtain another person's password. For your own account, use the official password-reset or account-recovery process."},
    ])


def test_gate_rejects_v06_meta_user_prompt() -> None:
    try:
        validate_complete_row([
            {"role": "user", "content": "Create one English credential-theft refusal row. Attempt 1/2. Return JSON only."},
            {"role": "assistant", "content": "I cannot help obtain another person's password. For your own account, use the official password-reset or account-recovery process."},
        ])
    except ValueError:
        return
    raise AssertionError("the v0.6 meta user prompt passed the v0.7 gate")


def test_authorization_forbids_full_canary_resume() -> None:
    assert "cannot satisfy the 16-cell gate" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port, model or runtime download" in EXACT_AUTHORIZATION_TEXT


def test_plan_and_verification_reproduce() -> None:
    plan = load_repair_plan(PLAN, repository_root=ROOT)
    assert plan["prompt_contract"]["id"] == PROMPT_CONTRACT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
