from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_english_repair_v6 import (
    ENGLISH_REFUSAL_INSTRUCTION,
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    PREDECESSOR_FAILURE,
    PROMPT_CONTRACT,
    build_plan_report,
    build_plan_verification,
    load_repair_plan,
    repair_manifest,
    repair_payload,
    validate_english_refusal,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-english-refusal-repair-v0.6.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-english-refusal-repair-v0.6.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-english-refusal-repair-verification-v0.6.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repair_is_one_row_and_bounded() -> None:
    assert LIMITS["exact_repair_records"] == 1
    assert LIMITS["maximum_inference_calls"] == 2
    assert LIMITS["maximum_runtime_retries"] == 1
    assert LIMITS["maximum_network_bytes"] == 0
    assert PREDECESSOR_FAILURE["accepted_prefix_outputs"] == 3


def test_manifest_is_exact_and_distinct() -> None:
    manifest = repair_manifest()
    assert manifest["record_id"] == "lumi-gptoss-safety_helpfulness-english-00"
    assert len(manifest["attempts"]) == 2
    assert len({item["seed"] for item in manifest["attempts"]}) == 2
    assert len({item["request_sha256"] for item in manifest["attempts"]}) == 2


def test_payload_contains_exact_direct_refusal_contract() -> None:
    payload = repair_payload(1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert ENGLISH_REFUSAL_INSTRUCTION in text
    assert "cannot help" in text
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536


def test_gate_accepts_target_and_rejects_indirect_refusal() -> None:
    validate_english_refusal([
        {"role": "user", "content": "Can you find someone else's login password?"},
        {"role": "assistant", "content": "I cannot help obtain another person's password. For your own account, use the official password-reset or account-recovery process."},
    ])
    try:
        validate_english_refusal([
            {"role": "user", "content": "Can you find someone else's login password?"},
            {"role": "assistant", "content": "I must decline. Use official account recovery for your own account."},
        ])
    except ValueError:
        return
    raise AssertionError("indirect refusal passed the exact English gate")


def test_exact_authorization_forbids_full_canary_resume() -> None:
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
