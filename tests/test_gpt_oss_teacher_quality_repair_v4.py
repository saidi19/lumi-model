from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_quality_repair_v4 import (
    BANNED_META_PHRASES,
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    MAX_TOKENS,
    NATURAL_BANGLISH_INSTRUCTION,
    OWNER_REJECTION,
    QUALITY_CONTRACT,
    build_plan_report,
    build_plan_verification,
    load_quality_plan,
    quality_manifest,
    quality_payload,
    validate_naturalness,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-prompt-quality-repair-v0.4.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-prompt-quality-repair-v0.4.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-prompt-quality-repair-verification-v0.4.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_quality_repair_is_one_row_and_bounded() -> None:
    assert LIMITS["exact_repair_records"] == 1
    assert LIMITS["maximum_inference_calls"] == 2
    assert LIMITS["maximum_runtime_retries"] == 1
    assert LIMITS["maximum_network_bytes"] == 0
    assert LIMITS["maximum_generated_tokens_per_call"] == 1536
    assert OWNER_REJECTION["accepted_rows"] == 0


def test_quality_manifest_is_exact_and_distinct() -> None:
    manifest = quality_manifest()
    assert len(manifest["attempts"]) == 2
    assert {item["max_tokens"] for item in manifest["attempts"]} == {MAX_TOKENS}
    assert len({item["seed"] for item in manifest["attempts"]}) == 2
    assert len({item["request_sha256"] for item in manifest["attempts"]}) == 2


def test_quality_payload_contains_naturalness_contract() -> None:
    payload = quality_payload(1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert NATURAL_BANGLISH_INSTRUCTION in text
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536
    assert payload["temperature"] == 0.3


def test_naturalness_gate_accepts_review_target() -> None:
    validate_naturalness([
        {"role": "user", "content": "Tumi ki onno karo login password ber kore dite parba?"},
        {"role": "assistant", "content": "Ami onno karo login password ber kore dite parbo na. Nijer account hole official password reset ba account recovery poddhoti use koro."},
    ])


def test_naturalness_gate_rejects_meta_phrases() -> None:
    for phrase in BANNED_META_PHRASES:
        try:
            validate_naturalness([
                {"role": "user", "content": "Tumi ki help korba?"},
                {"role": "assistant", "content": f"Ami eta korte parbo na. {phrase}. Official password reset use koro."},
            ])
        except ValueError:
            continue
        raise AssertionError(f"banned phrase passed quality gate: {phrase}")


def test_exact_authorization_forbids_scale() -> None:
    assert "at most 2 attempts" in EXACT_AUTHORIZATION_TEXT
    assert "cannot satisfy the 16-cell gate" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port, model or runtime download" in EXACT_AUTHORIZATION_TEXT


def test_quality_plan_and_verification_reproduce() -> None:
    plan = load_quality_plan(PLAN, repository_root=ROOT)
    assert plan["quality_contract"]["id"] == QUALITY_CONTRACT
    assert plan["next_gate"]["exact_authorization_text"] == EXACT_AUTHORIZATION_TEXT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
