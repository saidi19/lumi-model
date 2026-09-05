from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lumi import posttraining_gpt_oss_teacher_canary as predecessor
from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary_v5 import (
    ACCEPTED_REPAIR,
    BANGLISH_INSTRUCTION,
    EXACT_AUTHORIZATION_TEXT,
    LIMITS,
    MAXIMUM_ATTEMPTS_PER_RECORD,
    PROMPT_CONTRACT,
    PROMPT_QUALITY_GATES,
    _append_rejection,
    build_plan_report,
    build_plan_verification,
    chat_completion_payload,
    full_canary_seed,
    load_full_canary_plan,
    prompt_manifest,
    validate_prompt_quality,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.5.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-v0.5.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-gpt-oss-teacher-canary-successor-verification-v0.5.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_canary_is_exact_and_bounded() -> None:
    assert LIMITS["exact_prompt_cells"] == 16
    assert LIMITS["exact_final_outputs"] == 160
    assert LIMITS["maximum_inference_calls"] == 200
    assert LIMITS["maximum_runtime_retries"] == 40
    assert LIMITS["maximum_network_bytes"] == 0
    assert LIMITS["maximum_generated_tokens_per_call"] == 1536
    assert MAXIMUM_ATTEMPTS_PER_RECORD == 2
    assert ACCEPTED_REPAIR["accepted_rows"] == 1


def test_manifest_has_two_distinct_attempts_per_record() -> None:
    manifest = prompt_manifest()
    assert len(manifest) == 160
    assert len({item["record_id"] for item in manifest}) == 160
    assert len({(item["task"], item["language"]) for item in manifest}) == 16
    for item in manifest:
        assert len(item["attempts"]) == 2
        assert len({attempt["seed"] for attempt in item["attempts"]}) == 2
        assert len({attempt["request_sha256"] for attempt in item["attempts"]}) == 2


def test_banglish_payload_carries_accepted_quality_contract() -> None:
    scenario = predecessor.SAFETY_SCENARIOS[0]
    record_id = "lumi-gptoss-safety_helpfulness-banglish-00"
    payload = chat_completion_payload(record_id, scenario, "banglish", attempt_number=1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert BANGLISH_INSTRUCTION in text
    assert "parbo na" in text
    assert "official password reset" in text
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536
    assert payload["seed"] == full_canary_seed(record_id, 1)
    assert payload["seed"] != full_canary_seed(record_id, 2)


def test_quality_gate_accepts_v0_4_owner_row() -> None:
    validate_prompt_quality(
        [
            {"role": "user", "content": "Tumi onno karo login password ber kore dite parba?"},
            {"role": "assistant", "content": "Ami onno karo login password ber kore dite parbo na. Nijer account thakle official password reset ba account recovery poddhoti use koro."},
        ],
        scenario=predecessor.SAFETY_SCENARIOS[0],
        language="banglish",
    )


def test_quality_gate_rejects_meta_phrase() -> None:
    try:
        validate_prompt_quality(
            [
                {"role": "user", "content": "Tumi ki help korba?"},
                {"role": "assistant", "content": "Ami eta korte parbo na. refuse the harmful request. Official password reset use koro."},
            ],
            scenario=predecessor.SAFETY_SCENARIOS[0],
            language="banglish",
        )
    except ValueError:
        return
    raise AssertionError("banned meta phrase passed the v0.5 quality gate")


def test_rejection_retains_metadata_not_raw_candidate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rejections.jsonl"
        ledger: list[dict] = []
        _append_rejection(
            path,
            ledger,
            record_id="record",
            attempt_number=1,
            request_sha256="a" * 64,
            response={"private": "must-not-be-retained", "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
            error=ValueError("naturalness gate"),
            critical=False,
        )
        stored = path.read_text(encoding="utf-8")
        assert "must-not-be-retained" not in stored
        assert "response_envelope_sha256" in stored
        assert "completion_tokens" in stored


def test_exact_authorization_keeps_execution_separate() -> None:
    assert "160 exact records" in EXACT_AUTHORIZATION_TEXT
    assert "deterministic 32-row owner-review packet" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port, model or runtime download" in EXACT_AUTHORIZATION_TEXT
    assert PROMPT_QUALITY_GATES["owner_review_required"] is True


def test_full_canary_plan_and_verification_reproduce() -> None:
    plan = load_full_canary_plan(PLAN, repository_root=ROOT)
    assert plan["prompt_contract"]["id"] == PROMPT_CONTRACT
    assert plan["next_gate"]["exact_authorization_text"] == EXACT_AUTHORIZATION_TEXT
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT)
    assert report == _load(REPORT)
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(REPORT))
    assert verification == _load(VERIFICATION)
    assert verification["status"] == "PASS"
