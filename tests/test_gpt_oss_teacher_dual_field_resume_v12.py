from __future__ import annotations

from pathlib import Path

from lumi.posttraining_gpt_oss_teacher_dual_field_resume_v12 import (
    ACCEPTED_REPAIRS,
    LIMITS,
    _load_and_validate_prefix,
    _patched_v8,
    chat_completion_payload,
    generation_messages,
    prompt_manifest,
    v8,
)

ROOT = Path(__file__).resolve().parents[1]


def test_accepted_prefix_and_remaining_counts_are_exact() -> None:
    rows, telemetry, rejections = _load_and_validate_prefix(ROOT)
    assert len(rows) == 55
    assert len(telemetry) == 55
    assert len(rejections) == 4
    assert sum(int(item["attempts"]) for item in telemetry) == 59
    assert len(prompt_manifest()[55:]) == 105


def test_first_remaining_record_is_exact() -> None:
    assert prompt_manifest()[55]["record_id"] == "lumi-gptoss-writing_summarization-english-03"


def test_combined_call_and_retry_arithmetic_is_exact() -> None:
    assert LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == 200
    assert 4 + LIMITS["maximum_additional_runtime_retries"] == 40


def test_dual_field_instruction_repeats_each_literal() -> None:
    scenario = next(item for item in v8.v5.v2.predecessor.SCENARIOS if item.task == "writing_summarization" and item.variant == 4)
    text = generation_messages(scenario, "banglish", attempt_number=1)[0]["content"]
    assert "user_prompt must separately contain" in text
    assert "assistant_response must also separately contain" in text
    for literal in scenario.required_literals:
        assert text.count(literal) >= 2


def test_patch_context_payload_does_not_recurse() -> None:
    item = prompt_manifest()[55]
    scenario = next(value for value in v8.v5.v2.predecessor.SCENARIOS if value.task == item["task"] and value.variant == item["variant"])
    original = v8.chat_completion_payload
    with _patched_v8():
        payload = v8.chat_completion_payload(item["record_id"], scenario, item["language"], attempt_number=1)
        assert payload == chat_completion_payload(item["record_id"], scenario, item["language"], attempt_number=1)
    assert v8.chat_completion_payload is original


def test_v011_owner_acceptance_is_bound() -> None:
    repair = ACCEPTED_REPAIRS["bengali_dual_field_v0_11"]
    assert repair["owner_review_status"] == "ACCEPT_dual_field_literal_quality"
    assert repair["accepted_rows"] == 1
