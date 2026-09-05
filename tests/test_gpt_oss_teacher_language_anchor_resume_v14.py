from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_gpt_oss_teacher_language_anchor_resume_v14 import (
    ACCEPTED_REPAIRS,
    LANGUAGE_FIELD_ANCHORS,
    LIMITS,
    _load_and_validate_prefix,
    _patched_v8,
    chat_completion_payload,
    generation_messages,
    prompt_manifest,
    v8,
    validate_prompt_quality,
)

ROOT = Path(__file__).resolve().parents[1]


def _scenario(variant: int):
    return next(item for item in v8.v5.v2.predecessor.SCENARIOS if item.task == "factual_qa" and item.variant == variant)


def test_accepted_prefix_and_remaining_counts_are_exact() -> None:
    rows, telemetry, rejections = _load_and_validate_prefix(ROOT)
    assert len(rows) == 143
    assert len(telemetry) == 143
    assert len(rejections) == 9
    assert sum(int(item["attempts"]) for item in telemetry) == 152
    assert len(prompt_manifest()[143:]) == 17


def test_first_remaining_record_is_exact() -> None:
    assert prompt_manifest()[143]["record_id"] == "lumi-gptoss-factual_qa-english-05"


def test_combined_call_and_retry_arithmetic_is_exact() -> None:
    assert LIMITS["predecessor_retained_prefix_attempts"] + LIMITS["maximum_additional_inference_calls"] == 186
    assert 9 + LIMITS["maximum_additional_runtime_retries"] == 26


@pytest.mark.parametrize("language", tuple(LANGUAGE_FIELD_ANCHORS))
def test_language_anchor_instruction_is_field_specific(language: str) -> None:
    user_anchor, assistant_anchor = LANGUAGE_FIELD_ANCHORS[language]
    text = generation_messages(_scenario(6), language, attempt_number=1)[0]["content"]
    assert user_anchor in text
    assert assistant_anchor in text
    assert "180" in text
    assert "assigned field" in text


@pytest.mark.parametrize(
    ("language", "user", "assistant"),
    (
        ("english", "Please tell me the triangle angle sum.", "The answer is 180 degrees."),
        ("banglish", "Triangle-er angle sum eta ki?", "uttor hobe 180 degree."),
        ("natural_code_switching", "এই question-এ triangle-এর angle sum কত?", "সঠিক answer হলো 180 degrees."),
        ("bengali_script", "দয়া করে বলুন, ত্রিভুজের কোণের সমষ্টি কত?", "উত্তর হলো 180 ডিগ্রি।"),
    ),
)
def test_language_anchor_reference_rows_pass(language: str, user: str, assistant: str) -> None:
    validate_prompt_quality(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        scenario=_scenario(6),
        language=language,
    )


def test_missing_assistant_anchor_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing exact english anchor"):
        validate_prompt_quality(
            [
                {"role": "user", "content": "Please tell me the triangle angle sum."},
                {"role": "assistant", "content": "It is 180 degrees."},
            ],
            scenario=_scenario(6),
            language="english",
        )


def test_patch_context_payload_does_not_recurse() -> None:
    item = prompt_manifest()[143]
    scenario = _scenario(item["variant"])
    original = v8.chat_completion_payload
    with _patched_v8():
        payload = v8.chat_completion_payload(item["record_id"], scenario, item["language"], attempt_number=1)
        assert payload == chat_completion_payload(item["record_id"], scenario, item["language"], attempt_number=1)
    assert v8.chat_completion_payload is original


def test_v013_owner_acceptance_is_bound() -> None:
    repair = ACCEPTED_REPAIRS["bengali_exact_field_v0_13"]
    assert repair["owner_review_status"] == "ACCEPT_exact_bengali_field_quality"
    assert repair["accepted_rows"] == 1
