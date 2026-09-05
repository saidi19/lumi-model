from __future__ import annotations

import copy

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v11 as terminal_v11
from lumi.posttraining_first_party_teacher_canary_v13 import (
    CONTINUATION_CONTRACT,
    EXACT_MARKERS,
    FORMAT_EXAMPLES,
    PROMPT_CONTRACT,
    SAFETY_PROMPT_METHOD,
    _make_row,
    continuation_payload,
    continuation_prompt,
    validate_language_bound_safety,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


@pytest.mark.parametrize(
    ("scenario_id", "safety_label"),
    [
        ("credential_privacy", "refuse"),
        ("urgent_medical_caution", "medical_caution"),
    ],
)
@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english", "natural_code_switching"],
)
def test_every_non_safe_language_prompt_binds_exact_marker_and_example(
    scenario_id: str,
    safety_label: str,
    language_mode: str,
) -> None:
    prompt = continuation_prompt(_scenario(scenario_id), language_mode)
    marker = EXACT_MARKERS[(language_mode, safety_label)]
    example = FORMAT_EXAMPLES[(language_mode, safety_label)]
    assert marker in prompt
    assert example["user_prompt"] in prompt
    assert example["assistant_response"] in prompt
    assert "Safety-label contract" in prompt


@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english", "natural_code_switching"],
)
def test_safe_prompts_and_payloads_are_byte_equal_to_v11(language_mode: str) -> None:
    scenario = _scenario("benign_study_help")
    record_id = f"lumi-teacher-safety_helpfulness-{language_mode}-00"
    assert continuation_prompt(scenario, language_mode) == terminal_v11.continuation_prompt(
        scenario,
        language_mode,
    )
    assert continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    ) == terminal_v11.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )


@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english", "natural_code_switching"],
)
def test_non_safety_task_prompts_remain_byte_equal_to_v11(language_mode: str) -> None:
    scenario = _scenario("punctuation_repair")
    assert continuation_prompt(scenario, language_mode) == terminal_v11.continuation_prompt(
        scenario,
        language_mode,
    )


@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english", "natural_code_switching"],
)
def test_target_payload_changes_only_prompt(language_mode: str) -> None:
    scenario = _scenario("credential_privacy")
    record_id = f"lumi-teacher-safety_helpfulness-{language_mode}-01"
    previous = terminal_v11.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )
    corrected = continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )
    previous_without_prompt = copy.deepcopy(previous)
    corrected_without_prompt = copy.deepcopy(corrected)
    previous_without_prompt.pop("prompt")
    corrected_without_prompt.pop("prompt")
    assert corrected_without_prompt == previous_without_prompt
    assert corrected["prompt"] != previous["prompt"]


@pytest.mark.parametrize(
    ("language_mode", "safety_label", "valid", "invalid"),
    [
        ("banglish", "refuse", "Ami eta korte parbo na.", "Ami ei request e not help korbo."),
        ("banglish", "medical_caution", "Eta joruri.", "Doctor er kache jao."),
        ("bengali_script", "refuse", "আমি এটি করতে পারব না।", "আমি নিরাপদ বিকল্প দিচ্ছি।"),
        ("bengali_script", "medical_caution", "এটি জরুরি।", "হাসপাতালে যান।"),
        ("english", "refuse", "I cannot help with that.", "I won't help with that."),
        ("english", "medical_caution", "This is an emergency.", "Go to a hospital."),
        (
            "natural_code_switching",
            "refuse",
            "আমি এটি করতে পারব না (password)।",
            "আমি নিরাপদ বিকল্প দিচ্ছি (password)।",
        ),
        (
            "natural_code_switching",
            "medical_caution",
            "এটি জরুরি (emergency)।",
            "এখন হাসপাতালে যান (emergency)।",
        ),
    ],
)
def test_preaccept_validator_requires_exact_language_bound_marker(
    language_mode: str,
    safety_label: str,
    valid: str,
    invalid: str,
) -> None:
    validate_language_bound_safety(
        valid,
        safety_label,
        language_mode=language_mode,
    )
    with pytest.raises(ValueError, match="exact language-bound marker"):
        validate_language_bound_safety(
            invalid,
            safety_label,
            language_mode=language_mode,
        )


def test_target_row_records_language_bound_provenance() -> None:
    scenario = _scenario("credential_privacy")
    row = _make_row(
        "lumi-teacher-safety_helpfulness-bengali_script-01",
        scenario,
        "bengali_script",
        [
            {"role": "user", "content": "অন্য কারও পাসওয়ার্ড দেবে?"},
            {"role": "assistant", "content": "আমি অন্য কারও পাসওয়ার্ড দিতে পারব না।"},
        ],
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=None,
    )
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_13"
    assert row["provenance"]["prompt_contract"] == PROMPT_CONTRACT
    assert row["provenance"]["language_bound_safety_method"] == SAFETY_PROMPT_METHOD
    assert row["provenance"]["language_bound_safety_label"] == "refuse"
    assert row["provenance"]["language_bound_safety_marker"] == "পারব না"


def test_safe_row_has_no_language_bound_safety_provenance() -> None:
    scenario = _scenario("benign_study_help")
    row = _make_row(
        "lumi-teacher-safety_helpfulness-english-00",
        scenario,
        "english",
        [
            {"role": "user", "content": "Can you help me study?"},
            {"role": "assistant", "content": "Yes, I can help you make a study plan."},
        ],
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=None,
    )
    assert row["provenance"]["language_bound_safety_method"] is None
    assert row["provenance"]["language_bound_safety_label"] is None
    assert row["provenance"]["language_bound_safety_marker"] is None


def test_v13_continuation_is_exactly_bounded_to_remaining_rows() -> None:
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 129,
        "maximum_attempts_per_record": 3,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 43,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }
