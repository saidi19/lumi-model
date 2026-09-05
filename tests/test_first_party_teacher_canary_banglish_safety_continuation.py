from __future__ import annotations

import copy

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v11 as terminal_v11
from lumi.posttraining_first_party_teacher_canary_v12 import (
    BANGLISH_SAFETY_PROMPT_METHOD,
    CONTINUATION_CONTRACT,
    MEDICAL_ESCALATION,
    MEDICAL_FORMAT_EXAMPLE,
    PROMPT_CONTRACT,
    REFUSAL_BOUNDARY,
    REFUSAL_FORMAT_EXAMPLE,
    _make_row,
    continuation_payload,
    continuation_prompt,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


@pytest.mark.parametrize("scenario_id", ["credential_privacy", "phishing_request"])
def test_banglish_refusal_prompt_binds_exact_boundary(scenario_id: str) -> None:
    prompt = continuation_prompt(_scenario(scenario_id), "banglish")
    assert REFUSAL_BOUNDARY in prompt
    assert REFUSAL_FORMAT_EXAMPLE["user_prompt"] in prompt
    assert REFUSAL_FORMAT_EXAMPLE["assistant_response"] in prompt
    assert "romanized Bengali marker" in prompt


def test_banglish_medical_prompt_binds_exact_escalation() -> None:
    prompt = continuation_prompt(_scenario("urgent_medical_caution"), "banglish")
    assert MEDICAL_ESCALATION in prompt
    assert MEDICAL_FORMAT_EXAMPLE["user_prompt"] in prompt
    assert MEDICAL_FORMAT_EXAMPLE["assistant_response"] in prompt
    assert "immediate local emergency help" in prompt


def test_safe_banglish_prompt_and_payload_are_byte_equal_to_v11() -> None:
    scenario = _scenario("benign_study_help")
    record_id = "lumi-teacher-safety_helpfulness-banglish-00"
    assert continuation_prompt(scenario, "banglish") == terminal_v11.continuation_prompt(
        scenario,
        "banglish",
    )
    assert continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    ) == terminal_v11.continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    )


@pytest.mark.parametrize(
    "language_mode",
    ["bengali_script", "english", "natural_code_switching"],
)
@pytest.mark.parametrize(
    "scenario_id",
    ["credential_privacy", "urgent_medical_caution", "phishing_request"],
)
def test_all_non_banglish_safety_paths_are_byte_equal_to_v11(
    scenario_id: str,
    language_mode: str,
) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-safety_helpfulness-{language_mode}-{scenario.variant:02d}"
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


def test_target_payload_changes_only_prompt() -> None:
    scenario = _scenario("credential_privacy")
    record_id = "lumi-teacher-safety_helpfulness-banglish-01"
    previous = terminal_v11.continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    )
    corrected = continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    )
    previous_without_prompt = copy.deepcopy(previous)
    corrected_without_prompt = copy.deepcopy(corrected)
    previous_without_prompt.pop("prompt")
    corrected_without_prompt.pop("prompt")
    assert corrected_without_prompt == previous_without_prompt
    assert corrected["prompt"] != previous["prompt"]


def test_target_row_records_label_bound_provenance() -> None:
    scenario = _scenario("credential_privacy")
    row = _make_row(
        "lumi-teacher-safety_helpfulness-banglish-01",
        scenario,
        "banglish",
        [
            {"role": "user", "content": "Onno karo password dite parba?"},
            {
                "role": "assistant",
                "content": "Ami password dite parbo na, safe recovery te help korte pari.",
            },
        ],
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=None,
    )
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_12"
    assert row["provenance"]["prompt_contract"] == PROMPT_CONTRACT
    assert row["provenance"]["Banglish_safety_prompt_method"] == BANGLISH_SAFETY_PROMPT_METHOD
    assert row["provenance"]["Banglish_safety_prompt_label"] == "refuse"


def test_safe_row_has_no_safety_prompt_method() -> None:
    scenario = _scenario("benign_study_help")
    row = _make_row(
        "lumi-teacher-safety_helpfulness-banglish-00",
        scenario,
        "banglish",
        [
            {"role": "user", "content": "Quiz er jonno help korba?"},
            {"role": "assistant", "content": "Hae, ami plan korte help korbo."},
        ],
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=None,
    )
    assert row["provenance"]["Banglish_safety_prompt_method"] is None
    assert row["provenance"]["Banglish_safety_prompt_label"] is None


def test_v12_continuation_is_exactly_bounded_to_remaining_rows() -> None:
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 132,
        "maximum_attempts_per_record": 3,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 44,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }
