from __future__ import annotations

import copy

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v5 as corrected_v5
from lumi import posttraining_first_party_teacher_canary_v14 as punctuation_v14
from lumi.posttraining_first_party_teacher_canary_v15 import (
    BANGLISH_STYLE_METHOD,
    BANGLISH_TASK_CONTRACTS,
    BANGLISH_TEMPERATURE,
    CARRY_RECORDS,
    CONTINUATION_CONTRACT,
    NEW_BANGLISH_ROWS,
    NEW_NATURAL_CODE_SWITCHING_ROWS,
    NEW_RECORDS,
    PROMPT_CONTRACT,
    _make_row,
    _patched_v14_generation,
    continuation_payload,
    continuation_prompt,
    validate_task_anchored_banglish,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


def _output(
    *,
    scenario: base.Scenario,
    user: str,
    assistant: str,
) -> dict[str, object]:
    return {
        "assistant_response": assistant,
        "safety_label": scenario.safety_label,
        "scenario_id": scenario.scenario_id,
        "tool_call": (
            {"arguments": scenario.tool_arguments, "name": scenario.tool_name}
            if scenario.tool_name
            else None
        ),
        "tool_result": scenario.tool_result,
        "user_prompt": user,
    }


def test_contract_matches_exact_remaining_prefix() -> None:
    assert CARRY_RECORDS == 52
    assert NEW_RECORDS == 28
    assert NEW_BANGLISH_ROWS == 7
    assert NEW_NATURAL_CODE_SWITCHING_ROWS == 7
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 84,
        "maximum_attempts_per_record": 3,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 28,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }


def test_exact_remaining_rows_receive_the_intended_narrow_fix() -> None:
    matrix = [
        (scenario, language_mode, record_id)
        for scenario, language_mode, record_id in zip(
            (scenario for scenario in base.SCENARIOS for _ in base.LANGUAGE_MODES),
            (
                language_mode
                for _scenario in base.SCENARIOS
                for language_mode in base.LANGUAGE_MODES
            ),
            base.expected_record_ids(),
            strict=True,
        )
    ][CARRY_RECORDS:]
    targeted = [item for item in matrix if item[1] == "banglish"]
    code_switch = [item for item in matrix if item[1] == "natural_code_switching"]
    assert len(targeted) == NEW_BANGLISH_ROWS
    assert len(code_switch) == NEW_NATURAL_CODE_SWITCHING_ROWS
    assert [scenario.task for scenario, _language, _record_id in targeted].count("tool_api") == 3
    assert [scenario.task for scenario, _language, _record_id in targeted].count(
        "writing_summarization"
    ) == 4
    for scenario, language_mode, record_id in targeted:
        contract = BANGLISH_TASK_CONTRACTS[scenario.task]
        prompt = continuation_prompt(scenario, language_mode)
        payload = continuation_payload(
            record_id,
            scenario,
            language_mode,
            attempt_number=1,
        )
        assert str(contract["user_marker"]) in prompt
        assert str(contract["assistant_marker"]) in prompt
        assert payload["options"]["temperature"] == BANGLISH_TEMPERATURE


@pytest.mark.parametrize(
    "scenario_id",
    ["calculator_multiplication", "calculator_division", "project_summary"],
)
@pytest.mark.parametrize(
    "language_mode",
    ["bengali_script", "english", "natural_code_switching"],
)
def test_non_banglish_prompts_and_payloads_are_byte_equal_v14(
    scenario_id: str,
    language_mode: str,
) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-{scenario.task}-{language_mode}-02"
    assert continuation_prompt(scenario, language_mode) == punctuation_v14.continuation_prompt(
        scenario,
        language_mode,
    )
    assert continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    ) == punctuation_v14.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )


@pytest.mark.parametrize(
    "scenario_id",
    ["calculator_multiplication", "list_sort", "meeting_summary", "project_summary"],
)
def test_targeted_banglish_prompt_binds_task_example_and_exact_markers(
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)
    contract = BANGLISH_TASK_CONTRACTS[scenario.task]
    prompt = continuation_prompt(scenario, "banglish")
    assert "Banglish style contract" in prompt
    assert str(contract["user_marker"]) in prompt
    assert str(contract["assistant_marker"]) in prompt
    assert str(contract["example"]["user_prompt"]) in prompt
    assert str(contract["example"]["assistant_response"]) in prompt


@pytest.mark.parametrize(
    "scenario_id",
    ["calculator_multiplication", "project_summary"],
)
def test_target_payload_changes_only_prompt_and_temperature(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-{scenario.task}-banglish-02"
    previous = punctuation_v14.continuation_payload(
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
    previous_without_prompt_options = copy.deepcopy(previous)
    corrected_without_prompt_options = copy.deepcopy(corrected)
    previous_without_prompt_options.pop("prompt")
    corrected_without_prompt_options.pop("prompt")
    previous_without_prompt_options["options"].pop("temperature")
    corrected_without_prompt_options["options"].pop("temperature")
    assert corrected_without_prompt_options == previous_without_prompt_options
    assert corrected["prompt"] != previous["prompt"]
    assert corrected["options"]["temperature"] == BANGLISH_TEMPERATURE


def test_tool_banglish_exact_markers_and_contract_pass() -> None:
    scenario = _scenario("calculator_multiplication")
    output = _output(
        scenario=scenario,
        user="132 ar 11 gun korle uttor koto hobe?",
        assistant="Hishab korle uttor 1452 hobe.",
    )
    messages = validate_task_anchored_banglish(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    assert messages[0]["content"].endswith("hobe?")
    assert messages[-1]["content"].endswith("hobe.")


def test_writing_banglish_exact_markers_pass() -> None:
    scenario = _scenario("meeting_summary")
    output = _output(
        scenario=scenario,
        user="Eta note koro: meeting 10:00 te Room B te, ar ID ante hobe.",
        assistant="Ami summary dichhi: meeting 10:00 te Room B te, ebong ID ante hobe.",
    )
    messages = validate_task_anchored_banglish(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    assert len(messages) == 2


@pytest.mark.parametrize(
    ("user", "assistant", "error"),
    [
        (
            "132 এবং 11 এর গুণফল কত?",
            "132 এবং 11 গুণ করলে 1452 হয়।",
            "Bengali script",
        ),
        (
            "Calculate 132 multiplied by 11.",
            "132 multiplied by 11 is 1452.",
            "romanized Bengali markers",
        ),
        (
            "132 × 11 kina?",
            "132 multiplication 11 kina? 1452 ache.",
            "romanized Bengali markers",
        ),
    ],
)
def test_recorded_failure_classes_remain_rejected(
    user: str,
    assistant: str,
    error: str,
) -> None:
    scenario = _scenario("calculator_multiplication")
    output = _output(scenario=scenario, user=user, assistant=assistant)
    with pytest.raises(ValueError, match=error):
        validate_task_anchored_banglish(
            output,
            scenario=scenario,
            language_mode="banglish",
        )


def test_base_valid_banglish_without_task_marker_is_rejected() -> None:
    scenario = _scenario("calculator_multiplication")
    output = _output(
        scenario=scenario,
        user="132 ar 11 gun korle uttor koto ache?",
        assistant="Hishab korle uttor 1452 ache.",
    )
    with pytest.raises(ValueError, match="exact task-bound marker: hobe"):
        validate_task_anchored_banglish(
            output,
            scenario=scenario,
            language_mode="banglish",
        )


def test_row_records_task_anchored_provenance() -> None:
    scenario = _scenario("calculator_multiplication")
    output = _output(
        scenario=scenario,
        user="132 ar 11 gun korle uttor koto hobe?",
        assistant="Hishab korle uttor 1452 hobe.",
    )
    messages = validate_task_anchored_banglish(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    with _patched_v14_generation():
        row = _make_row(
            "lumi-teacher-tool_api-banglish-01",
            scenario,
            "banglish",
            messages,
            attempt_number=1,
            terminal_composition=None,
            banglish_composition={"method": "test"},
        )
    provenance = row["provenance"]
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_15"
    assert provenance["prompt_contract"] == PROMPT_CONTRACT
    assert provenance["Banglish_style_method"] == BANGLISH_STYLE_METHOD
    assert provenance["Banglish_user_marker"] == "hobe"
    assert provenance["Banglish_assistant_marker"] == "hobe"
    assert provenance["Banglish_temperature"] == BANGLISH_TEMPERATURE


def test_generation_patch_restores_predecessor_globals() -> None:
    payload = punctuation_v14.continuation_payload
    make_row = punctuation_v14._make_row  # noqa: SLF001
    validate = corrected_v5.validate_corrected_output
    with _patched_v14_generation():
        assert punctuation_v14.continuation_payload is continuation_payload
        assert punctuation_v14._make_row is _make_row  # noqa: SLF001
        assert corrected_v5.validate_corrected_output is validate_task_anchored_banglish
    assert punctuation_v14.continuation_payload is payload
    assert punctuation_v14._make_row is make_row  # noqa: SLF001
    assert corrected_v5.validate_corrected_output is validate
