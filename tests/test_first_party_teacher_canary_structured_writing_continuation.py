from __future__ import annotations

import copy

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v4 as schema_v4
from lumi import posttraining_first_party_teacher_canary_v15 as task_v15
from lumi.posttraining_first_party_teacher_canary_v16 import (
    ASSISTANT_WRAPPER,
    CARRY_RECORDS,
    CONTINUATION_CONTRACT,
    NEW_NATURAL_CODE_SWITCHING_ROWS,
    NEW_RECORDS,
    NEW_STRUCTURED_BANGLISH_ROWS,
    PROMPT_CONTRACT,
    USER_WRAPPER,
    WRITING_BODY_COMPOSITION_METHOD,
    WRITING_BODY_INSTRUCTIONS,
    WRITING_BODY_TEMPERATURE,
    _make_row,
    _patched_v4_runtime,
    _targeted,
    compose_writing_bodies,
    continuation_payload,
    continuation_prompt,
    writing_body_schema,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


def _output(
    *,
    scenario: base.Scenario,
    user_body: str,
    assistant_body: str,
) -> dict[str, object]:
    return {
        "assistant_response": {"body": assistant_body},
        "safety_label": scenario.safety_label,
        "scenario_id": scenario.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": {"body": user_body},
    }


def _meeting_output() -> dict[str, object]:
    scenario = _scenario("meeting_summary")
    return _output(
        scenario=scenario,
        user_body="Meeting 10:00 e Room B te hobe, ar ID ante hobe.",
        assistant_body="10:00 e Room B te meeting hobe, ar ID ante hobe.",
    )


def _remaining_matrix() -> list[tuple[base.Scenario, str, str]]:
    return list(
        zip(
            (scenario for scenario in base.SCENARIOS for _ in base.LANGUAGE_MODES),
            (
                language_mode
                for _scenario_item in base.SCENARIOS
                for language_mode in base.LANGUAGE_MODES
            ),
            base.expected_record_ids(),
            strict=True,
        )
    )[CARRY_RECORDS:]


def test_contract_matches_exact_sixty_four_row_prefix() -> None:
    assert CARRY_RECORDS == 64
    assert NEW_RECORDS == 16
    assert NEW_STRUCTURED_BANGLISH_ROWS == 4
    assert NEW_NATURAL_CODE_SWITCHING_ROWS == 4
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 48,
        "maximum_attempts_per_record": 3,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 16,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }


def test_exact_remaining_rows_receive_only_the_narrow_fix() -> None:
    matrix = _remaining_matrix()
    targeted = [item for item in matrix if _targeted(item[0], item[1])]
    code_switch = [item for item in matrix if item[1] == "natural_code_switching"]
    assert len(matrix) == NEW_RECORDS
    assert len(targeted) == NEW_STRUCTURED_BANGLISH_ROWS
    assert len(code_switch) == NEW_NATURAL_CODE_SWITCHING_ROWS
    assert {scenario.scenario_id for scenario, _language, _record_id in targeted} == set(
        WRITING_BODY_INSTRUCTIONS
    )
    assert all(scenario.task == "writing_summarization" for scenario, _mode, _id in targeted)


@pytest.mark.parametrize(
    "scenario_id",
    ["meeting_summary", "release_summary", "plant_care_summary", "project_summary"],
)
@pytest.mark.parametrize(
    "language_mode",
    ["bengali_script", "english", "natural_code_switching"],
)
def test_non_target_prompts_and_payloads_are_byte_equal_v15(
    scenario_id: str,
    language_mode: str,
) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-writing_summarization-{language_mode}-99"
    assert continuation_prompt(scenario, language_mode) == task_v15.continuation_prompt(
        scenario,
        language_mode,
    )
    assert continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    ) == task_v15.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )


@pytest.mark.parametrize(
    "scenario_id",
    ["meeting_summary", "release_summary", "plant_care_summary", "project_summary"],
)
def test_target_prompt_and_schema_bind_structured_bodies(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    prompt = continuation_prompt(scenario, "banglish")
    schema = writing_body_schema(scenario)
    assert WRITING_BODY_INSTRUCTIONS[scenario_id] in prompt
    assert "objects with exactly one field named body" in prompt
    assert "runtime will add fixed natural wrappers" in prompt
    assert "Eta or Ami" in prompt
    assert schema["properties"]["scenario_id"]["enum"] == [scenario_id]
    assert schema["properties"]["tool_call"] == {"type": "null"}
    assert schema["properties"]["tool_result"] == {"type": "null"}
    for field in ("user_prompt", "assistant_response"):
        assert schema["properties"][field]["required"] == ["body"]
        assert schema["properties"][field]["additionalProperties"] is False


def test_target_payload_changes_only_prompt_format_and_temperature_contract() -> None:
    scenario = _scenario("meeting_summary")
    record_id = "lumi-teacher-writing_summarization-banglish-00"
    previous = task_v15.continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    )
    structured = continuation_payload(
        record_id,
        scenario,
        "banglish",
        attempt_number=2,
    )
    previous_rest = copy.deepcopy(previous)
    structured_rest = copy.deepcopy(structured)
    previous_rest.pop("prompt")
    structured_rest.pop("prompt")
    previous_rest.pop("format")
    structured_rest.pop("format")
    assert structured_rest == previous_rest
    assert structured["prompt"] != previous["prompt"]
    assert structured["format"] != previous["format"]
    assert structured["options"]["temperature"] == WRITING_BODY_TEMPERATURE


def test_structured_bodies_receive_fixed_wrappers_and_remain_byte_exact() -> None:
    scenario = _scenario("meeting_summary")
    output = _meeting_output()
    normalized, messages, terminal, composition = compose_writing_bodies(
        output,
        scenario=scenario,
    )
    user_body = output["user_prompt"]["body"]  # type: ignore[index]
    assistant_body = output["assistant_response"]["body"]  # type: ignore[index]
    assert normalized["user_prompt"] == f"{USER_WRAPPER}{user_body}"
    assert normalized["assistant_response"] == f"{ASSISTANT_WRAPPER}{assistant_body}"
    assert messages[0]["content"] == normalized["user_prompt"]
    assert messages[-1]["content"] == normalized["assistant_response"]
    assert terminal["user_prompt_action"] == "preserve_existing_terminal"
    assert terminal["assistant_response_action"] == "preserve_existing_terminal"
    assert composition["body_fields_preserved_byte_exact"] is True
    assert composition["method"] == WRITING_BODY_COMPOSITION_METHOD
    assert task_v15._has_exact_word(normalized["user_prompt"], "eta")  # noqa: SLF001
    assert task_v15._has_exact_word(normalized["assistant_response"], "ami")  # noqa: SLF001


@pytest.mark.parametrize(
    ("user_body", "assistant_body", "error"),
    [
        (
            "Meeting 10:00 e Room B te hobe, ar ID ante hobe.",
            "Meeting 10:00 e Room B te hobe, ar ID ante hobe.",
            "identical",
        ),
        (
            "Meeting 10:00 e Room B te hobe, ar ID ante hobe.",
            "10:00 e Room B te meeting hobe, ar ID ante hobe?",
            "summary body is a question",
        ),
        (
            "মিটিং 10:00-এ Room B-তে হবে, ID আনতে হবে।",
            "10:00 e Room B te meeting hobe, ar ID ante hobe.",
            "Bengali script",
        ),
        (
            "Meeting at 10:00 in Room B; bring ID.",
            "The meeting is at 10:00 in Room B; bring ID.",
            "romanized Bengali markers",
        ),
        (
            "Meeting 10:00 e Room B te hobe, ar ID ante hobe",
            "10:00 e Room B te meeting hobe, ar ID ante hobe.",
            "terminal punctuation",
        ),
        (
            "Meeting 10:00 e Room B te hobe, ar ID ante hobe.",
            "10:00 e meeting hobe, ar ID ante hobe.",
            "Room B",
        ),
    ],
)
def test_invalid_structured_body_classes_fail_closed(
    user_body: str,
    assistant_body: str,
    error: str,
) -> None:
    scenario = _scenario("meeting_summary")
    with pytest.raises(ValueError, match=error):
        compose_writing_bodies(
            _output(
                scenario=scenario,
                user_body=user_body,
                assistant_body=assistant_body,
            ),
            scenario=scenario,
        )


def test_envelope_and_nested_field_schema_fail_closed() -> None:
    scenario = _scenario("meeting_summary")
    output = _meeting_output()
    output["unexpected"] = True
    with pytest.raises(ValueError, match="envelope fields changed"):
        compose_writing_bodies(output, scenario=scenario)
    output = _meeting_output()
    output["user_prompt"] = {"body": "valid", "extra": "forbidden"}
    with pytest.raises(ValueError, match="must contain only body"):
        compose_writing_bodies(output, scenario=scenario)


def test_row_records_structured_composition_provenance() -> None:
    scenario = _scenario("meeting_summary")
    _normalized, messages, terminal, writing = compose_writing_bodies(
        _meeting_output(),
        scenario=scenario,
    )
    row = _make_row(
        "lumi-teacher-writing_summarization-banglish-00",
        scenario,
        "banglish",
        messages,
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=terminal,
        writing_body_composition=writing,
    )
    provenance = row["provenance"]
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_16"
    assert provenance["prompt_contract"] == PROMPT_CONTRACT
    assert provenance["writing_body_composition_method"] == WRITING_BODY_COMPOSITION_METHOD
    assert provenance["writing_body_temperature"] == WRITING_BODY_TEMPERATURE
    assert provenance["Banglish_user_marker"] == "eta"
    assert provenance["Banglish_assistant_marker"] == "ami"


def test_runtime_patch_restores_predecessor_globals() -> None:
    contract = schema_v4.SCHEMA_COMPLETION_CONTRACT
    validate = schema_v4._validate_schema_completion_plan  # noqa: SLF001
    initialize = schema_v4._initialize_or_load  # noqa: SLF001
    generate = schema_v4.generate_schema_completion_attempt
    with _patched_v4_runtime():
        assert schema_v4.SCHEMA_COMPLETION_CONTRACT is CONTINUATION_CONTRACT
    assert schema_v4.SCHEMA_COMPLETION_CONTRACT is contract
    assert schema_v4._validate_schema_completion_plan is validate  # noqa: SLF001
    assert schema_v4._initialize_or_load is initialize  # noqa: SLF001
    assert schema_v4.generate_schema_completion_attempt is generate
