from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v4 as schema_v4
from lumi.posttraining_first_party_teacher_canary_v8 import (
    FORMAT_EXAMPLE,
    NATURAL_CODE_SWITCH_TEMPERATURE,
    SUFFIX_COMPOSITION_METHOD,
    _validate_punctuation_free_bengali_body,
    compose_suffix_output,
    suffix_payload,
    suffix_seed,
)


def _output(
    scenario: base.Scenario,
    *,
    user_prompt: str,
    assistant_response: str,
) -> dict[str, object]:
    return {
        "assistant_response": assistant_response,
        "safety_label": scenario.safety_label,
        "scenario_id": scenario.scenario_id,
        "tool_call": None
        if scenario.tool_name is None
        else {"name": scenario.tool_name, "arguments": scenario.tool_arguments},
        "tool_result": scenario.tool_result,
        "user_prompt": user_prompt,
    }


def test_format_example_is_bengali_punctuation_free_and_latin_free() -> None:
    assert set(FORMAT_EXAMPLE) == {"assistant_response", "user_prompt"}
    for field, text in FORMAT_EXAMPLE.items():
        _validate_punctuation_free_bengali_body(text, field=field)


def test_natural_code_switch_payload_uses_authorized_temperature() -> None:
    scenario = base.FACTUAL[0]
    payload = suffix_payload(
        "lumi-teacher-factual_qa-natural_code_switching-00",
        scenario,
        "natural_code_switching",
        attempt_number=1,
    )
    assert payload["options"]["temperature"] == NATURAL_CODE_SWITCH_TEMPERATURE == 0.2
    assert "এক বছরে কত মাস থাকে" in payload["prompt"]


def test_suffix_composition_is_the_only_edit() -> None:
    scenario = base.FACTUAL[1]
    flattened, messages, composition = compose_suffix_output(
        _output(
            scenario,
            user_prompt="এক সপ্তাহে কত দিন থাকে",
            assistant_response="এক সপ্তাহে ঠিক 7 দিন থাকে",
        ),
        scenario=scenario,
    )
    assert flattened["user_prompt"] == "এক সপ্তাহে কত দিন থাকে (week)?"
    assert flattened["assistant_response"] == "এক সপ্তাহে ঠিক 7 দিন থাকে (week)।"
    assert messages[0]["content"] == flattened["user_prompt"]
    assert messages[-1]["content"] == flattened["assistant_response"]
    assert composition == {
        "assistant_after": "এক সপ্তাহে ঠিক 7 দিন থাকে (week)।",
        "assistant_before": "এক সপ্তাহে ঠিক 7 দিন থাকে",
        "assistant_suffix": " (week)।",
        "english_schema_constant": "week",
        "method": SUFFIX_COMPOSITION_METHOD,
        "suffixes_per_field": 1,
        "user_after": "এক সপ্তাহে কত দিন থাকে (week)?",
        "user_before": "এক সপ্তাহে কত দিন থাকে",
        "user_suffix": " (week)?",
    }


def test_writing_required_literals_arrive_only_through_suffix() -> None:
    scenario = base.WRITING[0]
    flattened, _, _ = compose_suffix_output(
        _output(
            scenario,
            user_prompt="সভাটির তথ্য সংক্ষেপে বলো",
            assistant_response="সভাটির সময় স্থান ও প্রয়োজনীয় পরিচয়পত্র মনে রাখতে হবে",
        ),
        scenario=scenario,
    )
    for literal in scenario.required_literals:
        assert flattened["user_prompt"].count(literal) == 1
        assert flattened["assistant_response"].count(literal) == 1


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("এক সপ্তাহে কত দিন থাকে?", "pre-suffix punctuation"),
        ("এক সপ্তাহে কত দিন থাকে।", "pre-suffix punctuation"),
        ("এক সপ্তাহে কত দিন থাকে, বলো", "pre-suffix punctuation"),
        ("এক week কত দিন থাকে", "pre-suffix Latin"),
        (" এক সপ্তাহে কত দিন থাকে", "surrounding whitespace"),
        ("এক সপ্তাহে কত দিন থাকে\n", "surrounding whitespace"),
    ],
)
def test_body_gate_rejects_non_exact_inputs(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _validate_punctuation_free_bengali_body(text, field="user_prompt")


def test_pre_suffix_safety_cannot_be_satisfied_by_english_suffix_only() -> None:
    scenario = base.SAFETY[2]
    with pytest.raises(ValueError, match="escalation guidance"):
        compose_suffix_output(
            _output(
                scenario,
                user_prompt="আমার বুকে ব্যথা হচ্ছে কী করব",
                assistant_response="শান্ত থাকুন এবং একটু অপেক্ষা করুন",
            ),
            scenario=scenario,
        )


def test_every_required_ascii_literal_is_bound_inside_suffix_term() -> None:
    for scenario in base.SCENARIOS:
        term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
        assert all(literal in term for literal in scenario.required_literals)


def test_suffix_seeds_are_attempt_bound() -> None:
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    assert suffix_seed(record_id, 1) != suffix_seed(record_id, 2)
    with pytest.raises(ValueError, match="outside authorization"):
        suffix_seed(record_id, 4)
