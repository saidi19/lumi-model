from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v4 as schema_v4
from lumi.posttraining_first_party_teacher_canary_v9 import (
    ALLOWED_TERMINAL_PUNCTUATION,
    NATURAL_CODE_SWITCH_TEMPERATURE,
    TERMINAL_COMPOSITION_METHOD,
    compose_terminal_output,
    normalize_terminal_bengali_body,
    terminal_payload,
    terminal_seed,
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


def test_natural_code_switch_payload_uses_authorized_temperature() -> None:
    scenario = base.FACTUAL[0]
    payload = terminal_payload(
        "lumi-teacher-factual_qa-natural_code_switching-00",
        scenario,
        "natural_code_switching",
        attempt_number=1,
    )
    assert payload["options"]["temperature"] == NATURAL_CODE_SWITCH_TEMPERATURE == 0.2
    assert "terminal punctuation" in payload["prompt"]


@pytest.mark.parametrize("terminal", sorted(ALLOWED_TERMINAL_PUNCTUATION))
def test_exactly_one_allowed_terminal_punctuation_is_removed(terminal: str) -> None:
    body, removed = normalize_terminal_bengali_body(
        f"এক সপ্তাহে কত দিন থাকে{terminal}",
        field="user_prompt",
    )
    assert body == "এক সপ্তাহে কত দিন থাকে"
    assert removed == terminal


def test_punctuation_free_body_is_preserved_exactly() -> None:
    text = "এক সপ্তাহে কত দিন থাকে"
    body, removed = normalize_terminal_bengali_body(text, field="user_prompt")
    assert body == text
    assert removed is None


def test_terminal_normalization_and_suffix_are_the_only_edits() -> None:
    scenario = base.FACTUAL[1]
    flattened, messages, composition = compose_terminal_output(
        _output(
            scenario,
            user_prompt="এক সপ্তাহে কত দিন থাকে?",
            assistant_response="এক সপ্তাহে ঠিক 7 দিন থাকে।",
        ),
        scenario=scenario,
    )
    assert flattened["user_prompt"] == "এক সপ্তাহে কত দিন থাকে (week)?"
    assert flattened["assistant_response"] == "এক সপ্তাহে ঠিক 7 দিন থাকে (week)।"
    assert messages[0]["content"] == flattened["user_prompt"]
    assert messages[-1]["content"] == flattened["assistant_response"]
    assert composition["method"] == TERMINAL_COMPOSITION_METHOD
    assert composition["user_before"] == "এক সপ্তাহে কত দিন থাকে?"
    assert composition["user_normalized_body"] == "এক সপ্তাহে কত দিন থাকে"
    assert composition["user_removed_terminal"] == "?"
    assert composition["user_action"] == "replace_terminal_punctuation"
    assert composition["assistant_removed_terminal"] == "।"
    assert composition["assistant_action"] == "replace_terminal_punctuation"


def test_punctuation_free_fields_use_append_action() -> None:
    scenario = base.FACTUAL[1]
    _, _, composition = compose_terminal_output(
        _output(
            scenario,
            user_prompt="এক সপ্তাহে কত দিন থাকে",
            assistant_response="এক সপ্তাহে ঠিক 7 দিন থাকে",
        ),
        scenario=scenario,
    )
    assert composition["user_action"] == "append_suffix"
    assert composition["assistant_action"] == "append_suffix"
    assert composition["user_removed_terminal"] is None
    assert composition["assistant_removed_terminal"] is None


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("এক সপ্তাহে কত দিন, থাকে", "unsupported pre-suffix punctuation"),
        ("এক সপ্তাহে (কত) দিন থাকে", "unsupported pre-suffix punctuation"),
        ("এক সপ্তাহে কত দিন থাকে,", "unsupported pre-suffix punctuation"),
        ("এক week কত দিন থাকে", "pre-suffix Latin"),
        (" এক সপ্তাহে কত দিন থাকে", "surrounding whitespace"),
        ("এক সপ্তাহে কত দিন থাকে\n", "surrounding whitespace"),
    ],
)
def test_body_gate_rejects_non_exact_inputs(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        normalize_terminal_bengali_body(text, field="user_prompt")


def test_pre_suffix_safety_cannot_be_satisfied_by_bound_suffix() -> None:
    scenario = base.SAFETY[2]
    with pytest.raises(ValueError, match="escalation guidance"):
        compose_terminal_output(
            _output(
                scenario,
                user_prompt="আমার বুকে ব্যথা হচ্ছে কী করব?",
                assistant_response="শান্ত থাকুন এবং একটু অপেক্ষা করুন।",
            ),
            scenario=scenario,
        )


def test_every_required_ascii_literal_is_bound_inside_suffix_term() -> None:
    for scenario in base.SCENARIOS:
        term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
        assert all(literal in term for literal in scenario.required_literals)


def test_terminal_seeds_are_attempt_bound() -> None:
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    assert terminal_seed(record_id, 1) != terminal_seed(record_id, 2)
    with pytest.raises(ValueError, match="outside authorization"):
        terminal_seed(record_id, 4)
