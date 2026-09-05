from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v4 as schema_v4
from lumi.posttraining_first_party_teacher_canary_v7 import (
    PARENTHETICAL_INSERTION_METHOD,
    _insert_before_terminal,
    insert_parenthetical_output,
    parenthetical_seed,
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


def test_every_required_ascii_literal_is_bound_inside_parenthetical_term() -> None:
    assert set(schema_v4.CODE_SWITCH_TERMS) == {scenario.scenario_id for scenario in base.SCENARIOS}
    for scenario in base.SCENARIOS:
        term = schema_v4.CODE_SWITCH_TERMS[scenario.scenario_id]
        assert all(literal in term for literal in scenario.required_literals)


def test_parenthetical_insertion_is_the_only_edit() -> None:
    scenario = base.FACTUAL[1]
    flattened, messages, insertion = insert_parenthetical_output(
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
    assert insertion == {
        "assistant_after": "এক সপ্তাহে ঠিক 7 দিন থাকে (week)।",
        "assistant_before": "এক সপ্তাহে ঠিক 7 দিন থাকে।",
        "english_schema_constant": "week",
        "inserted_text": " (week)",
        "insertions_per_field": 1,
        "method": PARENTHETICAL_INSERTION_METHOD,
        "user_after": "এক সপ্তাহে কত দিন থাকে (week)?",
        "user_before": "এক সপ্তাহে কত দিন থাকে?",
    }


def test_writing_required_literals_arrive_only_through_bound_insertion() -> None:
    scenario = base.WRITING[0]
    flattened, _, _ = insert_parenthetical_output(
        _output(
            scenario,
            user_prompt="সভাটির তথ্য সংক্ষেপে বলো।",
            assistant_response="সভাটির সময়, স্থান ও প্রয়োজনীয় পরিচয়পত্র মনে রাখতে হবে।",
        ),
        scenario=scenario,
    )
    for literal in scenario.required_literals:
        assert flattened["user_prompt"].count(literal) == 1
        assert flattened["assistant_response"].count(literal) == 1


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("এক সপ্তাহে কত দিন থাকে", "lacks final punctuation"),
        ("এক সপ্তাহে কত দিন থাকে? ", "surrounding whitespace"),
        ("এক সপ্তাহে কত দিন থাকে?!", "stacked terminal punctuation"),
        ("এক week-এ কত দিন থাকে?", "pre-insertion Latin"),
    ],
)
def test_parenthetical_input_gate_rejects_non_exact_inputs(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _insert_before_terminal(text, "week", field="user_prompt")


def test_pre_insertion_safety_cannot_be_satisfied_by_injected_english_only() -> None:
    scenario = base.SAFETY[2]
    with pytest.raises(ValueError, match="escalation guidance"):
        insert_parenthetical_output(
            _output(
                scenario,
                user_prompt="আমার বুকে ব্যথা হচ্ছে, কী করব?",
                assistant_response="শান্ত থাকুন এবং একটু অপেক্ষা করুন।",
            ),
            scenario=scenario,
        )


def test_parenthetical_seeds_are_attempt_bound() -> None:
    record_id = "lumi-teacher-factual_qa-natural_code_switching-01"
    assert parenthetical_seed(record_id, 1) != parenthetical_seed(record_id, 2)
    with pytest.raises(ValueError, match="outside authorization"):
        parenthetical_seed(record_id, 4)
