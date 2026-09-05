from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi.posttraining_first_party_teacher_canary_v6 import (
    ANCHOR_SUBSTITUTION_METHOD,
    ANCHOR_SUBSTITUTIONS,
    _standalone_anchor_pattern,
    anchor_payload,
    anchor_seed,
    substitute_anchor_output,
)


def test_every_scenario_has_bengali_anchor_and_english_constant() -> None:
    assert set(ANCHOR_SUBSTITUTIONS) == {scenario.scenario_id for scenario in base.SCENARIOS}
    for anchor, english in ANCHOR_SUBSTITUTIONS.values():
        assert base._BENGALI.search(anchor)
        assert base._LATIN.search(anchor) is None
        assert base._LATIN.search(english)


def test_standalone_anchor_does_not_match_inside_longer_bengali_word() -> None:
    pattern = _standalone_anchor_pattern("মিটার")
    text = "এক কিলোমিটারে 1000 মিটার থাকে।"
    matches = list(pattern.finditer(text))
    assert len(matches) == 1
    assert matches[0].group() == "মিটার"


def test_anchor_substitution_is_the_only_text_change() -> None:
    scenario = base.SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": "এক কিলোমিটারে কত মিটার থাকে?",
        "assistant_response": "এক কিলোমিটারে 1000 মিটার থাকে।",
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    flattened, messages, evidence = substitute_anchor_output(output, scenario=scenario)
    assert flattened["user_prompt"] == "এক কিলোমিটারে কত meter থাকে?"
    assert flattened["assistant_response"] == "এক কিলোমিটারে 1000 meter থাকে।"
    assert evidence["method"] == ANCHOR_SUBSTITUTION_METHOD
    assert evidence["substitutions_per_field"] == 1
    assert [message["content"] for message in messages] == [
        flattened["user_prompt"],
        flattened["assistant_response"],
    ]


def test_anchor_substitution_rejects_unbound_latin_text() -> None:
    scenario = base.SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": "Please বলুন, এক কিলোমিটারে কত মিটার থাকে?",
        "assistant_response": "এক কিলোমিটারে 1000 মিটার থাকে।",
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    with pytest.raises(ValueError, match="unbound Latin"):
        substitute_anchor_output(output, scenario=scenario)


def test_required_ascii_literals_are_allowed_but_preserved() -> None:
    scenario = next(item for item in base.SCENARIOS if item.scenario_id == "meeting_summary")
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": "10:00, Room B এবং ID তথ্যের সারাংশ দিন।",
        "assistant_response": "সারাংশ হলো: 10:00-এ Room B-তে সভা; ID আনতে হবে।",
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    flattened, _, _ = substitute_anchor_output(output, scenario=scenario)
    assert "summary" in flattened["user_prompt"]
    assert "summary" in flattened["assistant_response"]
    for literal in scenario.required_literals:
        assert literal in flattened["user_prompt"]
        assert literal in flattened["assistant_response"]


def test_anchor_payload_uses_full_string_schema_and_bounded_seed() -> None:
    scenario = base.SCENARIOS[0]
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    payload = anchor_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    assert payload["options"]["seed"] == anchor_seed(record_id, 2)
    assert payload["format"]["properties"]["user_prompt"]["type"] == "string"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
