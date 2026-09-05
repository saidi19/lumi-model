from __future__ import annotations

import pytest

from lumi.posttraining_first_party_teacher_canary import SCENARIOS
from lumi.posttraining_first_party_teacher_canary_v4 import (
    CODE_SWITCH_TERMS,
    SCHEMA_COMPOSITION_METHOD,
    compose_schema_constrained_output,
    schema_completion_payload,
    schema_completion_seed,
    structured_code_switch_schema,
)


def test_every_scenario_has_a_schema_constant_term() -> None:
    assert set(CODE_SWITCH_TERMS) == {scenario.scenario_id for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        term = CODE_SWITCH_TERMS[scenario.scenario_id]
        assert any(character.isascii() and character.isalpha() for character in term)
        for literal in scenario.required_literals:
            assert literal in term


def test_schema_exposes_only_bengali_prefix_and_suffix_slots() -> None:
    scenario = SCENARIOS[0]
    schema = structured_code_switch_schema(scenario)
    assert schema["properties"]["scenario_id"]["enum"] == [scenario.scenario_id]
    assert schema["properties"]["safety_label"]["enum"] == [scenario.safety_label]
    assert set(schema["properties"]["user_prompt"]["properties"]) == {
        "prefix",
        "suffix",
    }
    assert "term" not in schema["properties"]["user_prompt"]["properties"]


def test_composition_inserts_only_the_schema_constant() -> None:
    scenario = SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": {"prefix": "এক কিলোমিটারে", "suffix": "কত থাকে?"},
        "assistant_response": {
            "prefix": "এক কিলোমিটারে 1000",
            "suffix": "থাকে।",
        },
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    flattened, messages = compose_schema_constrained_output(output, scenario=scenario)
    term = CODE_SWITCH_TERMS[scenario.scenario_id]
    assert flattened["user_prompt"].count(term) == 1
    assert flattened["assistant_response"].count(term) == 1
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_composition_rejects_freeform_latin_in_bengali_slots() -> None:
    scenario = SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": {"prefix": "এক kilometer-এ", "suffix": "কত থাকে?"},
        "assistant_response": {
            "prefix": "এক কিলোমিটারে 1000",
            "suffix": "থাকে।",
        },
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    with pytest.raises(ValueError, match="free-form Latin"):
        compose_schema_constrained_output(output, scenario=scenario)


def test_schema_completion_payload_is_local_bounded_and_deterministic() -> None:
    scenario = SCENARIOS[0]
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    payload = schema_completion_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    assert payload["options"]["seed"] == schema_completion_seed(record_id, 2)
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert set(payload["format"]["properties"]["assistant_response"]["properties"]) == {
        "prefix",
        "suffix",
    }
    assert SCHEMA_COMPOSITION_METHOD == "trim_join_single_spaces_v1"
