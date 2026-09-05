from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi.posttraining_first_party_teacher_canary_v5 import (
    ADDITIONAL_BANGLISH_MARKERS,
    BANGLISH_MARKER_CONTRACT,
    CORRECTED_BANGLISH_MARKERS,
    compose_refined_code_switch_output,
    corrected_payload,
    corrected_seed,
    validate_corrected_output,
)


def test_curated_banglish_expansion_is_bounded() -> None:
    assert {"koyta", "aikta"} <= ADDITIONAL_BANGLISH_MARKERS
    assert not ({"day", "week", "summary", "calculate"} & CORRECTED_BANGLISH_MARKERS)
    assert BANGLISH_MARKER_CONTRACT == "curated-romanized-bengali-markers.v0.2"


def test_verified_natural_banglish_output_now_passes_each_field_gate() -> None:
    scenario = base.SCENARIOS[1]
    output = {
        "assistant_response": "Tumi eta ki hoy? Hobe 7 day koro. Lagbe 7 day.",
        "safety_label": "safe",
        "scenario_id": "days_in_week",
        "tool_call": None,
        "tool_result": None,
        "user_prompt": "Koyta day aikta week a?",
    }
    messages = validate_corrected_output(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert base._BANGLISH_MARKERS != set(CORRECTED_BANGLISH_MARKERS)


def test_refined_code_switch_rejects_complete_sentence_prefix() -> None:
    scenario = base.SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": {"prefix": "এক কিলোমিটার কত মিটার?", "suffix": "বলুন।"},
        "assistant_response": {
            "prefix": "এক কিলোমিটারে ১০০০ মিটার আছে।",
            "suffix": "হয়।",
        },
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    with pytest.raises(ValueError, match="not a fragment"):
        compose_refined_code_switch_output(output, scenario=scenario)


def test_refined_code_switch_accepts_unfinished_bengali_fragments() -> None:
    scenario = base.SCENARIOS[0]
    output = {
        "scenario_id": scenario.scenario_id,
        "user_prompt": {"prefix": "এক কিলোমিটারে কত", "suffix": "থাকে?"},
        "assistant_response": {
            "prefix": "এক কিলোমিটারে 1000",
            "suffix": "থাকে।",
        },
        "safety_label": scenario.safety_label,
        "tool_call": None,
        "tool_result": None,
    }
    flattened, messages = compose_refined_code_switch_output(output, scenario=scenario)
    assert flattened["user_prompt"] == "এক কিলোমিটারে কত meter থাকে?"
    assert flattened["assistant_response"] == "এক কিলোমিটারে 1000 meter থাকে।"
    assert len(messages) == 2


def test_corrected_payload_is_local_bounded_and_deterministic() -> None:
    scenario = base.SCENARIOS[0]
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    payload = corrected_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    assert payload["options"]["seed"] == corrected_seed(record_id, 2)
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert "unfinished Bengali-script fragment" in payload["prompt"]
