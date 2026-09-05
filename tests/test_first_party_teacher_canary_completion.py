from __future__ import annotations

from lumi.posttraining_first_party_teacher_canary import SCENARIOS
from lumi.posttraining_first_party_teacher_canary_v3 import (
    CODE_SWITCH_TERMS,
    alternate_seed,
    completion_payload,
    completion_prompt,
)


def test_every_scenario_has_a_code_switch_term() -> None:
    assert set(CODE_SWITCH_TERMS) == {scenario.scenario_id for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        prompt = completion_prompt(scenario, "natural_code_switching")
        assert CODE_SWITCH_TERMS[scenario.scenario_id] in prompt
        assert "Each field must independently contain Bengali script" in prompt


def test_alternate_seeds_are_deterministic_and_distinct() -> None:
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    seeds = [alternate_seed(record_id, attempt) for attempt in (1, 2, 3)]
    assert seeds == [alternate_seed(record_id, attempt) for attempt in (1, 2, 3)]
    assert len(set(seeds)) == 3


def test_completion_payload_uses_bounded_alternate_seed() -> None:
    scenario = SCENARIOS[0]
    record_id = "lumi-teacher-factual_qa-natural_code_switching-00"
    payload = completion_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    assert payload["options"]["seed"] == alternate_seed(record_id, 2)
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
