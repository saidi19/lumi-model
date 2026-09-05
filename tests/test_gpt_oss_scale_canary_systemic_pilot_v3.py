from __future__ import annotations

import json

import pytest

from lumi import posttraining_first_party_scale_canary as scale
from lumi import posttraining_gpt_oss_scale_canary_systemic_pilot_v3 as pilot


def _response(text: str, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": json.dumps({"assistant_response": text})},
            }
        ]
    }


def test_schedule_is_exactly_four_train_one_validation_per_cell() -> None:
    schedule = pilot.build_schedule()
    assert len(schedule) == 100
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            cell = [
                slot for slot in schedule
                if slot.task == task and slot.language_mode == language
            ]
            assert len(cell) == 5
            assert sum(slot.split == "train" for slot in cell) == 4
            assert sum(slot.split == "validation" for slot in cell) == 1


def test_payload_generates_only_assistant_response_and_uses_2048_ceiling() -> None:
    slot = pilot.build_schedule()[0]
    payload = pilot.chat_completion_payload(slot, 1)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert payload["max_tokens"] == 2_048
    assert set(schema["properties"]) == {"assistant_response"}
    prompt = json.dumps(payload["messages"], ensure_ascii=False)
    assert "Attempt:" not in prompt
    assert "Record ID:" not in prompt


def test_parser_attaches_frozen_fields_locally() -> None:
    slot = next(
        item for item in pilot.build_schedule()
        if item.task == "factual_qa" and item.language_mode == "english"
    )
    text = (
        f"{pilot._assistant_anchor(slot)} {slot.expected_answer}. "
        "This directly answers the question with the requested factual value."
    )
    row = pilot._parse_candidate(
        _response(text), slot=slot, request_sha256="a" * 64,
        seed=1, attempt_number=1,
    )
    assert row["record_id"] == slot.record_id
    assert row["messages"][0]["content"] == slot.user_prompt
    assert row["provenance"]["prompt_contract"] == pilot.PROMPT_CONTRACT


def test_parser_rejects_extra_generated_envelope_fields() -> None:
    slot = pilot.build_schedule()[0]
    response = _response("A sufficiently long natural answer for schema validation.")
    content = json.loads(response["choices"][0]["message"]["content"])
    content["user_prompt"] = slot.user_prompt
    response["choices"][0]["message"]["content"] = json.dumps(content)
    with pytest.raises(ValueError, match="response_only_envelope_changed"):
        pilot._parse_candidate(
            response, slot=slot, request_sha256="b" * 64,
            seed=2, attempt_number=1,
        )


def test_writing_quality_hook_is_non_recursive_and_restored() -> None:
    slot = next(item for item in pilot.build_schedule() if item.task == "writing_summarization")
    original = scale._validate_response_quality
    response = (
        "This concise summary presents the main event, its impact, and the resolution "
        "without copying the original wording or repeating any sentence."
    )
    with pilot._task_aware_quality(slot):
        scale._validate_response_quality(slot.user_prompt, response)
    assert scale._validate_response_quality is original


def test_mathematical_stop_detects_cell_below_four_possible() -> None:
    schedule = pilot.build_schedule()
    cell = [
        slot for slot in schedule
        if slot.task == scale.TASKS[0] and slot.language_mode == scale.LANGUAGE_MODES[0]
    ]
    attempts = []
    for slot in cell[:2]:
        attempts.extend(
            [
                {"record_id": slot.record_id, "attempt_number": 1},
                {"record_id": slot.record_id, "attempt_number": 2},
            ]
        )
    assert pilot._mathematically_ineligible(schedule, [], attempts)
