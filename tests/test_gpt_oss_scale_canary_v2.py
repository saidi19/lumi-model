from __future__ import annotations

from collections import Counter

from lumi.posttraining_gpt_oss_scale_canary_v2 import (
    MAX_ATTEMPTS_PER_RECORD,
    _terminal_map,
    build_schedule,
    chat_completion_payload,
    schedule_sha256,
    seed_for_attempt,
)


def test_schedule_is_exact_1000_and_800_200() -> None:
    schedule = build_schedule()
    assert len(schedule) == 1_000
    assert Counter(slot.split for slot in schedule) == {"train": 800, "validation": 200}
    assert len({slot.record_id for slot in schedule}) == 1_000
    assert len(schedule_sha256()) == 64


def test_schedule_includes_all_20_cells_and_tool_api() -> None:
    schedule = build_schedule()
    cells = Counter((slot.task, slot.language_mode) for slot in schedule)
    assert len(cells) == 20
    assert sum(count for (task, _), count in cells.items() if task == "tool_api") == 266


def test_attempt_seeds_and_requests_are_distinct() -> None:
    slot = build_schedule()[0]
    assert MAX_ATTEMPTS_PER_RECORD == 2
    assert seed_for_attempt(slot.record_id, 1) != seed_for_attempt(slot.record_id, 2)
    assert chat_completion_payload(slot, 1) != chat_completion_payload(slot, 2)


def test_json_schema_binds_tool_fields_and_exact_user_prompt() -> None:
    tool = next(slot for slot in build_schedule() if slot.task == "tool_api")
    payload = chat_completion_payload(tool, 1)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["user_prompt"]["const"] == tool.user_prompt
    assert schema["properties"]["tool_call"]["const"] == {
        "name": tool.tool_name,
        "arguments": tool.tool_arguments,
    }
    assert schema["properties"]["tool_result"]["const"] == tool.tool_result


def test_prompt_never_requests_reasoning_retention() -> None:
    payload = chat_completion_payload(build_schedule()[0], 1)
    serialized = str(payload).lower()
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1_536
    assert "return final json only" in serialized


def test_single_failed_attempt_is_terminal_after_global_retry_exhaustion() -> None:
    schedule = build_schedule()[:2]
    attempts = [
        {
            "accepted": False,
            "attempt_number": 1,
            "elapsed_seconds": 1.0,
            "record_id": slot.record_id,
            "validation_error": "bounded_unresolved_or_retry_exhausted",
        }
        for slot in schedule
    ]
    terminals = _terminal_map(schedule, [], [], attempts)
    assert set(terminals) == {slot.record_id for slot in schedule}
    assert all(item["accepted"] is False for item in terminals.values())
