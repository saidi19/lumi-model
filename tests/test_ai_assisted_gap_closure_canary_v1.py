from __future__ import annotations

from collections import Counter

import pytest

from lumi.posttraining_ai_assisted_gap_closure_canary_v1 import (
    ALLOCATIONS,
    AUTHORIZATION_BYTES,
    AUTHORIZATION_SHA256,
    AUTHORIZATION_TEXT,
    EXPECTED_RECORDS,
    _assert_raw_text_free,
    _language_contract,
    build_schedule,
    render_records,
    schedule_sha256,
    sha256_value,
)


def test_authorization_binding_is_exact() -> None:
    assert len(AUTHORIZATION_TEXT.encode()) == AUTHORIZATION_BYTES
    import hashlib

    assert hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() == AUTHORIZATION_SHA256


def test_schedule_has_exact_split_and_cells() -> None:
    schedule = build_schedule()
    assert len(schedule) == EXPECTED_RECORDS
    assert Counter(slot.split for slot in schedule) == {"train": 800, "validation": 200}
    expected = {
        (task, mode): train + validation
        for task, mode, train, validation in ALLOCATIONS
    }
    assert Counter((slot.task_category, slot.language_mode) for slot in schedule) == expected


def test_schedule_and_renderer_are_deterministic() -> None:
    first = build_schedule()
    second = build_schedule()
    assert schedule_sha256(first) == schedule_sha256(second)
    assert sha256_value(render_records(first)) == sha256_value(render_records(second))


def test_all_rows_are_truthfully_ai_assisted_and_not_human_authored() -> None:
    rows = render_records()
    assert len(rows) == 1_000
    assert all(row["language"] == "multilingual" for row in rows)
    assert all(row["provenance"]["model_assistance_used"] is True for row in rows)
    assert all(row["provenance"]["human_written_from_scratch"] is False for row in rows)
    assert all(row["owner_review"]["human_review_completed"] is False for row in rows)


def test_message_sequences_and_tool_trajectories_are_exact() -> None:
    rows = render_records()
    for row in rows:
        roles = [message["role"] for message in row["messages"]]
        expected = (
            ["user", "tool", "tool_result", "assistant"]
            if row["task_category"] == "tool_api"
            else ["user", "assistant"]
        )
        assert roles == expected


def test_language_contract_rejects_crossed_modes() -> None:
    with pytest.raises(ValueError, match="Bengali script"):
        _language_contract(
            [
                {"role": "user", "content": "ami eta koro"},
                {"role": "assistant", "content": "আমি এটা করি"},
            ],
            "banglish",
        )
    with pytest.raises(ValueError, match="Bengali and Latin"):
        _language_contract(
            [
                {"role": "user", "content": "ami eta koro"},
                {"role": "assistant", "content": "eta thik hobe"},
            ],
            "natural_code_switching",
        )


def test_evidence_guard_rejects_raw_candidate_keys() -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        _assert_raw_text_free({"audit": {"messages": []}})
    _assert_raw_text_free({"audit": {"accepted_records": 1000}})

