from __future__ import annotations

import hashlib
from collections import Counter

import pytest

import lumi.posttraining_ai_assisted_gap_closure_scaled_v2 as contract


def test_authorization_binding_is_exact() -> None:
    assert len(contract.AUTHORIZATION_TEXT.encode()) == contract.AUTHORIZATION_BYTES
    assert hashlib.sha256(contract.AUTHORIZATION_TEXT.encode()).hexdigest() == contract.AUTHORIZATION_SHA256


def test_schedule_has_exact_split_modes_and_cells() -> None:
    schedule = contract.build_schedule()
    assert len(schedule) == 13_420
    assert Counter(slot.split for slot in schedule) == {"train": 10_736, "validation": 2_684}
    assert Counter(slot.language_mode for slot in schedule if slot.split == "train") == {
        "banglish": 7_800,
        "natural_code_switching": 2_936,
    }
    expected = {
        (task, mode, "train"): train
        for task, mode, train, _validation in contract.ALLOCATIONS
    } | {
        (task, mode, "validation"): validation
        for task, mode, _train, validation in contract.ALLOCATIONS
    }
    assert Counter((slot.task_category, slot.language_mode, slot.split) for slot in schedule) == expected


def test_unique_reference_serial_is_global_schedule_index() -> None:
    schedule = contract.build_schedule()
    serials = [contract._details(slot)["serial"] for slot in schedule]
    assert serials == list(range(100_000, 113_420))
    assert len({contract._details(slot)["reference"] for slot in schedule}) == len(schedule)


def test_renderer_is_deterministic_truthful_and_disjoint_from_canary() -> None:
    schedule = contract.build_schedule()
    assert contract.schedule_sha256(schedule) == contract.schedule_sha256(contract.build_schedule())
    rows = contract.render_records(schedule)
    assert len(rows) == 13_420
    assert all(row["source"]["source_id"] == contract.SOURCE_ID for row in rows)
    assert all(row["provenance"]["model_assistance_used"] is True for row in rows)
    assert all(row["provenance"]["human_written_from_scratch"] is False for row in rows)
    assert all(row["owner_review"]["human_review_completed"] is False for row in rows)
    old_ids = {row["record_id"] for row in contract.canary.render_records()}
    assert old_ids.isdisjoint({row["record_id"] for row in rows})


def test_expected_frozen_token_ledgers_close_current_gaps() -> None:
    assert contract.EXPECTED_TRAIN_SUPERVISED_TOKENS == sum(contract.EXPECTED_TRAIN_TOKENS_BY_MODE.values())
    assert contract.EXPECTED_TRAIN_SUPERVISED_TOKENS == sum(contract.EXPECTED_TRAIN_TOKENS_BY_TASK.values())
    assert contract.EXPECTED_TRAIN_TOKENS_BY_MODE["banglish"] >= contract.MINIMUM_BANGLISH_TRAIN_SUPERVISED_TOKENS
    assert contract.EXPECTED_TRAIN_TOKENS_BY_MODE["natural_code_switching"] >= contract.MINIMUM_NCS_TRAIN_SUPERVISED_TOKENS
    assert contract.EXPECTED_TRAIN_TOKENS_BY_TASK["factual_qa"] >= contract.MINIMUM_FACTUAL_TRAIN_SUPERVISED_TOKENS


def test_language_contract_and_raw_evidence_guard() -> None:
    sample = contract.build_schedule()[0]
    messages, _labels = contract.render_messages(sample)
    contract.canary._language_contract(messages, sample.language_mode)
    with pytest.raises(ValueError, match="raw candidate key"):
        contract._assert_raw_text_free({"nested": {"messages": []}})
    contract._assert_raw_text_free({"accepted_records": 13_420})
