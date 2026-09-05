from __future__ import annotations

from lumi.posttraining_candidate_mix_reaudit_v14 import (
    BILLSUM_SOURCE,
    EXPECTED_EVALUATION_PROHIBITION,
    EXPECTED_INCLUDED_SOURCE_COUNT,
    EXPECTED_INPUT_RECORDS,
    EXPECTED_INPUT_TRAIN_TOKENS,
    EXPECTED_INPUT_VALIDATION_TOKENS,
    EXPECTED_LIMITS,
    EXPECTED_NEAR_DUPLICATE_POLICY,
)


def test_v014_input_projection_is_exact() -> None:
    assert EXPECTED_INCLUDED_SOURCE_COUNT == 17
    assert EXPECTED_INPUT_RECORDS == 309_838 + 4_433
    assert EXPECTED_INPUT_TRAIN_TOKENS == 9_553_473 + 849_991
    assert EXPECTED_INPUT_VALIDATION_TOKENS == 578_618 + 49_986
    assert EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == (
        2 * EXPECTED_INPUT_RECORDS
    )


def test_v014_near_duplicate_and_billsum_binding_are_frozen() -> None:
    assert EXPECTED_NEAR_DUPLICATE_POLICY["threshold"] == 0.9
    assert EXPECTED_NEAR_DUPLICATE_POLICY["num_perm"] == 128
    assert BILLSUM_SOURCE["source_id"] == "fiscalnote_billsum_writing_candidate"
    assert BILLSUM_SOURCE["family"] == "fiscalnote_billsum_family"


def test_v014_billsum_evaluation_prohibition_is_permanent() -> None:
    assert set(EXPECTED_EVALUATION_PROHIBITION) == {
        "billsum_ca_test_claimed_evaluation",
        "billsum_test_claimed_evaluation",
        "reason",
    }
    assert all(
        value == "permanently_prohibited"
        for key, value in EXPECTED_EVALUATION_PROHIBITION.items()
        if key != "reason"
    )
