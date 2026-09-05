from __future__ import annotations

from lumi.posttraining_candidate_mix_reaudit_v13 import (
    EXPECTED_INCLUDED_SOURCE_COUNT,
    EXPECTED_INPUT_RECORDS,
    EXPECTED_INPUT_TRAIN_TOKENS,
    EXPECTED_INPUT_VALIDATION_TOKENS,
    EXPECTED_LIMITS,
    EXPECTED_NEAR_DUPLICATE_POLICY,
    PROSOCIAL_SOURCE,
)


def test_v013_input_projection_is_exact() -> None:
    assert EXPECTED_INCLUDED_SOURCE_COUNT == 16
    assert EXPECTED_INPUT_RECORDS == 280_574 + 29_264
    assert EXPECTED_INPUT_TRAIN_TOKENS == 8_653_475 + 899_998
    assert EXPECTED_INPUT_VALIDATION_TOKENS == 518_619 + 59_999
    assert EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == (
        2 * EXPECTED_INPUT_RECORDS
    )


def test_v013_near_duplicate_and_prosocial_binding_are_frozen() -> None:
    assert EXPECTED_NEAR_DUPLICATE_POLICY["threshold"] == 0.9
    assert EXPECTED_NEAR_DUPLICATE_POLICY["num_perm"] == 128
    assert PROSOCIAL_SOURCE["source_id"] == "allenai_prosocial_dialog_safety_candidate"
    assert PROSOCIAL_SOURCE["family"] == "allenai_prosocial_family"
