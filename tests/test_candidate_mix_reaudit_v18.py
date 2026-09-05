from __future__ import annotations

import copy
import hashlib

import pytest

import lumi.posttraining_candidate_mix_reaudit_v18 as contract
from lumi.posttraining_ai_assisted_gap_closure_scaled_v2 import render_records


def test_authorization_and_exact_projection() -> None:
    assert len(contract.EXPECTED_AUTHORIZATION_TEXT.encode()) == contract.EXPECTED_AUTHORIZATION_BYTES
    assert hashlib.sha256(contract.EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest() == contract.EXPECTED_AUTHORIZATION_SHA256
    assert contract.EXPECTED_INPUT_RECORDS == 315_780 + 13_420
    assert contract.EXPECTED_INPUT_TRAIN_TOKENS == 10_697_312 + 1_157_905
    assert contract.EXPECTED_INPUT_VALIDATION_TOKENS == 650_196 + 289_097
    assert contract.EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == 2 * contract.EXPECTED_INPUT_RECORDS


def test_scaled_exact_row_contract_and_allocations() -> None:
    rows = render_records()
    result = contract._validate_scaled_rows(rows)
    assert result["language_mode_field_validated_rows"] == 13_420
    assert result["human_authorship_claimed_rows"] == 0
    assert result["human_review_completed_rows"] == 0
    assert result["model_assisted_rows"] == 13_420
    assert result["split_records"] == {"train": 10_736, "validation": 2_684}
    assert result["language_mode_records"] == {
        "banglish": 9_750,
        "natural_code_switching": 3_670,
    }


def test_scaled_language_mode_mismatch_is_rejected() -> None:
    rows = render_records()
    rows[0]["language_mode"] = (
        "natural_code_switching"
        if rows[0]["language_mode"] == "banglish"
        else "banglish"
    )
    with pytest.raises(ValueError, match="row contract"):
        contract._validate_scaled_rows(rows)


def test_scaled_malformed_record_id_is_rejected() -> None:
    rows = render_records()
    rows[0]["record_id"] = "malformed"
    with pytest.raises(ValueError, match="row contract"):
        contract._validate_scaled_rows(rows)


def test_near_duplicate_audit_detects_copy() -> None:
    rows = render_records()[:2]
    rows[1] = copy.deepcopy(rows[1])
    rows[1]["messages"] = copy.deepcopy(rows[0]["messages"])
    near = contract.v2._near_duplicate_audit(
        rows, policy=contract.EXPECTED_NEAR_DUPLICATE_POLICY
    )
    assert near["multi_record_components"] == 1


def test_raw_text_evidence_guard() -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        contract._assert_raw_text_free({"nested": {"messages": []}})
    contract._assert_raw_text_free({"accepted_records": 329_200})
