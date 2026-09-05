from __future__ import annotations

import copy
import hashlib

import pytest

import lumi.posttraining_candidate_mix_reaudit_v17 as contract
from lumi.posttraining_ai_assisted_gap_closure_canary_v1 import render_records


def test_authorization_and_exact_projection() -> None:
    assert len(contract.EXPECTED_AUTHORIZATION_TEXT.encode()) == contract.EXPECTED_AUTHORIZATION_BYTES
    assert hashlib.sha256(contract.EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest() == contract.EXPECTED_AUTHORIZATION_SHA256
    assert contract.EXPECTED_INPUT_RECORDS == 314_780 + 1_000
    assert contract.EXPECTED_INPUT_TRAIN_TOKENS == 10_610_600 + 86_712
    assert contract.EXPECTED_INPUT_VALIDATION_TOKENS == 628_604 + 21_592
    assert contract.EXPECTED_LIMITS["maximum_total_candidate_row_reads"] == 2 * contract.EXPECTED_INPUT_RECORDS


def test_ai_gap_exact_row_contract_and_language_mode_counts() -> None:
    rows = render_records()
    result = contract._validate_ai_gap_rows(rows)
    assert result["language_mode_field_validated_rows"] == 1_000
    assert result["human_authorship_claimed_rows"] == 0
    assert result["human_review_completed_rows"] == 0
    assert result["model_assisted_rows"] == 1_000
    assert result["runtime_inference_call_rows"] == 0
    assert result["language_mode_records"] == {"banglish": 666, "natural_code_switching": 334}


def test_ai_gap_language_mode_mismatch_is_rejected() -> None:
    rows = render_records()
    rows[0]["language_mode"] = (
        "natural_code_switching" if rows[0]["language_mode"] == "banglish" else "banglish"
    )
    with pytest.raises(ValueError, match="row contract"):
        contract._validate_ai_gap_rows(rows)


def test_ai_gap_malformed_record_id_is_rejected_without_parser_leak() -> None:
    rows = render_records()
    rows[0]["record_id"] = "malformed-no-numeric-suffix"
    with pytest.raises(ValueError, match="row contract"):
        contract._validate_ai_gap_rows(rows)


def test_ai_gap_mode_resolver_uses_exact_top_level_field() -> None:
    row = render_records()[0]
    assert contract._mode_from_row(row, contract.AI_GAP_SOURCE["language_policy"]) == row["language_mode"]
    changed = copy.deepcopy(row)
    changed["language"] = "en"
    with pytest.raises(ValueError, match="exact AI gap"):
        contract._mode_from_row(changed, contract.AI_GAP_SOURCE["language_policy"])


def test_near_duplicate_audit_detects_cross_source_copy() -> None:
    rows = render_records()[:2]
    rows[1] = copy.deepcopy(rows[1])
    rows[1]["messages"] = copy.deepcopy(rows[0]["messages"])
    near = contract.v2._near_duplicate_audit(rows, policy=contract.EXPECTED_NEAR_DUPLICATE_POLICY)
    assert near["multi_record_components"] == 1


def test_raw_text_evidence_guard() -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        contract._assert_raw_text_free({"nested": {"messages": []}})
    contract._assert_raw_text_free({"accepted_records": 315_780})
