from __future__ import annotations

import hashlib
import json

import pytest

from lumi.posttraining_banglabridge_expanded_delta_owner_review_v1 import (
    EXPECTED_AUTHORIZATION,
    EXPECTED_BINDING_SHA256,
    EXPECTED_DECISIONS_CONTENT_SHA256,
    EXPECTED_DECISIONS_FILE_SHA256,
    EXPECTED_OWNER_REQUEST,
    EXPECTED_OWNER_REQUEST_SHA256,
    EXPECTED_REVIEWED_IDS_SHA256,
    EXPECTED_SAMPLE_CONTENT_SHA256,
    EXPECTED_SAMPLE_FILE_SHA256,
    build_verification,
    sha256_value,
    validate_receipt,
    validate_verification,
)


def _receipt() -> dict:
    value = {
        "acceptance_rate": 1.0,
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_file_sha256": EXPECTED_BINDING_SHA256["candidate"],
            "decisions_content_sha256": EXPECTED_DECISIONS_CONTENT_SHA256,
            "decisions_file_sha256": EXPECTED_DECISIONS_FILE_SHA256,
            "delta_ledger_file_sha256": EXPECTED_BINDING_SHA256["delta_ledger"],
            "owner_request_sha256": EXPECTED_OWNER_REQUEST_SHA256,
            "owner_sample_content_sha256": EXPECTED_SAMPLE_CONTENT_SHA256,
            "owner_sample_file_sha256": EXPECTED_SAMPLE_FILE_SHA256,
            "plan_file_sha256": "3" * 64,
            "recovery_plan_file_sha256": EXPECTED_BINDING_SHA256["recovery_plan"],
            "recovery_report_file_sha256": EXPECTED_BINDING_SHA256[
                "recovery_report"
            ],
            "recovery_report_internal_sha256": (
                "8dab0c08ff0e51a343624929c6f2c6dc86282bd3bdb587aa1c563abea5d90730"
            ),
            "recovery_verification_file_sha256": EXPECTED_BINDING_SHA256[
                "recovery_verification"
            ],
            "recovery_verification_internal_sha256": (
                "7bd7fe589e529de8a8a2cca3a6c46a2e36940f4c81e76979aedafac19f758691"
            ),
            "reviewed_record_ids_sha256": EXPECTED_REVIEWED_IDS_SHA256,
        },
        "candidate": {
            "records": 561,
            "train_records": 544,
            "train_supervised_tokens": 250_441,
            "validation_records": 17,
            "validation_supervised_tokens": 4_960,
        },
        "delta": {"records": 445, "train_supervised_tokens": 200_685},
        "next_gate": {
            "action": "separate_bounded_registry_successor_promotion_then_full_mix_reaudit",
            "dataset_seal_or_sft_authorized": False,
            "owner_review_complete": True,
            "registry_change_authorized": False,
            "separate_authorization_required": True,
        },
        "owner_review_claim": (
            "banglabridge_cc_by_4_0_source_derived_or_deterministically_transformed_"
            "owner_spot_checked_32_row_level_human_authorship_not_established"
        ),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "rejected_record_ids": [],
        "rejected_records": 0,
        "reviewed_at": "2026-08-13T13:35:40Z",
        "reviewer_alias": "owner-afnan",
        "sample_records": 32,
        "schema_version": (
            "lumi.sft-banglabridge-expanded-delta-owner-review-receipt.v0.1"
        ),
        "status": "owner_accepted_all_32_banglabridge_expanded_delta_rows",
        "threshold": {"minimum_accepts": 29, "satisfied": True},
    }
    value["receipt_sha256"] = sha256_value(value)
    return value


def test_owner_request_and_two_hash_domains_are_exact() -> None:
    assert hashlib.sha256(EXPECTED_OWNER_REQUEST.encode()).hexdigest() == (
        EXPECTED_OWNER_REQUEST_SHA256
    )
    assert EXPECTED_SAMPLE_CONTENT_SHA256 != EXPECTED_SAMPLE_FILE_SHA256
    assert EXPECTED_DECISIONS_CONTENT_SHA256 != EXPECTED_DECISIONS_FILE_SHA256


def test_receipt_is_raw_text_free_and_nonproduction() -> None:
    receipt = _receipt()
    validate_receipt(receipt)
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert '"messages"' not in serialized
    assert receipt["next_gate"]["separate_authorization_required"] is True
    assert receipt["production_eligible"] is False


def test_receipt_rejects_inaccurate_ai_generated_claim() -> None:
    receipt = _receipt()
    receipt["owner_review_claim"] = "ai_generated_plus_owner_reviewed"
    receipt["receipt_sha256"] = sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="receipt changed"):
        validate_receipt(receipt)


def test_verification_binds_receipt_and_forbidden_actions() -> None:
    receipt = _receipt()
    verification = build_verification(receipt, receipt_file_sha256="a" * 64)
    validate_verification(verification)
    assert all(verification["checks"].values())
    assert verification["production_eligible"] is False
