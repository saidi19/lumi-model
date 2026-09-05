from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_CANDIDATE,
    load_exact_owner_sample,
    load_owner_review_plan,
    sha256_value,
    validate_owner_decisions,
    validate_owner_review_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-aya-bengali-owner-review-v0.1.json"
RECEIPT = ROOT / "evidence" / "lumi-4b-sft-aya-bengali-owner-review-v0.1.json"


def _sample() -> list[dict]:
    return [{"record_id": f"aya-{index:02d}"} for index in range(32)]


def _decisions(sample: list[dict], *, reviewer: str = "owner-01") -> dict:
    return {
        "bindings": {
            "attribution_file_sha256": EXPECTED_CANDIDATE["attribution_file_sha256"],
            "candidate_file_sha256": EXPECTED_CANDIDATE["candidate_file_sha256"],
            "preflight_receipt_sha256": EXPECTED_CANDIDATE["preflight_receipt_sha256"],
            "sample_record_ids_sha256": EXPECTED_CANDIDATE["sample_record_ids_sha256"],
        },
        "records": [
            {"decision": "accept", "note": "", "record_id": row["record_id"]} for row in sample
        ],
        "reviewer_alias": reviewer,
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_decisions_make_only_a_raw_text_free_nonpromoting_receipt() -> None:
    sample = _sample()
    receipt = validate_owner_decisions(
        _decisions(sample),
        sample_rows=sample,
        decision_file_sha256="a" * 64,
        plan_file_sha256="b" * 64,
        reviewed_at="2026-08-06T08:30:00Z",
    )
    validate_owner_review_receipt(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    assert receipt["receipt_sha256"] == sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


@pytest.mark.parametrize("reviewer", ["ChatGPT", "owner-claude", "codex-reviewer"])
def test_ai_alias_cannot_claim_human_owner_review(reviewer: str) -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="cannot claim Aya owner review"):
        validate_owner_decisions(
            _decisions(sample, reviewer=reviewer),
            sample_rows=sample,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
            reviewed_at="2026-08-06T08:30:00Z",
        )


def test_decisions_must_cover_every_sample_row() -> None:
    sample = _sample()
    decisions = _decisions(sample)
    decisions["records"].pop()
    with pytest.raises(ValueError, match="cover every sampled row"):
        validate_owner_decisions(
            decisions,
            sample_rows=sample,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
            reviewed_at="2026-08-06T08:30:00Z",
        )


def test_actual_plan_rebuilds_exact_sample() -> None:
    if not PLAN.exists():
        pytest.skip("exact Aya owner-review plan is not materialized")
    plan = load_owner_review_plan(PLAN, repository_root=ROOT)
    report = json.loads(
        (ROOT / plan["bindings"]["materialization_report"]["path"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    sample = load_exact_owner_sample(
        (ROOT / plan["bindings"]["accepted_records"]["path"]).resolve(),
        materialization_report=report,
    )
    assert len(sample) == 32


def test_actual_owner_receipt_is_complete_and_nonpromoting() -> None:
    if not RECEIPT.exists():
        pytest.skip("Aya owner-review receipt is not finalized")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    validate_owner_review_receipt(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["all_sampled_rows_accepted"] is True
    assert receipt["status"] == ("owner_quality_sample_passed_source_promotion_still_separate")
