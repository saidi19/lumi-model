from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_deepmind_scale_generation import sha256_value
from lumi.posttraining_deepmind_scale_owner_review import (
    EXPECTED_AUTHORIZATION,
    validate_owner_decisions,
    validate_owner_review_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "evidence/lumi-4b-sft-deepmind-scale-owner-review-v0.1.json"


def _sample() -> list[dict[str, str]]:
    return [{"record_id": f"deepmind-owner-sample-{index:02d}"} for index in range(32)]


def _plan() -> dict:
    bindings = {
        name: {"sha256": character * 64}
        for name, character in (
            ("accepted_records", "a"),
            ("owner_sample", "b"),
            ("scale_report", "c"),
            ("scale_verification", "d"),
        )
    }
    bindings["owner_sample"]["internal_sha256"] = "e" * 64
    bindings["scale_report"]["internal_sha256"] = "f" * 64
    bindings["scale_verification"]["internal_sha256"] = "1" * 64
    return {
        "bindings": bindings,
        "candidate": {
            "records": 236_698,
            "train_records": 224_688,
            "train_supervised_tokens": 2_221_113,
            "validation_records": 12_010,
            "validation_supervised_tokens": 118_053,
        },
    }


def _decisions(sample: list[dict[str, str]], *, accepted: bool = True) -> dict:
    return {
        "decisions": [
            {"accepted": accepted, "note": "", "record_id": row["record_id"]}
            for row in sample
        ],
        "reviewed_at": "2026-08-07T06:48:30Z",
        "reviewer_id": "owner-afnan",
        "sample_sha256": sha256_value(sample),
        "schema_version": "lumi.ai-assisted-owner-decisions.v0.3",
    }


def test_owner_review_receipt_is_raw_free_and_non_promoting() -> None:
    sample = _sample()
    receipt = validate_owner_decisions(
        _decisions(sample),
        sample_rows=sample,
        plan=_plan(),
        decision_file_sha256="2" * 64,
        plan_file_sha256="3" * 64,
    )
    validate_owner_review_receipt(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["authorization"] == EXPECTED_AUTHORIZATION
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False


def test_owner_review_requires_exact_sample_coverage() -> None:
    sample = _sample()
    decisions = _decisions(sample)
    decisions["decisions"][0]["record_id"] = "unknown-record"
    with pytest.raises(ValueError, match="unknown record ID"):
        validate_owner_decisions(
            decisions,
            sample_rows=sample,
            plan=_plan(),
            decision_file_sha256="2" * 64,
            plan_file_sha256="3" * 64,
        )


def test_finalized_owner_receipt_validates_when_present() -> None:
    if not RECEIPT.is_file():
        pytest.skip("DeepMind scale owner-review receipt is not finalized")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    validate_owner_review_receipt(receipt)
