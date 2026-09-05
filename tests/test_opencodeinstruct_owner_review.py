from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_opencodeinstruct_curation import sha256_value
from lumi.posttraining_opencodeinstruct_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_AUTHORIZATION,
    EXPECTED_CANDIDATE,
    RECEIPT_SCHEMA_VERSION,
    load_owner_review_plan,
    validate_owner_decisions,
    validate_owner_review_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-opencodeinstruct-owner-review-v0.1.json"
RECEIPT = ROOT / "evidence/lumi-4b-sft-opencodeinstruct-owner-review-v0.1.json"


def _sample() -> list[dict[str, str]]:
    return [{"record_id": f"opencode-owner-{index:02d}"} for index in range(32)]


def _plan() -> dict:
    bindings = {
        name: {"sha256": character * 64}
        for name, character in (
            ("accepted_records", "a"),
            ("owner_sample", "b"),
            ("retry_report", "c"),
            ("retry_verification", "d"),
        )
    }
    bindings["retry_report"]["internal_sha256"] = "e" * 64
    bindings["retry_verification"]["internal_sha256"] = "f" * 64
    return {"bindings": bindings, "candidate": EXPECTED_CANDIDATE}


def _decisions(rows: list[dict[str, str]], *, reviewer: str = "owner-afnan") -> dict:
    return {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in rows
        ],
        "reviewed_at": "2026-08-07T10:06:20Z",
        "reviewer_id": reviewer,
        "sample_sha256": sha256_value(rows),
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_owner_review_receipt_is_raw_free_and_non_promoting() -> None:
    rows = _sample()
    receipt = validate_owner_decisions(
        _decisions(rows),
        sample_rows=rows,
        plan=_plan(),
        decision_file_sha256="1" * 64,
        plan_file_sha256="2" * 64,
    )
    validate_owner_review_receipt(receipt)
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["authorization"] == EXPECTED_AUTHORIZATION
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    assert rows[0]["record_id"] not in json.dumps(receipt)


@pytest.mark.parametrize("reviewer", ["ChatGPT", "owner-claude", "codex-reviewer"])
def test_ai_alias_cannot_claim_owner_review(reviewer: str) -> None:
    rows = _sample()
    with pytest.raises(ValueError, match="AI alias"):
        validate_owner_decisions(
            _decisions(rows, reviewer=reviewer),
            sample_rows=rows,
            plan=_plan(),
            decision_file_sha256="1" * 64,
            plan_file_sha256="2" * 64,
        )


def test_decisions_must_cover_exact_sample() -> None:
    rows = _sample()
    decisions = _decisions(rows)
    decisions["decisions"][0]["record_id"] = "unknown-record"
    with pytest.raises(ValueError, match="unknown record ID"):
        validate_owner_decisions(
            decisions,
            sample_rows=rows,
            plan=_plan(),
            decision_file_sha256="1" * 64,
            plan_file_sha256="2" * 64,
        )


def test_exact_plan_and_receipt_validate_when_present() -> None:
    if not PLAN.is_file():
        pytest.skip("OpenCodeInstruct owner-review plan is not materialized")
    load_owner_review_plan(PLAN, repository_root=ROOT)
    if not RECEIPT.is_file():
        pytest.skip("OpenCodeInstruct owner-review receipt is not finalized")
    validate_owner_review_receipt(json.loads(RECEIPT.read_text(encoding="utf-8")))
