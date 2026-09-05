from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_dmath_curation import sha256_value
from lumi.posttraining_dmath_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    load_exact_owner_sample,
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
    validate_owner_review_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-dmath-owner-review-v0.1.json"
RECEIPT_PATH = ROOT / "evidence/lumi-4b-sft-dmath-owner-review-v0.1.json"


def _rows() -> list[dict]:
    return [
        {
            "record_id": f"dmath-train-{index:06d}",
            "messages": [
                {"content": f"Question {index}", "role": "user"},
                {"content": f"The answer is {index}.", "role": "assistant"},
            ],
        }
        for index in range(32)
    ]


def _decisions(rows: list[dict], *, reviewer: str = "owner-afnan") -> dict:
    return {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in rows
        ],
        "reviewed_at": "2026-08-06T14:32:03Z",
        "reviewer_id": reviewer,
        "sample_sha256": sha256_value(rows),
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_owner_decisions_make_only_a_raw_text_free_non_promoting_receipt() -> None:
    rows = _rows()
    receipt = validate_owner_decisions(
        _decisions(rows),
        sample_rows=rows,
        decision_file_sha256="a" * 64,
        plan_file_sha256="b" * 64,
    )
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    validate_owner_review_receipt(receipt)
    assert rows[0]["messages"][0]["content"] not in json.dumps(receipt)


@pytest.mark.parametrize("reviewer", ["ChatGPT", "owner-claude", "codex-reviewer"])
def test_ai_alias_cannot_claim_human_owner_review(reviewer: str) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="cannot claim DMath owner review"):
        validate_owner_decisions(
            _decisions(rows, reviewer=reviewer),
            sample_rows=rows,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_owner_decisions_must_cover_every_sampled_row() -> None:
    rows = _rows()
    decisions = _decisions(rows)
    decisions["decisions"].pop()
    with pytest.raises(ValueError, match="cover every sampled row"):
        validate_owner_decisions(
            decisions,
            sample_rows=rows,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_exact_plan_and_sample_are_bound_when_available() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("exact DMath owner-review plan is not materialized")
    plan = load_owner_review_plan(PLAN_PATH, repository_root=ROOT)
    report = json.loads(
        resolve_plan_binding(ROOT, plan["bindings"]["curation_report"])
        .read_text(encoding="utf-8")
    )
    verification = json.loads(
        resolve_plan_binding(ROOT, plan["bindings"]["curation_verification"])
        .read_text(encoding="utf-8")
    )
    rows = load_exact_owner_sample(
        resolve_plan_binding(ROOT, plan["bindings"]["owner_sample"]),
        curation_report=report,
        verification=verification,
    )
    assert len(rows) == 32


def test_actual_owner_receipt_is_complete_and_non_promoting_when_available() -> None:
    if not RECEIPT_PATH.exists():
        pytest.skip("DMath owner-review receipt has not been finalized")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    validate_owner_review_receipt(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["all_sampled_rows_accepted"] is True
    assert receipt["status"] == (
        "owner_quality_sample_passed_source_promotion_still_separate"
    )
