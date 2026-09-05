from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_openr1_full_run import sha256_value
from lumi.posttraining_openr1_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_CANDIDATE,
    RECEIPT_SCHEMA_VERSION,
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
    validate_owner_review_receipt,
    validate_owner_sample,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-sft-openr1-visible-final-owner-review-v0.1.json"
RECEIPT_PATH = (
    ROOT / "evidence" / "lumi-4b-sft-openr1-visible-final-owner-review-v0.1.json"
)


def _rows() -> list[dict]:
    rows = []
    for split in ("train", "validation"):
        for index in range(16):
            rows.append(
                {
                    "language_mode": "english",
                    "messages": [
                        {"content": f"Question {split} {index}", "role": "user"},
                        {"content": f"Answer ${index} + 1$", "role": "assistant"},
                    ],
                    "record_id": f"visible-{split}-{index}",
                    "skill_id": "reasoning_math_science_visible_final",
                    "split": split,
                }
            )
    return rows


def _decisions(rows: list[dict], reviewer: str = "owner-afnan") -> dict:
    return {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in rows
        ],
        "reviewed_at": "2026-08-06T11:18:24Z",
        "reviewer_id": reviewer,
        "sample_sha256": sha256_value(rows),
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_decisions_make_only_a_raw_free_nonpromoting_receipt() -> None:
    rows = _rows()
    receipt = validate_owner_decisions(
        _decisions(rows),
        sample_rows=rows,
        decision_file_sha256="a" * 64,
        plan_file_sha256="b" * 64,
    )
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    validate_owner_review_receipt(receipt)
    assert rows[0]["messages"][0]["content"] not in json.dumps(receipt)


@pytest.mark.parametrize("reviewer", ["ChatGPT", "owner-claude", "codex-reviewer"])
def test_ai_alias_cannot_claim_owner_review(reviewer: str) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="cannot claim OpenR1 owner review"):
        validate_owner_decisions(
            _decisions(rows, reviewer),
            sample_rows=rows,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_decisions_must_cover_every_sampled_row() -> None:
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


def test_exact_plan_sample_and_receipt_when_materialized() -> None:
    if not PLAN_PATH.exists() or not RECEIPT_PATH.exists():
        pytest.skip("exact OpenR1 owner-review artifacts are not materialized")
    plan = load_owner_review_plan(PLAN_PATH, repository_root=ROOT)
    sample = [
        json.loads(line)
        for line in resolve_plan_binding(ROOT, plan["bindings"]["owner_sample"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    report = json.loads(
        resolve_plan_binding(ROOT, plan["bindings"]["visible_final_report"])
        .read_text(encoding="utf-8")
    )
    verification = json.loads(
        resolve_plan_binding(ROOT, plan["bindings"]["visible_final_verification"])
        .read_text(encoding="utf-8")
    )
    coverage = validate_owner_sample(sample, report=report, verification=verification)
    assert coverage == {
        "records": 32,
        "sample_sha256": EXPECTED_CANDIDATE["owner_sample_internal_sha256"],
        "train_records": 16,
        "validation_records": 16,
        "visible_think_tag_records": 0,
    }
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    validate_owner_review_receipt(receipt)
