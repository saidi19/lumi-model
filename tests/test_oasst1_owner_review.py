from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_oasst1_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_CANDIDATE,
    build_oasst1_owner_review_html,
    load_owner_review_plan,
    review_payload,
    sha256_value,
    validate_owner_decisions,
    validate_owner_sample,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-sft-oasst1-owner-review-v0.1.json"
RECEIPT_PATH = ROOT / "evidence" / "lumi-4b-sft-oasst1-owner-review-v0.1.json"

TASKS = (
    "coding",
    "conversation",
    "factual_qa",
    "writing_summarization",
)


def _rows() -> list[dict]:
    rows = []
    for split in ("train", "validation"):
        for task in TASKS:
            for index in range(4):
                rows.append(
                    {
                        "language": "en",
                        "messages": [
                            {"content": f"Question {split} {task} {index}", "role": "user"},
                            {
                                "content": f"Answer {split} {task} {index}",
                                "role": "assistant",
                            },
                        ],
                        "record_id": f"oasst1-{split}-{task}-{index}",
                        "schema_version": "lumi.sft.v0.1",
                        "source": {
                            "revision": EXPECTED_CANDIDATE["source_revision"],
                            "source_id": "openassistant_oasst1",
                        },
                        "split": split,
                        "task_category": task,
                    }
                )
    return rows


def _sample_inputs() -> tuple[list[dict], dict, dict, dict]:
    rows = _rows()
    report = {
        "owner_review": {
            "complete": False,
            "record_count": 32,
            "sample_record_ids_sha256": (
                "05c993c61f52217b60ce5cd8261e632edf6234ea43328b1903450a3356eaeb19"
            ),
            "status": "pending_owner_review",
        },
        "status": "curated_nonproduction_owner_review_pending",
    }
    report["report_sha256"] = sha256_value(report)
    verification = {
        "production_eligible": False,
        "status": "verified_nonproduction_owner_review_pending",
    }
    verification["verification_sha256"] = sha256_value(verification)
    plan = {
        "candidate": {
            "curation_report_internal_sha256": report["report_sha256"],
            "owner_sample_internal_sha256": sha256_value(rows),
            "verification_internal_sha256": verification["verification_sha256"],
        }
    }
    return rows, report, verification, plan


def _owner_decisions(rows: list[dict], *, reviewer: str = "owner-afnan") -> dict:
    return {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]}
            for row in rows
        ],
        "reviewed_at": "2026-08-06T03:00:00Z",
        "reviewer_id": reviewer,
        "sample_sha256": sha256_value(review_payload(rows)),
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_owner_sample_requires_exact_split_task_balance() -> None:
    rows, report, verification, plan = _sample_inputs()
    coverage = validate_owner_sample(
        rows,
        curation_report=report,
        verification=verification,
        plan=plan,
    )
    assert coverage == {
        "payload_sha256": sha256_value(review_payload(rows)),
        "records": 32,
        "records_per_split_task_pair": 4,
        "splits": 2,
        "task_categories": 4,
    }


def test_owner_review_html_is_private_and_uses_dedicated_schema() -> None:
    rows = _rows()
    html, payload_sha256 = build_oasst1_owner_review_html(rows)
    assert payload_sha256 == sha256_value(review_payload(rows))
    assert "Lumi OASST1 quality spot-check" in html
    assert DECISIONS_SCHEMA_VERSION in html
    assert "lumi-oasst1-owner-decisions-v0.1.json" in html
    assert "upstream human-authored OASST1" in html
    assert "permission to start SFT" in html
    assert "https://" not in html


def test_human_decisions_make_only_a_non_promoting_raw_free_receipt() -> None:
    rows = _rows()
    value = _owner_decisions(rows)
    receipt = validate_owner_decisions(
        value,
        sample_rows=rows,
        decision_file_sha256="a" * 64,
        plan_file_sha256="b" * 64,
    )
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
    seal = receipt.pop("receipt_sha256")
    assert seal == sha256_value(receipt)
    assert rows[0]["messages"][0]["content"] not in json.dumps(receipt)


@pytest.mark.parametrize("reviewer", ["ChatGPT", "owner-claude", "codex-reviewer"])
def test_ai_alias_cannot_claim_human_owner_review(reviewer: str) -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="cannot claim OASST1 owner review"):
        validate_owner_decisions(
            _owner_decisions(rows, reviewer=reviewer),
            sample_rows=rows,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_owner_decisions_must_cover_every_row() -> None:
    rows = _rows()
    value = _owner_decisions(rows)
    value["decisions"].pop()
    with pytest.raises(ValueError, match="cover every sampled row"):
        validate_owner_decisions(
            value,
            sample_rows=rows,
            decision_file_sha256="a" * 64,
            plan_file_sha256="b" * 64,
        )


def test_exact_local_plan_and_private_sample_are_bound_when_available() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("exact local OASST1 owner-review plan is not materialized")
    plan = load_owner_review_plan(PLAN_PATH, repository_root=ROOT)
    sample_path = (ROOT / plan["bindings"]["owner_sample"]["path"]).resolve()
    sample = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(
        (ROOT / plan["bindings"]["curation_report"]["path"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    verification = json.loads(
        (ROOT / plan["bindings"]["curation_verification"]["path"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    coverage = validate_owner_sample(
        sample,
        curation_report=report,
        verification=verification,
        plan=plan,
    )
    assert coverage["records"] == 32


def test_actual_owner_receipt_is_complete_sealed_and_non_promoting() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    seal = receipt.pop("receipt_sha256")
    assert seal == sha256_value(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["all_sampled_rows_accepted"] is True
    assert receipt["status"] == (
        "owner_quality_sample_passed_source_promotion_still_separate"
    )
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0
    assert receipt["raw_training_text_retained"] is False
