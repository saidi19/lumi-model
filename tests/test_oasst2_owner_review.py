from __future__ import annotations

import pytest

from lumi.posttraining_oasst2_curation import sha256_value
from lumi.posttraining_oasst2_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    EXPECTED_CANDIDATE,
    validate_owner_decisions,
    validate_owner_review_receipt,
)


def _sample() -> list[dict[str, object]]:
    return [
        {
            "language_mode": "english",
            "messages": [
                {"content": f"Question {index}", "role": "user"},
                {"content": f"Answer {index}", "role": "assistant"},
            ],
            "record_id": f"oasst2-row-{index:02d}",
            "skill_id": "conversation",
            "split": "validation" if index % 2 else "train",
        }
        for index in range(32)
    ]


def _plan(sample_sha256: str) -> dict[str, object]:
    return {
        "bindings": {
            "accepted_records": {"sha256": "1" * 64},
            "curation_report": {
                "internal_sha256": "2" * 64,
                "sha256": "3" * 64,
            },
            "curation_verification": {
                "internal_sha256": "4" * 64,
                "sha256": "5" * 64,
            },
            "owner_sample": {
                "internal_sha256": sample_sha256,
                "sha256": "6" * 64,
            },
        },
        "candidate": dict(EXPECTED_CANDIDATE),
    }


def _decisions(sample: list[dict[str, object]]) -> dict[str, object]:
    return {
        "decisions": [
            {"accepted": True, "note": "", "record_id": row["record_id"]} for row in sample
        ],
        "reviewed_at": "2026-08-08T08:26:52Z",
        "reviewer_id": "owner-afnan",
        "sample_sha256": sha256_value(sample),
        "schema_version": DECISIONS_SCHEMA_VERSION,
    }


def test_all_accepted_owner_decisions_seal_without_promotion() -> None:
    sample = _sample()
    receipt = validate_owner_decisions(
        _decisions(sample),
        sample_rows=sample,
        plan=_plan(sha256_value(sample)),
        decision_file_sha256="7" * 64,
        plan_file_sha256="8" * 64,
    )
    validate_owner_review_receipt(receipt)
    assert receipt["accepted_records"] == 32
    assert receipt["rejected_records"] == 0
    assert receipt["production_eligible"] is False
    assert receipt["production_promotions"] == 0


def test_ai_alias_cannot_claim_owner_review() -> None:
    sample = _sample()
    value = _decisions(sample)
    value["reviewer_id"] = "codex-reviewer"
    with pytest.raises(ValueError, match="AI alias"):
        validate_owner_decisions(
            value,
            sample_rows=sample,
            plan=_plan(sha256_value(sample)),
            decision_file_sha256="7" * 64,
            plan_file_sha256="8" * 64,
        )


def test_decisions_must_cover_exact_sample() -> None:
    sample = _sample()
    value = _decisions(sample)
    value["decisions"][0]["record_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown record ID"):
        validate_owner_decisions(
            value,
            sample_rows=sample,
            plan=_plan(sha256_value(sample)),
            decision_file_sha256="7" * 64,
            plan_file_sha256="8" * 64,
        )
