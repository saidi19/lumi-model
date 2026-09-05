from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_openr1_pii_secondary import (
    diagnose_line_scoped_phone_patterns,
    load_secondary_pii_plan,
    sha256_value,
    validate_secondary_pii_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-openr1-pii-secondary-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-openr1-pii-secondary-v0.1.json"


def _record(record_id: str, prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {"content": prompt, "role": "user"},
            {"content": answer, "role": "assistant"},
        ],
        "record_id": record_id,
    }


def test_line_scoped_math_number_is_not_treated_as_contact_data() -> None:
    records = [_record("row-1", "Call this value x.", "x≈1.2345678901")]
    metrics = diagnose_line_scoped_phone_patterns(records)
    assert metrics["line_scoped_math_numeric_false_positive_records"] == 1
    assert metrics["same_line_contact_keyword_records"] == 0
    assert metrics["unresolved_records"] == 0


def test_same_line_contact_number_remains_unresolved() -> None:
    records = [_record("row-2", "Solve.", "Call 2025550123 = x")]
    metrics = diagnose_line_scoped_phone_patterns(records)
    assert metrics["same_line_contact_keyword_records"] == 1
    assert metrics["unresolved_records"] == 1


def test_cross_message_regex_join_is_classified_without_global_override() -> None:
    records = [_record("row-3", "Value 12345", "67890 is the continuation.")]
    metrics = diagnose_line_scoped_phone_patterns(records)
    assert metrics["global_phone_detector_records"] == 1
    assert metrics["line_phone_match_count"] == 0
    assert metrics["cross_message_join_artifact_records"] == 1
    assert metrics["unresolved_records"] == 0


def test_exact_secondary_report_is_reproducible_and_nonproduction() -> None:
    plan = load_secondary_pii_plan(PLAN, repository_root=ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_secondary_pii_report(report)
    assert report["diagnostic_metrics"]["global_phone_detector_records"] == 6
    assert report["diagnostic_metrics"]["unresolved_records"] == 0
    assert report["global_pii_filter_modified"] is False
    assert report["production_promotions"] == 0
    seal = report["report_sha256"]
    assert seal == sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert plan["output"]["contains_raw_training_text"] is False


def test_secondary_report_contains_no_prompt_or_response_payloads() -> None:
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in ("messages", "assistant_response", "user_prompt", "input_ids", "labels"):
        assert f'"{forbidden}"' not in serialized
