from __future__ import annotations

import json

from lumi.posttraining_openr1_pii import (
    build_openr1_pii_report,
    diagnose_openr1_pii_patterns,
    sha256_value,
)


def _record(answer: str) -> dict:
    return {
        "messages": [
            {"content": "Solve this problem.", "role": "user"},
            {"content": answer, "role": "assistant"},
        ]
    }


def test_math_decimal_collision_is_diagnostic_not_an_override() -> None:
    metrics = diagnose_openr1_pii_patterns([_record("x≈1.2345678901")])
    assert metrics["detector_record_counts"] == {"phone": 1}
    assert metrics["phone_matches_by_role"] == {"assistant": 1}
    assert metrics["phone_matches_in_math_context"] == 1
    assert metrics["contact_keyword_records"] == 0
    assert metrics["likely_math_numeric_false_positive_records"] == 1
    assert metrics["unresolved_phone_like_records"] == 0
    assert metrics["review_outcome"].endswith("not_automatically_overridden")


def test_contact_number_remains_unresolved() -> None:
    metrics = diagnose_openr1_pii_patterns([_record("Call 2025550123")])
    assert metrics["detector_record_counts"] == {"phone": 1}
    assert metrics["contact_keyword_records"] == 1
    assert metrics["likely_math_numeric_false_positive_records"] == 0
    assert metrics["unresolved_phone_like_records"] == 1


def test_pii_report_is_raw_text_free_sealed_and_non_promoting() -> None:
    metrics = diagnose_openr1_pii_patterns([_record("x≈1.2345678901")])
    report = build_openr1_pii_report(
        diagnostic_metrics=metrics,
        observed_at="2026-08-06",
        plan_sha256="a" * 64,
        raw_rows_sha256="b" * 64,
        source_filter_sha256="c" * 64,
        verified_trace_evidence_file_sha256="d" * 64,
        verified_trace_evidence_internal_sha256="e" * 64,
    )
    assert report["global_pii_filter_modified"] is False
    assert report["production_eligible"] is False
    assert report["production_promotions"] == 0
    seal = report.pop("report_sha256")
    assert seal == sha256_value(report)
    serialized = json.dumps(report, ensure_ascii=False)
    assert "1.2345678901" not in serialized
