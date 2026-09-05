from __future__ import annotations

import json

from lumi.posttraining_aya_reconciliation import (
    build_aya_audit_reconciliation_report,
)
from lumi.posttraining_source_quality import sha256_value


def _wrapped(index: int, prompt: str, target: str, user: str) -> dict:
    return {
        "row_idx": index,
        "row": {
            "annotation_type": "original-annotations",
            "inputs": prompt,
            "language": "Bengali",
            "language_code": "ben",
            "targets": target,
            "user_id": user,
        },
    }


def test_aya_reconciliation_separates_index_collisions_from_source_duplicates() -> None:
    rows = [
        _wrapped(1, "প্রশ্ন এক", "উত্তর এক", "one"),
        _wrapped(1, "প্রশ্ন দুই", "উত্তর দুই", "one"),
        _wrapped(2, "প্রশ্ন তিন", "উত্তর তিন", "two"),
        _wrapped(3, "প্রশ্ন তিন", "উত্তর তিন", "two"),
    ]
    original = {
        "duplicate_metrics": {
            "duplicate_pair_records": 1,
            "duplicate_row_indices": 1,
        },
        "format_and_content_metrics": {"trailing_pipe_records": 2},
        "language_metrics": {"language_code_mismatches": 0},
        "production_promotions": 0,
        "provenance_metrics": {"top_annotator_records": 3},
        "raw_text_retained": False,
        "remaining_production_blockers": [
            "full_bengali_subset_top_annotator_exceeds_half",
            "full_bengali_subset_contains_trailing_pipe_artifacts",
            "full_bengali_subset_contains_duplicate_row_indices",
            "full_bengali_subset_exact_pairs_require_deduplication",
        ],
        "schema_version": "lumi.sft-aya-bengali-source-audit-report.v0.1",
    }
    original["report_sha256"] = sha256_value(original)
    statistics = {
        "partial": False,
        "statistics": [
            {
                "column_name": "language_code",
                "column_statistics": {"frequencies": {"ben": 4}},
            }
        ],
    }
    report = build_aya_audit_reconciliation_report(
        bindings={"input_sha256": "a" * 64},
        observed_at="2026-08-06",
        original_report=original,
        statistics=statistics,
        wrapped_rows=rows,
    )
    integrity = report["retrieval_integrity"]
    duplicates = report["duplicate_interpretation"]
    assert integrity["row_index_collision_excess"] == 1
    assert integrity["distinct_content_index_collision_groups"] == 1
    assert duplicates["exact_full_payload_duplicate_excess"] == 1
    assert duplicates["full_duplicate_groups_with_distinct_wrapper_indices"] == 1
    assert "full_bengali_subset_contains_duplicate_row_indices" not in report[
        "remaining_production_blockers"
    ]
    assert any(
        "pagination_index_instability" in blocker
        for blocker in report["remaining_production_blockers"]
    )
    assert report["production_promotions"] == 0
    assert "প্রশ্ন" not in json.dumps(report, ensure_ascii=False)
