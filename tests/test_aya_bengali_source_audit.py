from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_source_quality import (
    build_aya_bengali_source_audit_report,
    load_aya_bengali_audit_plan,
    sha256_value,
)

ROOT = Path(__file__).parents[1]


def _row(index: int, *, annotator: str, target: str) -> dict:
    return {
        "row": {
            "annotation_type": "original-annotations",
            "inputs": f"বাংলা প্রশ্ন {index}",
            "language": "Bengali",
            "language_code": "ben",
            "targets": target,
            "user_id": annotator,
        },
        "row_idx": index,
    }


def test_aya_bengali_audit_plan_is_exact_and_non_promoting() -> None:
    plan = load_aya_bengali_audit_plan(
        ROOT / "configs" / "lumi-4b-sft-aya-bengali-source-audit-v0.1.json",
        repository_root=ROOT,
    )
    assert plan["limits"]["maximum_fetched_rows"] == 1534
    assert plan["limits"]["maximum_total_response_bytes"] == 64 * 1024 * 1024
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["external_api_spend_authorized"] is False
    assert plan["output"]["production_promotion_permitted"] is False


def test_aya_bengali_full_audit_is_raw_text_free_and_fail_closed() -> None:
    rows = [
        _row(1, annotator="one", target="প্রথম উত্তর"),
        _row(2, annotator="one", target="দ্বিতীয় উত্তর |"),
        _row(3, annotator="two", target="তৃতীয় উত্তর"),
        _row(4, annotator="two", target="চতুর্থ উত্তর"),
    ]
    report = build_aya_bengali_source_audit_report(
        expected_rows=4,
        maximum_top_annotator_fraction=0.5,
        metadata_post_sha256="a" * 64,
        metadata_pre_sha256="b" * 64,
        observed_at="2026-08-06",
        page_inventory=[
            {
                "filename": "aya-bengali-page-000.json",
                "offset": 0,
                "rows": 4,
                "sha256": "c" * 64,
                "size_bytes": 1024,
            }
        ],
        plan_sha256="d" * 64,
        response_bytes=2048,
        response_count=3,
        source_revision="e" * 40,
        wrapped_rows=rows,
    )
    assert report["provenance_metrics"]["top_annotator_fraction"] == 0.5
    assert report["format_and_content_metrics"]["trailing_pipe_records"] == 1
    assert "full_bengali_subset_contains_trailing_pipe_artifacts" in report[
        "remaining_production_blockers"
    ]
    assert report["production_eligible"] is False
    assert report["production_promotions"] == 0
    assert report["raw_text_retained"] is False

    seal = report.pop("report_sha256")
    assert seal == sha256_value(report)
    serialized = json.dumps(report, ensure_ascii=False)
    assert "প্রথম উত্তর" not in serialized
    assert "বাংলা প্রশ্ন" not in serialized
