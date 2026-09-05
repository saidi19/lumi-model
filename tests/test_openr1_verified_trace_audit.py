from __future__ import annotations

import json

from lumi.posttraining_openr1 import (
    build_openr1_audit_report,
    select_verified_openr1_records,
    sha256_value,
)


def _wrapped_row(
    *,
    problem: str = "Solve the bounded test problem.",
    source: str = "test-source",
    solution: str = "Reference solution.",
    traces: list[str] | None = None,
) -> dict:
    selected_traces = traces or ["Incomplete trace.", "Verified generated trace."]
    return {
        "row_idx": 7,
        "row": {
            "correctness_math_verify": [False, True],
            "generations": selected_traces,
            "is_reasoning_complete": [False, True],
            "problem": problem,
            "solution": solution,
            "source": source,
            "uuid": "row-uuid",
        },
    }


def test_openr1_selects_verified_generation_not_reference_solution() -> None:
    records, metrics = select_verified_openr1_records(
        [_wrapped_row()],
        revision="a" * 40,
        maximum_selected_records=24,
    )
    assert len(records) == 1
    assert records[0]["messages"][-1]["content"] == "Verified generated trace."
    assert records[0]["messages"][-1]["content"] != "Reference solution."
    assert records[0]["generation_provenance"]["generation_index"] == 1
    assert records[0]["generation_provenance"]["verifier_passed"] is True
    assert metrics["selected_records"] == 1
    assert metrics["selected_generation_index_counts"] == {"1": 1}


def test_openr1_rejects_external_dependency_and_ambiguous_reference_copy() -> None:
    dependency = _wrapped_row(
        problem="See ![](https://example.invalid/problem.png)",
    )
    records, metrics = select_verified_openr1_records(
        [dependency],
        revision="b" * 40,
        maximum_selected_records=24,
    )
    assert records == []
    assert metrics["problem_external_image_records"] == 1
    assert metrics["problem_external_url_records"] == 1
    assert metrics["rejection_counts"] == {"problem_external_dependency": 1}

    reference_copy = _wrapped_row(
        solution="Verified generated trace.",
    )
    records, metrics = select_verified_openr1_records(
        [reference_copy],
        revision="b" * 40,
        maximum_selected_records=24,
    )
    assert records == []
    assert metrics["exact_reference_match_candidates_rejected"] == 1


def test_openr1_report_is_raw_text_free_sealed_and_non_promoting() -> None:
    _, metrics = select_verified_openr1_records(
        [_wrapped_row()],
        revision="c" * 40,
        maximum_selected_records=24,
    )
    report = build_openr1_audit_report(
        encoding_report={
            "accepted": 1,
            "rejected": 0,
            "total": 1,
            "total_tokens": 64,
        },
        filter_report={
            "accepted": 1,
            "rejected": 0,
            "rejection_counts": {},
            "report_sha256": "d" * 64,
        },
        metadata_sha256="e" * 64,
        observed_at="2026-08-06",
        plan_sha256="f" * 64,
        raw_rows_sha256="0" * 64,
        selection_metrics=metrics,
        source_revision="c" * 40,
    )
    assert report["adapter_repair"]["reference_solution_field_used"] is False
    assert report["production_eligible"] is False
    assert report["production_promotions"] == 0
    assert report["raw_text_retained"] is False
    seal = report.pop("report_sha256")
    assert seal == sha256_value(report)
    serialized = json.dumps(report, ensure_ascii=False)
    assert "Verified generated trace." not in serialized
    assert "Solve the bounded test problem." not in serialized


def test_openr1_report_reconciles_normal_filter_rejections() -> None:
    _, metrics = select_verified_openr1_records(
        [_wrapped_row()],
        revision="1" * 40,
        maximum_selected_records=24,
    )
    report = build_openr1_audit_report(
        encoding_report={"accepted": 0, "rejected": 0, "total": 0},
        filter_report={
            "accepted": 0,
            "rejected": 1,
            "rejection_counts": {"benchmark_contamination": 1},
            "report_sha256": "2" * 64,
        },
        metadata_sha256="3" * 64,
        observed_at="2026-08-06",
        plan_sha256="4" * 64,
        raw_rows_sha256="5" * 64,
        selection_metrics=metrics,
        source_revision="1" * 40,
    )
    assert "bounded_slice_contains_normal_filter_rejections" in report[
        "remaining_production_blockers"
    ]
