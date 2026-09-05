from __future__ import annotations

import pytest

from lumi.posttraining_openr1_style_repair import (
    VisibleFinalRepairRejection,
    build_visible_final_review_html,
    normalize_markdown_math,
    repair_record,
    select_visible_final_review_sample,
    split_think_response,
    style_metrics,
)


def _record() -> dict:
    return {
        "generation_provenance": {
            "generation_index": 0,
            "reasoning_complete": True,
            "trace_field": "generations",
            "verifier_field": "correctness_math_verify",
            "verifier_passed": True,
        },
        "language": "en",
        "messages": [
            {"content": "What is one half?", "role": "user"},
            {
                "content": (
                    "<think>We divide one by two.</think>"
                    "The result is \\(\\boxed{\\frac{1}{2}}\\)."
                ),
                "role": "assistant",
            },
        ],
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": "a" * 64,
            "transform_sha256": "b" * 64,
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": "source",
            "review_status": "accepted",
            "score": 0.9,
            "secret_scan": "passed",
        },
        "record_id": "openr1-parent",
        "safety": {"policy_version": "v1", "risk_labels": []},
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": "open-r1/OpenR1-Math-220k",
            "document_id": "row-1",
            "license_id": "apache-2.0",
            "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
            "source_id": "openr1_math_220k",
            "uri": "https://example.invalid/row-1",
        },
        "split": "train",
        "task_category": "reasoning_math_science",
    }


def _attribution() -> dict:
    return {
        "record_id": "openr1-parent",
        "rights_status": "unresolved_quarantined",
        "source_object_path": "data/train-00000-of-00010.parquet",
        "source_object_sha256": "c" * 64,
    }


def test_split_think_response_requires_exact_pair() -> None:
    reasoning, final = split_think_response("<think>reason</think>answer")
    assert reasoning == "reason"
    assert final == "answer"
    with pytest.raises(VisibleFinalRepairRejection, match="exactly one"):
        split_think_response("answer only")
    with pytest.raises(VisibleFinalRepairRejection, match="before think"):
        split_think_response("prefix<think>reason</think>answer")


def test_math_delimiters_normalize_and_braces_fail_closed() -> None:
    assert normalize_markdown_math(r"Answer: \(\frac{1}{2}\).") == (
        r"Answer: $\frac{1}{2}$."
    )
    assert "$$" in normalize_markdown_math(r"\[x = 2\]")
    with pytest.raises(ValueError, match="braces"):
        normalize_markdown_math(r"\(\frac{1}{2\)")


def test_repair_separates_hidden_reasoning_from_sft_target() -> None:
    repaired, hidden, attribution = repair_record(_record(), _attribution())

    visible = repaired["messages"][-1]["content"]
    assert "<think>" not in visible and "</think>" not in visible
    assert visible == r"The result is $\boxed{\frac{1}{2}}$."
    assert hidden["content"] == "We divide one by two."
    assert hidden["sft_target"] is False
    assert attribution["parent_record_id"] == "openr1-parent"
    assert repaired["record_id"] != "openr1-parent"


def test_style_metrics_close_visible_gates() -> None:
    repaired, _, _ = repair_record(_record(), _attribution())
    metrics = style_metrics([repaired])

    assert metrics["records"] == 1
    assert metrics["visible_think_tag_records"] == 0
    assert metrics["legacy_tex_delimiter_records"] == 0
    assert metrics["unbalanced_latex_brace_records"] == 0
    assert metrics["unbalanced_display_math_records"] == 0


def test_review_page_is_final_answer_only() -> None:
    repaired, _, _ = repair_record(_record(), _attribution())
    html, sample_sha256 = build_visible_final_review_html([repaired])

    assert "OpenR1 final-answer review" in html
    assert "Hidden reasoning has been removed" in html
    assert "We divide one by two" not in html
    assert len(sample_sha256) == 64


def test_review_sample_replaces_a_rejected_parent() -> None:
    repaired = []
    repaired_by_parent = {}
    for index in range(32):
        row = _record()
        row["record_id"] = f"visible-{index:02d}"
        repaired.append(row)
        repaired_by_parent[f"parent-{index:02d}"] = row
    source_sample = [{"record_id": f"parent-{index:02d}"} for index in range(31)]
    source_sample.append({"record_id": "rejected-parent"})

    selected, retained = select_visible_final_review_sample(
        source_sample,
        repaired_by_parent,
        repaired,
    )

    assert retained == 31
    assert len(selected) == 32
    assert len({row["record_id"] for row in selected}) == 32


def test_review_sample_keeps_exact_clean_source_sample() -> None:
    repaired = []
    repaired_by_parent = {}
    for index in range(32):
        row = _record()
        row["record_id"] = f"visible-{index:02d}"
        repaired.append(row)
        repaired_by_parent[f"parent-{index:02d}"] = row
    source_sample = [{"record_id": f"parent-{index:02d}"} for index in range(32)]

    selected, retained = select_visible_final_review_sample(
        source_sample,
        repaired_by_parent,
        repaired,
    )

    assert retained == 32
    assert [row["record_id"] for row in selected] == [
        f"visible-{index:02d}" for index in range(32)
    ]
