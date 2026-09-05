from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_dmath_curation import (
    apply_prompt_near_dedup,
    assign_validation_split,
    build_candidate_records,
    build_dmath_owner_review_html,
    deterministic_owner_sample,
    load_dmath_curation_plan,
    run_isolated_verifier,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_curation_plan_and_bindings_load() -> None:
    plan = load_dmath_curation_plan(
        ROOT / "configs/lumi-4b-sft-dmath-curation-v0.1.json",
        repository_root=ROOT,
    )
    assert plan["status"] == "authorized_cpu_only_dmath_curation_no_promotion"
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["test_split_access_authorized"] is False


def _source_row(
    *,
    answer: str = "59",
    category: str = "Arithmetic calculation",
    code: str = "a = 9\nb = 6\nprint(a * b + 5)",
    question: str = "What is nine times six plus five?",
) -> dict[str, str]:
    return {
        "answer_en": answer,
        "answer_ko": "답",
        "category": category,
        "question_en": question,
        "question_ko": "질문",
        "solution_abst_en": "9 6 [OP_MUL] 5 [OP_ADD]",
        "solution_abst_ko": "표현",
        "solution_code_en": code,
        "solution_code_ko": code,
    }


def test_isolated_verifier_passes_safe_program_and_fails_closed() -> None:
    rows = {
        "1": _source_row(),
        "2": _source_row(code="import os\nprint(59)"),
    }
    ledger, passed, reasons = run_isolated_verifier(
        rows,
        worker_path=ROOT / "scripts/dmath_verifier_worker.py",
    )
    assert passed == {"1"}
    assert reasons == {"forbidden_ast_node:Import": 1, "verified": 1}
    assert len(ledger) == 2
    assert "solution_code_en" not in ledger[0]


def test_candidate_uses_only_question_and_final_answer() -> None:
    records, attributions = build_candidate_records(
        {"1": _source_row()},
        verified_source_ids={"1"},
    )
    assert records[0]["messages"] == [
        {"content": "What is nine times six plus five?", "role": "user"},
        {"content": "The answer is 59.", "role": "assistant"},
    ]
    assert attributions[0]["excluded_training_target_fields"] == [
        "solution_abst_en",
        "solution_code_en",
    ]
    assert "solution_abst_en" not in records[0]
    assert "solution_code_en" not in records[0]


def test_prompt_near_dedup_is_global_and_deterministic() -> None:
    rows = {
        "1": _source_row(),
        "2": _source_row(answer="59", question="What is nine times six plus five?"),
        "3": _source_row(answer="60", question="What is ten times six?"),
    }
    records, _ = build_candidate_records(rows, verified_source_ids=set(rows))
    accepted, ledger, metrics = apply_prompt_near_dedup(
        records,
        near_duplicate_config={
            "num_perm": 128,
            "package_version": "2.0.0",
            "scheme": "affine32",
            "seed": 1,
            "shingle_tokens": 5,
            "threshold": 0.9,
        },
    )
    assert [row["record_id"] for row in accepted] == [
        "dmath-train-000001",
        "dmath-train-000003",
    ]
    assert [row["accepted"] for row in ledger] == [True, False, True]
    assert metrics["rejected"] == 1


def test_split_and_owner_sample_are_deterministic() -> None:
    source_rows = {
        str(index): _source_row(
            answer=str(index),
            category=("Arithmetic calculation" if index % 2 else "Geometry"),
            question=f"Unique deterministic math question number {index}?",
        )
        for index in range(1, 81)
    }
    records, attributions = build_candidate_records(
        source_rows,
        verified_source_ids=set(source_rows),
    )
    attribution_by_id = {row["record_id"]: row for row in attributions}
    first = assign_validation_split(
        records,
        attribution_by_id=attribution_by_id,
        validation_fraction=0.05,
    )
    second = assign_validation_split(
        records,
        attribution_by_id=attribution_by_id,
        validation_fraction=0.05,
    )
    assert first == second
    assert sum(row["split"] == "validation" for row in first) == 4
    sample = deterministic_owner_sample(
        first,
        attribution_by_id=attribution_by_id,
        maximum_records=32,
    )
    assert len(sample) == 32
    assert {row["source_category"] for row in sample} == {
        "Arithmetic calculation",
        "Geometry",
    }
    html, sample_sha256 = build_dmath_owner_review_html(sample)
    assert sample_sha256
    assert "lumi.dmath-owner-decisions.v0.1" in html
    assert "hidden abstract and Python solutions are verifier evidence only" in html


def test_validation_fraction_fails_closed() -> None:
    with pytest.raises(ValueError, match="validation fraction"):
        assign_validation_split([], attribution_by_id={}, validation_fraction=0.0)
