from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lumi.posttraining_capabilities import (
    generate_arithmetic_cases,
    load_basic_capability_plan,
    parse_single_integer_answer,
    score_arithmetic_responses,
)
from lumi.posttraining_first_party import (
    audit_first_party_authoring_rows,
    build_blank_authoring_template,
    load_first_party_authoring_plan,
    validate_completed_authoring_row,
)

ROOT = Path(__file__).parents[1]
AUTHORING_PLAN = ROOT / "configs" / "lumi-4b-first-party-authoring-pilot-v0.1.json"
CAPABILITY_PLAN = ROOT / "configs" / "lumi-4b-basic-capability-gate-v0.1.json"


def test_first_party_plan_builds_only_blank_bounded_rows() -> None:
    plan = load_first_party_authoring_plan(AUTHORING_PLAN)
    rows = build_blank_authoring_template(plan)
    assert len(rows) == 320
    assert sum(row["split"] == "train" for row in rows) == 240
    assert sum(row["split"] == "validation" for row in rows) == 80
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert all(row["user_message"] == "" for row in rows)
    assert all(row["assistant_response"] == "" for row in rows)
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["model_generated_training_text_authorized"] is False


def test_completed_first_party_row_requires_human_rights_and_independent_review() -> None:
    plan = load_first_party_authoring_plan(AUTHORING_PLAN)
    row = build_blank_authoring_template(plan)[0]
    with pytest.raises(ValueError, match="empty user_message"):
        validate_completed_authoring_row(row)

    completed = copy.deepcopy(row)
    completed["user_message"] = "Hello"
    completed["assistant_response"] = "Hi! What are we working on?"
    completed["author_declaration"] = {
        "author_id": "writer-001",
        "authored_at": "2026-08-05T00:00:00Z",
        "human_written_from_scratch": True,
        "rights_confirmed": True,
    }
    completed["independent_review"] = {
        "accepted": True,
        "quality_score": 0.95,
        "reviewer_id": "reviewer-001",
        "safety_review_passed": True,
    }
    assert validate_completed_authoring_row(completed)["record_id"] == row["record_id"]

    completed["independent_review"]["reviewer_id"] = "writer-001"
    with pytest.raises(ValueError, match="must differ"):
        validate_completed_authoring_row(completed)


def test_first_party_authoring_audit_is_exact_and_raw_text_free() -> None:
    plan = load_first_party_authoring_plan(AUTHORING_PLAN)
    rows = build_blank_authoring_template(plan)

    accepted = rows[0]
    accepted["user_message"] = "Private authored prompt sentinel"
    accepted["assistant_response"] = "Private authored response sentinel"
    accepted["author_declaration"] = {
        "author_id": "writer-001",
        "authored_at": "2026-08-05T00:00:00Z",
        "human_written_from_scratch": True,
        "rights_confirmed": True,
    }
    accepted["independent_review"] = {
        "accepted": True,
        "quality_score": 0.95,
        "reviewer_id": "reviewer-001",
        "safety_review_passed": True,
    }

    rows[1]["user_message"] = "Work in progress sentinel"
    rows[2]["author_declaration"]["rights_confirmed"] = False
    report = audit_first_party_authoring_rows(rows, plan=plan)

    assert report["status_counts"] == {
        "accepted": 1,
        "blank": 317,
        "in_progress": 1,
        "invalid": 1,
    }
    assert report["authoring_complete"] is False
    assert report["raw_training_text_retained"] is False
    assert "Private authored prompt sentinel" not in str(report)
    assert "Private authored response sentinel" not in str(report)
    assert "Work in progress sentinel" not in str(report)
    assert report["reason_counts"]["rights_rejected"] == 1


def test_first_party_authoring_audit_rejects_missing_or_changed_rows() -> None:
    plan = load_first_party_authoring_plan(AUTHORING_PLAN)
    rows = build_blank_authoring_template(plan)
    with pytest.raises(ValueError, match="exact planned IDs"):
        audit_first_party_authoring_rows(rows[:-1], plan=plan)

    changed = build_blank_authoring_template(plan)
    changed[0]["language"] = "en" if changed[0]["language"] != "en" else "bn"
    with pytest.raises(ValueError, match="immutable metadata changed"):
        audit_first_party_authoring_rows(changed, plan=plan)


def test_arithmetic_gate_is_deterministic_exact_and_balanced() -> None:
    plan = load_basic_capability_plan(CAPABILITY_PLAN)
    first = generate_arithmetic_cases(plan)
    second = generate_arithmetic_cases(plan)
    assert first == second
    assert len(first) == 256
    assert len({case.case_id for case in first}) == 256
    assert {case.operation for case in first} == {
        "addition",
        "exact_division",
        "multiplication",
        "subtraction",
    }
    division_cases = [case for case in first if case.operation == "exact_division"]
    assert all(case.left // case.right == case.expected for case in division_cases)
    assert all(case.left % case.right == 0 for case in division_cases)


def test_arithmetic_scoring_accepts_one_integer_and_retains_no_response_text() -> None:
    plan = load_basic_capability_plan(CAPABILITY_PLAN)
    cases = generate_arithmetic_cases(plan)[:4]
    responses = {
        case.case_id: f"The answer is {case.expected}."
        for case in cases
    }
    report = score_arithmetic_responses(cases, responses)
    assert report["pass_rate"] == 1.0
    assert report["raw_responses_retained"] is False
    assert all("response_sha256" in item for item in report["case_results"])
    assert all(response not in str(report) for response in responses.values())
    assert parse_single_integer_answer("উত্তর 1,024।") == 1024
    assert parse_single_integer_answer("It could be 10 or 11") is None
