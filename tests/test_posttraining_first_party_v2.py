from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from lumi.posttraining_ai_assisted import expected_candidate_metadata
from lumi.posttraining_first_party_v2 import (
    audit_first_party_authoring_rows_v2,
    build_blank_authoring_template_v2,
    load_first_party_authoring_v2_plan,
    validate_completed_authoring_row_v2,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-first-party-authoring-pilot-v0.2.json"


def _loaded() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_first_party_authoring_v2_plan(PLAN_PATH, repository_root=ROOT)


def _human_evidence(row: dict[str, Any]) -> None:
    row["author_declaration"] = {
        "author_id": "human-author-01",
        "authored_at": "2026-08-06T00:00:00Z",
        "human_written_from_scratch": True,
        "model_assistance_used": False,
        "rights_confirmed": True,
    }
    row["independent_review"] = {
        "accepted": True,
        "context_grounding_passed": True,
        "language_naturalness_passed": True,
        "quality_score": 0.95,
        "reviewer_id": "human-reviewer-02",
        "safety_review_passed": True,
        "task_correctness_passed": True,
    }
    row["template_family_id"] = f"{row['skill_id']}:{row['split']}:human-test"


def test_message_native_blank_template_is_exact_and_raw_text_free() -> None:
    plan, coverage = _loaded()
    rows = build_blank_authoring_template_v2(plan, coverage)
    assert len(rows) == 320
    assert all(row["messages"] == [] for row in rows)
    audit = audit_first_party_authoring_rows_v2(
        rows,
        plan=plan,
        coverage=coverage,
    )
    assert audit["status_counts"] == {"accepted": 0, "blank": 320, "invalid": 0}
    assert audit["raw_training_text_retained"] is False


def test_human_calculator_row_requires_real_tool_trajectory() -> None:
    plan, coverage = _loaded()
    rows = build_blank_authoring_template_v2(plan, coverage)
    row = copy.deepcopy(
        next(
            item
            for item in rows
            if item["skill_id"] == "calculator_tool_routing"
            and item["language"] == "en"
        )
    )
    _human_evidence(row)
    row["messages"] = [
        {"content": "Use the calculator for (48 × 7) + 3.", "role": "user"},
        {
            "content": (
                '{"arguments":{"expression":"(48 * 7) + 3"},'
                '"name":"calculator"}'
            ),
            "role": "tool",
        },
        {"content": '{"result":"339"}', "role": "tool_result"},
        {"content": "The verified result is 339.", "role": "assistant"},
    ]
    row["task_evidence"] = {
        "expected_result": 339,
        "required_response_terms": ["339"],
        "route_kind": "success",
    }
    expected = expected_candidate_metadata(coverage)[row["record_id"]]
    assert validate_completed_authoring_row_v2(
        row,
        expected=expected,
        plan=plan,
    )["record_id"] == row["record_id"]

    flat = copy.deepcopy(row)
    flat["messages"] = [flat["messages"][0], flat["messages"][-1]]
    with pytest.raises(ValueError, match="skill_role_shape"):
        validate_completed_authoring_row_v2(flat, expected=expected, plan=plan)


def test_human_multi_turn_row_is_scoped_to_current_conversation() -> None:
    plan, coverage = _loaded()
    rows = build_blank_authoring_template_v2(plan, coverage)
    row = copy.deepcopy(
        next(
            item
            for item in rows
            if item["skill_id"] == "multi_turn_context_and_correction"
            and item["language"] == "en"
        )
    )
    _human_evidence(row)
    row["messages"] = [
        {"content": "For this chat's draft, use blue.", "role": "user"},
        {"content": "The draft in this chat is currently blue.", "role": "assistant"},
        {"content": "Correction: use green instead.", "role": "user"},
        {
            "content": "For this conversation's draft, green replaces blue.",
            "role": "assistant",
        },
    ]
    row["task_evidence"] = {
        "context_retained": True,
        "context_scope_explicit": True,
        "required_response_terms": ["green", "draft"],
    }
    expected = expected_candidate_metadata(coverage)[row["record_id"]]
    validate_completed_authoring_row_v2(row, expected=expected, plan=plan)

    false_memory = copy.deepcopy(row)
    false_memory["messages"][-1]["content"] += " I'll remember this."
    with pytest.raises(ValueError, match="persistent_memory_claim"):
        validate_completed_authoring_row_v2(
            false_memory,
            expected=expected,
            plan=plan,
        )


def test_human_track_rejects_model_assistance_and_same_person_review() -> None:
    plan, coverage = _loaded()
    rows = build_blank_authoring_template_v2(plan, coverage)
    row = copy.deepcopy(
        next(
            item
            for item in rows
            if item["skill_id"] == "greeting_and_small_talk"
            and item["language"] == "en"
        )
    )
    _human_evidence(row)
    row["messages"] = [
        {"content": "Hi, can we plan my day?", "role": "user"},
        {"content": "Sure—let's choose the first task for your day.", "role": "assistant"},
    ]
    row["task_evidence"] = {"required_response_terms": ["day"]}
    expected = expected_candidate_metadata(coverage)[row["record_id"]]

    assisted = copy.deepcopy(row)
    assisted["author_declaration"]["model_assistance_used"] = True
    with pytest.raises(ValueError, match="model_assistance_used_in_human_track"):
        validate_completed_authoring_row_v2(assisted, expected=expected, plan=plan)

    same_person = copy.deepcopy(row)
    same_person["independent_review"]["reviewer_id"] = "human-author-01"
    with pytest.raises(ValueError, match="author_reviewer_not_independent"):
        validate_completed_authoring_row_v2(
            same_person,
            expected=expected,
            plan=plan,
        )
