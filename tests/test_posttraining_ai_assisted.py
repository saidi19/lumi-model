from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lumi.posttraining_ai_assisted import (
    AI_ASSISTED_ROW_SCHEMA_VERSION,
    audit_ai_assisted_candidate_rows,
    build_ai_assisted_manifest,
    candidate_source_revision,
    expected_candidate_metadata,
    load_ai_assisted_candidate_evidence,
    load_ai_assisted_pilot_plan,
    select_owner_review_sample,
    to_sft_candidate_records,
    validate_ai_assisted_candidate_row,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-ai-assisted-sft-pilot-v0.1.json"
EVIDENCE_PATH = ROOT / "evidence" / "lumi-4b-ai-assisted-sft-pilot-candidate-v0.1.json"


def _messages(metadata: dict, index: int) -> tuple[list[dict[str, str]], dict]:
    record_id = metadata["record_id"]
    language = metadata["language"]
    prefix = "পরীক্ষা" if language == "bn" else "test"
    skill = metadata["skill_id"]
    if skill == "simple_arithmetic_model_answer":
        return (
            [
                {"role": "user", "content": f"{prefix} {record_id}: 2 + {index}?"},
                {"role": "assistant", "content": str(2 + index)},
            ],
            {"expected_integer": 2 + index},
        )
    if skill == "calculator_tool_routing":
        result = 1000 + index
        return (
            [
                {"role": "user", "content": f"{prefix} {record_id}: calculate it"},
                {"role": "tool", "content": f'{{"expression":"{result}"}}'},
                {"role": "tool_result", "content": f'{{"result":"{result}"}}'},
                {"role": "assistant", "content": f"Result: {result}"},
            ],
            {"expected_result": result},
        )
    if skill == "multi_turn_context_and_correction":
        return (
            [
                {"role": "user", "content": f"{prefix} {record_id}: remember blue"},
                {"role": "assistant", "content": "Got it: blue."},
                {"role": "user", "content": "Correction: use green."},
                {"role": "assistant", "content": "Understood—I'll use green."},
            ],
            {"context_retained": True},
        )
    if skill == "identity_and_capability_boundaries":
        return (
            [
                {"role": "user", "content": f"{prefix} {record_id}: who are you?"},
                {
                    "role": "assistant",
                    "content": "I'm Lumi, an AI assistant developed by Afnan Technologies Inc.",
                },
            ],
            {"verified_claims_only": True},
        )
    if skill == "greeting_and_small_talk":
        evidence: dict = {}
        response = f"Hi—good to see you. ({record_id})"
    else:
        evidence = {"intent_recovered": True}
        response = f"I understood the intended request and handled it clearly. ({record_id})"
    return (
        [
            {"role": "user", "content": f"{prefix} {record_id}"},
            {"role": "assistant", "content": response},
        ],
        evidence,
    )


def _rows() -> tuple[dict, dict, list[dict]]:
    plan, coverage = load_ai_assisted_pilot_plan(PLAN_PATH, repository_root=ROOT)
    rows = []
    for index, metadata in enumerate(expected_candidate_metadata(coverage).values()):
        messages, task_evidence = _messages(metadata, index)
        rows.append(
            {
                **metadata,
                "automated_review": {
                    "identity_tool_claims_passed": True,
                    "language_mode_passed": True,
                    "message_sequence_passed": True,
                    "pii_secret_scan_passed": True,
                    "quality_score": 0.9,
                    "reviewer_id": plan["automated_review"]["reviewer_id"],
                    "safety_passed": True,
                    "task_correctness_passed": True,
                },
                "generation": {
                    "generated_at": "2026-08-06T00:00:00Z",
                    "generator_id": "openai-codex-current-session",
                    "human_written_from_scratch": False,
                    "method": plan["generation_policy"]["method"],
                    "rights_confirmed": True,
                },
                "messages": messages,
                "owner_review": {
                    "accepted": None,
                    "reviewed_at": "",
                    "reviewer_id": "",
                    "status": "pending",
                },
                "schema_version": AI_ASSISTED_ROW_SCHEMA_VERSION,
                "task_evidence": task_evidence,
            }
        )
    return plan, coverage, rows


def test_ai_assisted_candidate_audit_is_exact_honest_and_raw_text_free() -> None:
    plan, coverage, rows = _rows()
    report = audit_ai_assisted_candidate_rows(rows, coverage_plan=coverage, plan=plan)
    assert report["records"] == 320
    assert report["automated_review_passed_records"] == 320
    assert report["normalized_unique_records"] == 320
    assert report["split_counts"] == {"train": 240, "validation": 80}
    assert report["owner_review"] == {
        "complete": False,
        "sample_records": 32,
        "status": "pending",
    }
    assert report["production_eligible"] is False
    assert report["raw_training_text_retained"] is False
    assert rows[0]["messages"][0]["content"] not in str(report)


def test_ai_assisted_candidate_evidence_is_raw_text_free_and_nonproduction() -> None:
    evidence = load_ai_assisted_candidate_evidence(EVIDENCE_PATH)
    assert evidence["candidate"]["records"] == 320
    assert evidence["owner_gate"]["complete"] is False
    assert evidence["production_eligible"] is False
    assert evidence["private_artifact"]["raw_training_text_in_this_evidence"] is False


def test_ai_assisted_owner_sample_and_sft_conversion_remain_nonproduction() -> None:
    plan, coverage, rows = _rows()
    audit = audit_ai_assisted_candidate_rows(rows, coverage_plan=coverage, plan=plan)
    sample = select_owner_review_sample(rows)
    assert len(sample) == 32
    assert {row["skill_id"] for row in sample} == {
        row["skill_id"] for row in rows
    }
    assert all(any(row["split"] == split for row in sample) for split in ("train", "validation"))
    revision = candidate_source_revision(rows)
    candidates = to_sft_candidate_records(rows, plan=plan, source_revision=revision)
    assert len(candidates) == 320
    assert all(record["source"]["revision"] == revision for record in candidates)
    assert all(
        record["provenance"]["generation_method"] == "ai_generated"
        and record["provenance"]["human_review_status"] == "owner_spot_check_pending"
        for record in candidates
    )
    manifest = build_ai_assisted_manifest(
        audit=audit,
        candidate_records=candidates,
        owner_sample=sample,
    )
    assert manifest["production_eligible"] is False
    assert manifest["owner_review_complete"] is False
    assert manifest["raw_training_text_retained"] is False


def test_ai_assisted_candidate_rejects_human_claim_and_metadata_drift() -> None:
    plan, coverage, rows = _rows()
    expected = expected_candidate_metadata(coverage)[rows[0]["record_id"]]
    changed = copy.deepcopy(rows[0])
    changed["generation"]["human_written_from_scratch"] = True
    with pytest.raises(ValueError, match="must not claim human authorship"):
        validate_ai_assisted_candidate_row(changed, expected=expected, plan=plan)

    changed = copy.deepcopy(rows)
    changed[0]["split"] = "validation"
    with pytest.raises(ValueError, match="immutable metadata changed"):
        audit_ai_assisted_candidate_rows(changed, coverage_plan=coverage, plan=plan)


def test_ai_assisted_candidate_rejects_duplicate_conversation() -> None:
    plan, coverage, rows = _rows()
    rows[1]["messages"] = copy.deepcopy(rows[0]["messages"])
    rows[1]["task_evidence"] = copy.deepcopy(rows[0]["task_evidence"])
    with pytest.raises(ValueError):
        audit_ai_assisted_candidate_rows(rows, coverage_plan=coverage, plan=plan)
