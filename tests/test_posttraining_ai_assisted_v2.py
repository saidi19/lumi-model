from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lumi.posttraining_ai_assisted import expected_candidate_metadata
from lumi.posttraining_ai_assisted_v2 import (
    audit_ai_assisted_v2_rows,
    build_ai_assisted_v2_manifest,
    load_ai_assisted_v2_evidence,
    load_ai_assisted_v2_plan,
    select_owner_review_sample_v2,
    to_sft_candidate_records_v2,
    validate_ai_assisted_v2_row,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-ai-assisted-sft-pilot-v0.2.json"


def _language_text(language: str, *, train: bool, record_id: str) -> tuple[str, str]:
    if language == "bn":
        if train:
            return (
                f"পরীক্ষার আগে পরিকল্পনা চাই {record_id}",
                f"পরীক্ষার প্রস্তুতির জন্য ধারণা ও অনুশীলন সাজাও—নমুনা {record_id}",
            )
        return (
            f"সাপ্তাহিক অনুশীলন কীভাবে করব {record_id}",
            f"সাপ্তাহিক লক্ষ্য ধরে ভুল বিশ্লেষণ ও পুনরাবৃত্তি করো—যাচাই {record_id}",
        )
    if train:
        return (
            f"Plan before the quiz {record_id}",
            f"Use concepts and practice before the quiz; training example {record_id}",
        )
    return (
        f"How should weekend practice work {record_id}",
        f"For weekend practice, review mistakes and set a fresh goal; heldout {record_id}",
    )


def _base_row(metadata: Mapping[str, str]) -> dict[str, Any]:
    record_id = metadata["record_id"]
    skill = metadata["skill_id"]
    split = metadata["split"]
    language = metadata["language"]
    is_train = split == "train"
    user_text, response_text = _language_text(
        language,
        train=is_train,
        record_id=record_id,
    )
    messages: list[dict[str, str]]
    evidence: dict[str, Any]

    if skill == "greeting_and_small_talk":
        required = (
            ("প্রস্তুতি" if is_train else "লক্ষ্য")
            if language == "bn"
            else ("concepts" if is_train else "goal")
        )
        messages = [
            {"content": user_text, "role": "user"},
            {"content": response_text, "role": "assistant"},
        ]
        evidence = {"required_response_terms": [required]}
    elif skill.startswith("typo_recovery_"):
        terms = (
            (["প্রস্তুতি", "অনুশীলন"] if is_train else ["লক্ষ্য", "ভুল"])
            if language == "bn"
            else (["concepts", "quiz"] if is_train else ["weekend", "goal"])
        )
        messages = [
            {"content": user_text, "role": "user"},
            {"content": response_text, "role": "assistant"},
        ]
        evidence = {
            "intent_recovered": True,
            "modifier_grounded": True,
            "required_response_terms": terms,
        }
    elif skill == "simple_arithmetic_model_answer":
        index = int(record_id.rsplit("-", 1)[1])
        expected = index + (1000 if is_train else 9000)
        arithmetic_prompt = (
            f"হিসাব করো: {expected - 1} + 1"
            if language == "bn"
            else f"Calculate {expected - 1} + 1"
        )
        messages = [
            {"content": arithmetic_prompt, "role": "user"},
            {"content": str(expected), "role": "assistant"},
        ]
        evidence = {
            "expected_integer": expected,
            "required_response_terms": [str(expected)],
        }
    elif skill == "calculator_tool_routing":
        index = int(record_id.rsplit("-", 1)[1])
        expected = index + (2000 if is_train else 8000)
        calculator_prompt = (
            f"ক্যালকুলেটর দিয়ে {expected - 2} + 2 হিসাব করো"
            if language == "bn"
            else f"Use a calculator for {expected - 2} + 2"
        )
        calculator_response = (
            f"যাচাই করা ফল: {expected}"
            if language == "bn"
            else (
                f"Verified calculation: {expected}"
                if is_train
                else f"The calculator returned {expected}; that is the exact result."
            )
        )
        messages = [
            {"content": calculator_prompt, "role": "user"},
            {
                "content": (
                    f'{{"arguments":{{"expression":"{expected - 2} + 2"}},'
                    '"name":"calculator"}'
                ),
                "role": "tool",
            },
            {"content": f'{{"result":"{expected}"}}', "role": "tool_result"},
            {"content": calculator_response, "role": "assistant"},
        ]
        evidence = {
            "expected_result": expected,
            "required_response_terms": [str(expected)],
            "route_kind": "success",
        }
    elif skill == "multi_turn_context_and_correction":
        if language == "bn":
            first = f"এই কথোপকথনে খসড়ার রং নীল ধরো—{record_id}"
            first_response = (
                "এই খসড়ার বর্তমান রং নীল ধরছি।"
                if is_train
                else "শুধু চলতি আলাপের খসড়াটিতে নীল রং প্রযোজ্য।"
            )
            correction = "সংশোধন: নীল নয়, সবুজ হবে।"
            final = f"এই কথোপকথনের খসড়ায় নীল বাদ দিয়ে সবুজ ব্যবহার করব—{record_id}"
            terms = ["সবুজ", "খসড়া"]
        else:
            first = f"For this chat, make draft {record_id} blue."
            first_response = (
                "For this draft, the current color is blue."
                if is_train
                else "Blue applies only to the draft discussed in this chat."
            )
            correction = "Correction: use green instead of blue."
            final = (
                f"For this conversation's draft, I will replace blue with green; {record_id}."
                if is_train
                else (
                    "Green now governs this draft in the current chat, replacing blue; "
                    f"{record_id}."
                )
            )
            terms = ["green", "draft"]
        messages = [
            {"content": first, "role": "user"},
            {"content": first_response, "role": "assistant"},
            {"content": correction, "role": "user"},
            {"content": final, "role": "assistant"},
        ]
        evidence = {
            "context_retained": True,
            "context_scope_explicit": True,
            "required_response_terms": terms,
        }
    elif skill == "identity_and_capability_boundaries":
        if is_train:
            response = (
                f"আমি Lumi, Afnan Technologies Inc.-এর তৈরি AI assistant; নমুনা {record_id}."
                if language == "bn"
                else (
                    "I am Lumi, an AI assistant developed by Afnan Technologies Inc.; "
                    f"{record_id}."
                )
            )
            terms = ["Lumi", "Afnan Technologies Inc."]
            response_kind = "identity"
        else:
            response = (
                f"আমি লেখা ও সাধারণ কাজ ব্যাখ্যা করতে পারি; নমুনা {record_id}."
                if language == "bn"
                else f"I can help explain writing and general tasks; heldout {record_id}."
            )
            terms = ["লেখা"] if language == "bn" else ["tasks"]
            response_kind = "capability"
        messages = [
            {"content": user_text, "role": "user"},
            {"content": response, "role": "assistant"},
        ]
        evidence = {
            "required_response_terms": terms,
            "response_kind": response_kind,
            "verified_claims_only": True,
        }
    else:  # pragma: no cover - the coverage plan is closed
        raise AssertionError(skill)

    return {
        "automated_review": {
            "context_grounding_passed": True,
            "identity_tool_claims_passed": True,
            "language_mode_passed": True,
            "message_sequence_passed": True,
            "pii_secret_scan_passed": True,
            "quality_score": 0.95,
            "reviewer_id": "lumi-deterministic-ai-assisted-guard-v0.2",
            "safety_passed": True,
            "task_correctness_passed": True,
        },
        "generation": {
            "generated_at": "2026-08-06T00:00:00Z",
            "generator_id": "openai-codex-current-session",
            "human_written_from_scratch": False,
            "method": "ai_generated_context_grounded_split_family_isolated_v0.2",
            "rights_confirmed": True,
            "template_family_id": f"{skill}:{split}:synthetic-{record_id}",
        },
        "language": language,
        "messages": messages,
        "owner_review": {
            "accepted": None,
            "reviewed_at": "",
            "reviewer_id": "",
            "status": "pending",
        },
        "record_id": record_id,
        "schema_version": "lumi.ai-assisted-sft-row.v0.2",
        "skill_id": skill,
        "split": split,
        "task_category": metadata["task_category"],
        "task_evidence": evidence,
    }


@pytest.fixture(scope="module")
def plan_and_rows() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan, coverage = load_ai_assisted_v2_plan(PLAN_PATH, repository_root=ROOT)
    metadata = expected_candidate_metadata(coverage)
    rows = [_base_row(metadata[key]) for key in sorted(metadata)]
    return plan, coverage, rows


def test_v2_plan_and_complete_synthetic_candidate_pass(
    plan_and_rows: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
) -> None:
    plan, coverage, rows = plan_and_rows
    audit = audit_ai_assisted_v2_rows(rows, coverage_plan=coverage, plan=plan)
    assert audit["records"] == 320
    assert audit["quality_gate_results"] == {
        "assistant_response_exact_cross_split_overlap": 0,
        "assistant_response_normalized_cross_split_overlap": 0,
        "context_required_term_coverage": 1.0,
        "identity_branding_on_capability_records": 0,
        "near_duplicate_cross_split_pairs": 0,
        "persistent_memory_claim_records": 0,
        "template_family_cross_split_overlap": 0,
    }
    owner_sample = select_owner_review_sample_v2(rows)
    candidates = to_sft_candidate_records_v2(
        rows,
        plan=plan,
        source_revision=audit["candidate_source_revision"],
    )
    manifest = build_ai_assisted_v2_manifest(
        audit=audit,
        candidate_records=candidates,
        owner_sample=owner_sample,
    )
    assert len(owner_sample) == 32
    assert len(candidates) == 320
    assert manifest["production_eligible"] is False


def test_v2_rejects_cross_split_response_leakage(
    plan_and_rows: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
) -> None:
    plan, coverage, original = plan_and_rows
    rows = copy.deepcopy(original)
    skill_rows = [row for row in rows if row["skill_id"] == "greeting_and_small_talk"]
    train = next(row for row in skill_rows if row["split"] == "train")
    validation = next(
        row
        for row in skill_rows
        if row["split"] == "validation" and row["language"] == train["language"]
    )
    validation["messages"] = copy.deepcopy(train["messages"])
    validation["task_evidence"] = copy.deepcopy(train["task_evidence"])
    with pytest.raises(ValueError, match="duplicate conversation|response leakage"):
        audit_ai_assisted_v2_rows(rows, coverage_plan=coverage, plan=plan)


def test_v2_rejects_false_persistent_memory_claim(
    plan_and_rows: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
) -> None:
    plan, coverage, original = plan_and_rows
    metadata = expected_candidate_metadata(coverage)
    row = copy.deepcopy(
        next(
            item
            for item in original
            if item["skill_id"] == "multi_turn_context_and_correction"
            and item["language"] == "en"
        )
    )
    row["messages"][-1]["content"] += " I'll remember this."
    with pytest.raises(ValueError, match="persistent-memory"):
        validate_ai_assisted_v2_row(
            row,
            expected=metadata[row["record_id"]],
            plan=plan,
        )


def test_v2_rejects_irrelevant_company_branding_in_capability_answer(
    plan_and_rows: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]],
) -> None:
    plan, coverage, original = plan_and_rows
    metadata = expected_candidate_metadata(coverage)
    row = copy.deepcopy(
        next(
            item
            for item in original
            if item["skill_id"] == "identity_and_capability_boundaries"
            and item["split"] == "validation"
            and item["language"] == "en"
        )
    )
    row["messages"][-1]["content"] += " Afnan Technologies Inc."
    with pytest.raises(ValueError, match="irrelevant branding"):
        validate_ai_assisted_v2_row(
            row,
            expected=metadata[row["record_id"]],
            plan=plan,
        )


def test_v2_evidence_keeps_owner_gate_pending() -> None:
    evidence = load_ai_assisted_v2_evidence(
        ROOT / "evidence/lumi-4b-ai-assisted-sft-pilot-candidate-v0.2.json"
    )
    assert evidence["candidate"]["records"] == 320
    assert evidence["owner_gate"]["status"] == "pending"
    assert evidence["production_eligible"] is False
