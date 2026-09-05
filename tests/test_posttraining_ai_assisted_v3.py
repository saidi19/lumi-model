from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lumi.posttraining_ai_assisted import expected_candidate_metadata
from lumi.posttraining_ai_assisted_v3 import (
    CalculatorExpressionError,
    build_ai_assisted_v3_evidence,
    evaluate_integer_expression,
    load_ai_assisted_v3_plan,
    validate_ai_assisted_v3_row,
)
from lumi.posttraining_format import encode_sft_messages, load_chat_template

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-ai-assisted-sft-pilot-v0.3.json"
TEMPLATE_PATH = ROOT / "configs/lumi-4b-chat-template-v0.1.json"


@pytest.fixture(scope="module")
def plan_and_metadata() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    plan, coverage = load_ai_assisted_v3_plan(PLAN_PATH, repository_root=ROOT)
    return plan, expected_candidate_metadata(coverage)


def _metadata(
    metadata: Mapping[str, Mapping[str, str]],
    *,
    skill: str,
    language: str,
) -> Mapping[str, str]:
    return next(
        value
        for value in metadata.values()
        if value["skill_id"] == skill and value["language"] == language
    )


def _row(
    metadata: Mapping[str, str],
    *,
    messages: list[dict[str, str]],
    task_evidence: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    return {
        "automated_review": {
            "calculator_expression_verified": True,
            "context_grounding_passed": True,
            "final_assistant_grounding_passed": True,
            "identity_tool_claims_passed": True,
            "language_mode_passed": True,
            "message_sequence_passed": True,
            "pii_secret_scan_passed": True,
            "quality_score": 0.97,
            "reviewer_id": "lumi-deterministic-ai-assisted-guard-v0.3",
            "safety_passed": True,
            "semantic_family_passed": True,
            "task_correctness_passed": True,
        },
        "generation": {
            "generated_at": "2026-08-06T00:00:00Z",
            "generator_id": "openai-codex-current-session",
            "human_written_from_scratch": False,
            "method": "ai_generated_owner_review_pending_semantic_family_strict_v0.3",
            "rights_confirmed": True,
            "semantic_family_id": family,
        },
        "language": metadata["language"],
        "language_mode": {
            "bn": "bengali_script",
            "en": "english",
            "multilingual": "banglish_romanized",
        }[metadata["language"]],
        "messages": messages,
        "owner_review": {
            "accepted": None,
            "reviewed_at": "",
            "reviewer_id": "",
            "status": "pending",
        },
        "record_id": metadata["record_id"],
        "schema_version": "lumi.ai-assisted-sft-row.v0.3",
        "skill_id": metadata["skill_id"],
        "split": metadata["split"],
        "task_category": metadata["task_category"],
        "task_evidence": task_evidence,
    }


def test_v3_plan_loads_and_independent_arithmetic_is_exact() -> None:
    plan, coverage = load_ai_assisted_v3_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["owner_gate"]["sample_records"] == 32
    assert sum(coverage["totals"].values()) == 320
    assert evaluate_integer_expression("(483 + 34) / 2 * 2") == 517
    with pytest.raises(CalculatorExpressionError, match="division_by_zero"):
        evaluate_integer_expression("9 / (4 - 4)")
    with pytest.raises(CalculatorExpressionError, match="non_integer_result"):
        evaluate_integer_expression("5 / 2")


def test_v3_rejects_self_consistent_but_false_calculator_result(
    plan_and_metadata: tuple[dict[str, Any], dict[str, dict[str, str]]],
) -> None:
    plan, metadata = plan_and_metadata
    expected = _metadata(metadata, skill="calculator_tool_routing", language="en")
    row = _row(
        expected,
        messages=[
            {
                "content": "Use the calculator for (483 + 34) / 2 * 2.",
                "role": "user",
            },
            {
                "content": (
                    '{"arguments":{"expression":"(483 + 34) / 2 * 2"},'
                    '"name":"calculator"}'
                ),
                "role": "tool",
            },
            {"content": '{"result":"500"}', "role": "tool_result"},
            {"content": "The verified result is 500.", "role": "assistant"},
        ],
        task_evidence={
            "expected_result": 500,
            "required_response_terms": ["500"],
            "route_kind": "success",
        },
        family="calculator_tool_routing:parenthesized-four-operation",
    )
    with pytest.raises(ValueError, match="independent evaluation"):
        validate_ai_assisted_v3_row(row, expected=expected, plan=plan)


def test_v3_requires_canonical_integer_response(
    plan_and_metadata: tuple[dict[str, Any], dict[str, dict[str, str]]],
) -> None:
    plan, metadata = plan_and_metadata
    expected = _metadata(
        metadata,
        skill="simple_arithmetic_model_answer",
        language="en",
    )
    row = _row(
        expected,
        messages=[
            {"content": "Give only the integer result: 17 + 25", "role": "user"},
            {"content": "The answer is 42.", "role": "assistant"},
        ],
        task_evidence={
            "expected_integer": 42,
            "expression": "17 + 25",
            "required_response_terms": ["42"],
        },
        family="simple_arithmetic_model_answer:add-two-integers",
    )
    with pytest.raises(ValueError, match="not canonical"):
        validate_ai_assisted_v3_row(row, expected=expected, plan=plan)


def test_v3_grounding_is_checked_only_in_final_assistant_message(
    plan_and_metadata: tuple[dict[str, Any], dict[str, dict[str, str]]],
) -> None:
    plan, metadata = plan_and_metadata
    expected = _metadata(
        metadata,
        skill="multi_turn_context_and_correction",
        language="en",
    )
    row = _row(
        expected,
        messages=[
            {"content": "Use blue for this chat's draft.", "role": "user"},
            {"content": "The draft is blue for this chat.", "role": "assistant"},
            {"content": "Correction: use green instead.", "role": "user"},
            {"content": "Understood; I updated it.", "role": "assistant"},
        ],
        task_evidence={
            "context_retained": True,
            "context_scope_explicit": True,
            "required_response_terms": ["green", "draft"],
        },
        family="multi_turn_context_and_correction:draft-color-correction",
    )
    with pytest.raises(ValueError, match="final response misses"):
        validate_ai_assisted_v3_row(row, expected=expected, plan=plan)


def test_v3_rejects_split_in_family_and_bengali_script_in_banglish(
    plan_and_metadata: tuple[dict[str, Any], dict[str, dict[str, str]]],
) -> None:
    plan, metadata = plan_and_metadata
    expected = _metadata(metadata, skill="greeting_and_small_talk", language="multilingual")
    row = _row(
        expected,
        messages=[
            {"content": "Mama, আজ ki kaj korbo?", "role": "user"},
            {"content": "Ami ekta kaj choose korte help korbo.", "role": "assistant"},
        ],
        task_evidence={"required_response_terms": ["help"]},
        family="greeting_and_small_talk:train:daily-plan",
    )
    with pytest.raises(ValueError, match="contains Bengali text"):
        validate_ai_assisted_v3_row(row, expected=expected, plan=plan)
    row = copy.deepcopy(row)
    row["messages"][0]["content"] = "Mama, aj ki kaj korbo?"
    with pytest.raises(ValueError, match="embeds the split"):
        validate_ai_assisted_v3_row(row, expected=expected, plan=plan)


class _Tokenizer:
    all_special_ids = list(range(11))
    vocabulary = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|doc|>": 3,
        "<|unk|>": 4,
        "<|system|>": 5,
        "<|user|>": 6,
        "<|assistant|>": 7,
        "<|tool|>": 8,
        "<|tool_result|>": 9,
        "<|message_end|>": 10,
    }

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        assert split_special_tokens is True
        return {
            "question": [101],
            "tool-call": [201],
            "tool-result": [301],
            "answer": [401],
        }.get(text, [999])


def test_v3_tool_invocation_is_supervised_and_tool_result_is_masked() -> None:
    encoded = encode_sft_messages(
        [
            {"content": "question", "role": "user"},
            {"content": "tool-call", "role": "tool"},
            {"content": "tool-result", "role": "tool_result"},
            {"content": "answer", "role": "assistant"},
        ],
        tokenizer=_Tokenizer(),
        template=load_chat_template(TEMPLATE_PATH),
    )
    tool = encoded.input_ids.index(8)
    result = encoded.input_ids.index(9)
    assistant = encoded.input_ids.index(7)
    assert encoded.labels[tool + 1 : result] == (201, 10)
    assert all(label == -100 for label in encoded.labels[result:assistant])
    assert encoded.labels[assistant + 1 :] == (401, 10, 2)


def test_v3_evidence_uses_actual_sft_filter_report_counts() -> None:
    evidence = build_ai_assisted_v3_evidence(
        audit={
            "automated_review_passed_records": 320,
            "candidate_source_revision": "a" * 64,
            "normalized_unique_conversations": 320,
            "quality_gate_results": {"all_passed": True},
            "records": 320,
            "report_sha256": "b" * 64,
        },
        manifest={
            "candidate_source_revision": "a" * 64,
            "candidate_sft_jsonl_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "owner_sample_records": 32,
            "owner_sample_sha256": "e" * 64,
        },
        private_artifact_path="private://lumi/sft-candidate.jsonl",
        filter_report={
            "accepted": 320,
            "rejected": 0,
            "report_sha256": "f" * 64,
        },
    )
    assert evidence["candidate"]["sft_filter_accepted_records"] == 320
    assert evidence["candidate"]["sft_filter_rejected_records"] == 0
