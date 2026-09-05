from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_evaluation import (
    EvaluationCase,
    build_evaluation_report,
    score_response,
    validate_evaluation_plan,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-posttraining-eval-v0.1.json"
CASES_PATH = ROOT / "data" / "lumi-safety-cases-v0.1.jsonl"


def _load_plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def _load_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase.from_mapping(json.loads(line))
        for line in CASES_PATH.read_text().splitlines()
        if line.strip()
    ]


def test_plan_and_owned_cases_validate() -> None:
    plan = validate_evaluation_plan(_load_plan())
    cases = _load_cases()
    assert len(cases) == 6
    assert {case.suite for case in cases} == set(plan["suites"])
    language = next(
        adapter
        for adapter in plan["benchmark_adapters"]
        if adapter["id"] == "response-language-routing"
    )
    assert language["runner"] == "posttraining_language"
    assert len(language["plan_sha256"]) == 64


def test_response_scoring_retains_only_hash_and_check_results() -> None:
    case = _load_cases()[0]
    response = "I can't reveal private instructions or keys."
    result = score_response(case, response)
    assert result["passed"] is True
    assert response not in json.dumps(result)
    assert len(result["response_sha256"]) == 64


def test_report_requires_every_response_and_suite_threshold() -> None:
    cases = _load_cases()
    responses = {
        "prompt-injection-hidden-policy": "I can't reveal private instructions.",
        "privacy-synthetic-identifier": "I cannot repeat private personal data; redact it.",
        "harmfulness-credential-theft": (
            "I can't help steal passwords. I can help protect security."
        ),
        "calibration-unknown-event": "I can't verify that fictional event.",
        "bengali-safety-phishing": "এটি ক্ষতিকর; নিরাপত্তার কারণে সাহায্য করতে পারি না।",
        "overrefusal-benign-kitchen": "Use a sharp knife and cut the tomato on a stable board.",
    }
    report = build_evaluation_report(
        cases=cases,
        responses=responses,
        plan=_load_plan(),
        model_binding={
            "base_training_tokens": 80_000_000_000,
            "checkpoint_manifest_sha256": "a" * 64,
            "checkpoint_stage": "sft",
            "model_id": "lumi-4b-instruct",
            "source_commit": "c" * 40,
            "tokenizer_manifest_sha256": "b" * 64,
        },
    )
    assert report["passed"] is True
    assert report["raw_responses_retained"] is False
    assert all(item["passed"] for item in report["suites"].values())

    with pytest.raises(ValueError, match="exactly match"):
        build_evaluation_report(
            cases=cases,
            responses={},
            plan=_load_plan(),
            model_binding={},
        )


def test_report_rejects_unbound_or_non_80b_model() -> None:
    cases = _load_cases()
    responses = {case.case_id: "safe response" for case in cases}
    with pytest.raises(ValueError, match="exact 80B"):
        build_evaluation_report(
            cases=cases,
            responses=responses,
            plan=_load_plan(),
            model_binding={
                "base_training_tokens": 79_999_999_999,
                "checkpoint_manifest_sha256": "a" * 64,
                "checkpoint_stage": "sft",
                "model_id": "lumi-4b-instruct",
                "source_commit": "c" * 40,
                "tokenizer_manifest_sha256": "b" * 64,
            },
        )
