from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v9 import (
    EXPECTED_BENCHMARK_REGISTRY_SHA256,
    EXPECTED_INPUT_RECORDS,
    EXPECTED_INPUT_TRAIN_TOKENS,
    EXPECTED_INPUT_VALIDATION_TOKENS,
    EXPECTED_REGISTRY_SHA256,
    build_candidate_mix_reaudit_v9_verification,
    load_candidate_mix_reaudit_v9_plan,
    validate_candidate_mix_reaudit_v9_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.9.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.9.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.9.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate-mix v0.9 plan is not materialized")
    return load_candidate_mix_reaudit_v9_plan(PLAN, repository_root=ROOT)


def test_v9_plan_is_bounded_local_and_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["limits"] == {
        "maximum_candidate_records_read": 300_000,
        "maximum_network_bytes": 0,
        "maximum_runtime_minutes": 180,
    }
    assert plan["bindings"]["source_registry"]["sha256"] == EXPECTED_REGISTRY_SHA256
    assert plan["benchmark_decontamination"]["registry_sha256"] == (
        EXPECTED_BENCHMARK_REGISTRY_SHA256
    )
    assert plan["authorization"]["production_sft_authorized"] is False


def test_v9_report_has_exact_input_and_raw_free_quarantine_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("candidate-mix v0.9 report is not materialized")
    report = _load(REPORT)
    validate_candidate_mix_reaudit_v9_report(report)
    pool = report["candidate_pool"]
    assert pool["candidate_records_read"] == EXPECTED_INPUT_RECORDS
    assert pool["input_train_supervised_tokens"] == EXPECTED_INPUT_TRAIN_TOKENS
    assert pool["input_validation_supervised_tokens"] == (EXPECTED_INPUT_VALIDATION_TOKENS)
    assert (
        report["generic_sft_validation"]["accepted"] + report["quarantine"]["total_records"]
        == EXPECTED_INPUT_RECORDS
    )


def test_v9_verification_reproduces_and_outputs_are_raw_text_free_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate-mix v0.9 verification is not materialized")
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_v9_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized
