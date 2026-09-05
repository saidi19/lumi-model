from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v7 import (
    EXPECTED_BENCHMARK_REGISTRY_SHA256,
    EXPECTED_RECORDS,
    EXPECTED_REGISTRY_SHA256,
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_VALIDATION_TOKENS,
    build_candidate_mix_reaudit_v7_verification,
    load_candidate_mix_reaudit_v7_plan,
    validate_candidate_mix_reaudit_v7_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.7.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.7.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.7.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate-mix v0.7 plan is not materialized")
    return load_candidate_mix_reaudit_v7_plan(PLAN, repository_root=ROOT)


def test_v7_plan_is_bounded_local_and_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["limits"] == {
        "maximum_candidate_records_read": 300_000,
        "maximum_network_bytes": 0,
        "maximum_runtime_minutes": 180,
    }
    assert plan["bindings"]["source_registry"]["sha256"] == (
        EXPECTED_REGISTRY_SHA256
    )
    assert plan["benchmark_decontamination"]["registry_sha256"] == (
        EXPECTED_BENCHMARK_REGISTRY_SHA256
    )
    assert plan["authorization"]["production_sft_authorized"] is False


def test_v7_report_has_clean_exact_expected_totals_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("candidate-mix v0.7 report is not materialized")
    report = _load(REPORT)
    validate_candidate_mix_reaudit_v7_report(report)
    pool = report["candidate_pool"]
    assert pool["candidate_records_read"] == EXPECTED_RECORDS
    assert pool["accepted_train_supervised_tokens"] == EXPECTED_TRAIN_TOKENS
    assert pool["registry_admissible_validation_supervised_tokens"] == (
        EXPECTED_VALIDATION_TOKENS
    )
    assert report["generic_sft_validation"]["rejected"] == 0
    assert report["quarantine"]["total_records"] == 0
    assert report["exact_duplicate_audit"]["duplicate_groups"] == 0
    assert report["near_duplicate_audit"]["multi_record_components"] == 0


def test_v7_verification_reproduces_and_outputs_are_raw_text_free_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate-mix v0.7 verification is not materialized")
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_v7_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"'):
            assert forbidden not in serialized
