from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v12 import (
    EXPECTED_INPUT_RECORDS,
    EXPECTED_INPUT_TRAIN_TOKENS,
    EXPECTED_INPUT_VALIDATION_TOKENS,
    EXPECTED_REGISTRY_SHA256,
    GPT_SOURCE_ID,
    build_candidate_mix_reaudit_v12_verification,
    load_candidate_mix_reaudit_v12_plan,
    validate_candidate_mix_reaudit_v12_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-reaudit-v0.12.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.12.json"
VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.12.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate-mix v0.12 plan is not materialized")
    return load_candidate_mix_reaudit_v12_plan(PLAN, repository_root=ROOT)


def test_v12_plan_is_local_bounded_and_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["limits"] == {
        "maximum_candidate_records_per_pass": 264_981,
        "maximum_full_audit_passes": 2,
        "maximum_network_bytes": 0,
        "maximum_private_write_bytes": 2_097_152,
        "maximum_runtime_minutes": 180,
        "maximum_total_candidate_row_reads": 529_962,
    }
    assert plan["bindings"]["source_registry"]["sha256"] == EXPECTED_REGISTRY_SHA256
    assert plan["source_addition"]["source_id"] == GPT_SOURCE_ID
    assert plan["authorization"]["production_sft_authorized"] is False


def test_v12_report_reaudits_exact_registry_v17_pool_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("candidate-mix v0.12 report is not materialized")
    report = _load(REPORT)
    validate_candidate_mix_reaudit_v12_report(report)
    pool = report["candidate_pool"]
    assert pool["candidate_records_read"] == EXPECTED_INPUT_RECORDS
    assert pool["input_train_supervised_tokens"] == EXPECTED_INPUT_TRAIN_TOKENS
    assert pool["input_validation_supervised_tokens"] == EXPECTED_INPUT_VALIDATION_TOKENS
    assert report["source_metrics"][GPT_SOURCE_ID]["records"] == 984


def test_v12_verification_is_sealed_and_raw_text_free_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate-mix v0.12 verification is not materialized")
    report = _load(REPORT)
    expected = build_candidate_mix_reaudit_v12_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    for path in (REPORT, VERIFICATION):
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ('"messages"', '"content"', '"assistant_response"', '"user_prompt"'):
            assert forbidden not in serialized
