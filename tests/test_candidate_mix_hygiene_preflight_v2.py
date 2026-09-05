from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_hygiene_preflight_v2 import (
    EXPECTED_LIMITS,
    EXPECTED_PROJECTED_RECORDS,
    EXPECTED_PROJECTED_REMOVALS,
    build_hygiene_preflight_v2_verification,
    load_hygiene_preflight_v2_plan,
    rebuild_hygiene_preflight_v2,
    validate_hygiene_preflight_v2_report,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.2.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.2.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-verification-v0.2.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate hygiene v0.2 plan is not materialized")
    return load_hygiene_preflight_v2_plan(PLAN, repository_root=ROOT)


def test_hygiene_v2_plan_is_cpu_only_and_non_mutating() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["candidate_materialization_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_hygiene_v2_report_is_exact_and_raw_text_free_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("candidate hygiene v0.2 report is not materialized")
    report = _load(REPORT)
    validate_hygiene_preflight_v2_report(report)
    assert report["projected_pool"]["records"] == EXPECTED_PROJECTED_RECORDS
    assert report["projected_pool"]["removed_records"] == (EXPECTED_PROJECTED_REMOVALS)
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        assert forbidden not in serialized


def test_hygiene_v2_verification_replays_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate hygiene v0.2 verification is not materialized")
    report = _load(REPORT)
    replayed = rebuild_hygiene_preflight_v2(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    expected = build_hygiene_preflight_v2_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
        replayed_report=replayed,
    )
    assert expected == _load(VERIFICATION)
