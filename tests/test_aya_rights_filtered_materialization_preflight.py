from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_rights_filtered_preflight import (
    build_report,
    build_verification,
    file_sha256,
    load_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-aya-rights-filtered-materialization-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-aya-rights-filtered-materialization-preflight-v0.1.json"
VERIFICATION = (
    ROOT
    / "evidence/lumi-4b-sft-aya-rights-filtered-materialization-preflight-verification-v0.1.json"
)


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("Aya rights-filtered preflight plan is not materialized")
    return load_plan(PLAN, repository_root=ROOT)


def test_preflight_is_nonmutating_and_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["authorization"]["candidate_materialization_authorized"] is False
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["gpu_authorized"] is False
    assert plan["authorization"]["registry_change_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False


def test_projection_is_exact_and_rights_filtered() -> None:
    plan = _load_plan()
    report = build_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["projection"]["input_records"] == 5_079
    assert report["projection"]["output_records"] == 3_995
    assert report["projection"]["quarantined"]["records"] == 1_084
    assert report["checks"]["candidate_not_written"] is True


def test_materialized_outputs_reproduce_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("Aya rights-filtered preflight outputs are not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = build_report(
        plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert report == expected
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    assert verification == build_verification(report, report_file_sha256=file_sha256(REPORT))
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
        assert forbidden not in serialized
