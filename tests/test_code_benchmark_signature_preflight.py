from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_code_benchmark_preflight import (
    build_preflight_outputs,
    load_preflight_plan,
    validate_acquisition_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-code-benchmark-signature-source-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-code-benchmark-signature-source-preflight-v0.1.json"
ACQUISITION = ROOT / "configs/lumi-4b-code-benchmark-signature-acquisition-plan-v0.1.json"


def _load_actual_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("code benchmark source-preflight plan is not materialized")
    return load_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_metadata_only_and_fail_closed() -> None:
    plan = _load_actual_plan()
    report, acquisition = build_preflight_outputs(plan, repository_root=ROOT)
    validate_report(report)
    validate_acquisition_plan(acquisition)
    assert report["benchmark_data_downloaded"] is False
    assert report["solutions_or_tests_downloaded"] is False
    assert report["signatures_created"] == 0
    assert report["captured_metadata"]["payload_bytes"] == 135_458
    assert report["production_eligible"] is False


def test_rights_and_acquisition_decisions_are_narrow() -> None:
    plan = _load_actual_plan()
    _report, acquisition = build_preflight_outputs(plan, repository_root=ROOT)
    assert acquisition["bigcodebench"]["decision"] == (
        "eligible_for_separate_private_prompt_only_hash_acquisition"
    )
    assert acquisition["livecodebench"]["decision"] == (
        "blocked_pending_dataset_content_license_clarification"
    )
    assert acquisition["authorization"]["acquisition_authorized"] is False
    assert acquisition["bigcodebench"]["prompt_fields"] == [
        "code_prompt",
        "complete_prompt",
        "instruct_prompt",
    ]


def test_materialized_outputs_match_reconstruction_when_present() -> None:
    plan = _load_actual_plan()
    expected_report, expected_acquisition = build_preflight_outputs(
        plan, repository_root=ROOT
    )
    if not REPORT.is_file() or not ACQUISITION.is_file():
        pytest.skip("code benchmark source-preflight outputs are not materialized")
    assert json.loads(REPORT.read_text(encoding="utf-8")) == expected_report
    assert json.loads(ACQUISITION.read_text(encoding="utf-8")) == expected_acquisition
