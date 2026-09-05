from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_scale_canary_preflight_v2 import (
    build_gpt_oss_scale_canary_preflight_report,
    build_gpt_oss_scale_canary_preflight_verification,
    load_gpt_oss_scale_canary_preflight_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-scale-canary-preflight-v0.2.json"
REPORT = ROOT / "evidence/lumi-4b-sft-gpt-oss-scale-canary-preflight-v0.2.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-gpt-oss-scale-canary-preflight-verification-v0.2.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("gpt-oss scale-canary preflight plan is not materialized")
    return load_gpt_oss_scale_canary_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_local_only_and_nonexecuting() -> None:
    plan = _plan()
    assert plan["limits"]["candidate_records_generated"] == 0
    assert plan["limits"]["gpu_use"] == 0
    assert plan["authorization"]["remote_access_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_scale_canary_is_1000_rows_and_tool_inclusive() -> None:
    contract = _plan()["canary_contract"]
    assert contract["candidate_records"] == 1_000
    assert contract["train_candidate_slots"] == 800
    assert contract["validation_candidate_slots"] == 200
    assert contract["schedule"]["task_allocation"]["tool_api"] == 266
    assert contract["failure_policy"] == (
        "aggregate_cell_report_fail_closed_no_one_row_repair_loop"
    )


def test_report_and_verification_reproduce() -> None:
    plan = _plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("gpt-oss scale-canary preflight evidence is not materialized")
    observed = _load(REPORT)
    rebuilt = build_gpt_oss_scale_canary_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == observed
    expected = build_gpt_oss_scale_canary_preflight_verification(
        plan=plan,
        report=observed,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    serialized = REPORT.read_text(encoding="utf-8") + VERIFICATION.read_text(
        encoding="utf-8"
    )
    for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
        assert forbidden not in serialized
