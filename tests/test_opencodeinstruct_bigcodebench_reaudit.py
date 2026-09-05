from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_bigcodebench_signatures import sha256_value
from lumi.posttraining_opencodeinstruct_bigcodebench_reaudit import (
    build_report,
    load_reaudit_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-opencodeinstruct-bigcodebench-reaudit-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-opencodeinstruct-bigcodebench-reaudit-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("OpenCodeInstruct BigCodeBench re-audit plan is not materialized")
    return load_reaudit_plan(PLAN, repository_root=ROOT)


def test_reaudit_is_cpu_only_and_non_mutating() -> None:
    plan = _load_plan()
    assert plan["authorization"]["cpu_only_local_reaudit_authorized"] is True
    assert plan["authorization"]["candidate_modification_authorized"] is False
    assert plan["authorization"]["livecodebench_access_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False


def test_reaudit_replays_all_rows_without_bigcodebench_match() -> None:
    plan = _load_plan()
    report = build_report(plan, repository_root=ROOT)
    assert report["candidate"]["records"] == 11_100
    assert report["registry"]["bigcodebench_present"] is True
    assert report["registry"]["livecodebench_present"] is False
    assert report["results"]["matched_records"] == 0


def test_materialized_report_is_sealed_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("OpenCodeInstruct BigCodeBench report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    assert sha256_value(unsigned) == report["report_sha256"]
