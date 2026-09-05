from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_bigcodebench_signatures import (
    MAXIMUM_SIGNATURES,
    build_prompt_signatures,
    load_acquisition_plan,
    load_prompt_rows,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-bigcodebench-prompt-signature-acquisition-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-bigcodebench-prompt-signature-acquisition-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("BigCodeBench acquisition plan is not materialized")
    return load_acquisition_plan(PLAN, repository_root=ROOT)


def test_plan_is_prompt_only_and_bounded() -> None:
    plan = _load_plan()
    assert plan["limits"]["maximum_network_bytes"] == 4 * 1024 * 1024
    assert plan["limits"]["maximum_decoded_bytes"] == 32 * 1024 * 1024
    assert plan["limits"]["maximum_prompt_signatures"] == MAXIMUM_SIGNATURES
    assert plan["authorization"]["active_registry_mutation_authorized"] is False
    assert plan["authorization"]["livecodebench_data_authorized"] is False


def test_materialized_source_replays_exact_signatures_when_present() -> None:
    plan = _load_plan()
    private_root = (ROOT / plan["output"]["private_root"]).resolve()
    source = private_root / "raw" / plan["source"]["object"]["path"]
    if not source.is_file():
        pytest.skip("BigCodeBench source object is not acquired")
    rows, decoded = load_prompt_rows(
        source,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    signatures = build_prompt_signatures(rows)
    assert len(rows) == 1_140
    assert len(signatures) == 3_420
    assert decoded["decoded_bound_bytes"] <= 32 * 1024 * 1024
    assert all("text" not in entry and "prompt" not in entry for entry in signatures)


def test_materialized_report_is_sealed_and_non_production_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("BigCodeBench acquisition report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    assert report["production_eligible"] is False
    assert report["candidate_registry"]["candidate_not_active"] is True
