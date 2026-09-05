from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_livecodebench_signatures import (
    EXPECTED_SIGNATURES,
    MAXIMUM_SIGNATURES,
    build_prompt_signatures,
    load_acquisition_authorization,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-livecodebench-research-signature-acquisition-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-livecodebench-research-signature-acquisition-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("LiveCodeBench acquisition authorization is not materialized")
    return load_acquisition_authorization(PLAN, repository_root=ROOT)


def test_plan_is_private_research_only_and_bounded() -> None:
    plan = _load_plan()
    assert plan["limits"]["maximum_network_bytes"] == 160 * 1024 * 1024
    assert plan["limits"]["maximum_prompt_signatures"] == MAXIMUM_SIGNATURES
    assert plan["authorization"]["rights_clearance_claimed"] is False
    assert plan["authorization"]["training_ingestion_authorized"] is False
    assert plan["authorization"]["gpu_authorized"] is False


def test_signature_builder_retains_only_hashes() -> None:
    rows = [
        {
            "platform": "example",
            "question_content": "Compute the sum of two integers.",
            "question_id": str(index),
            "question_title": f"Addition {index}",
            "starter_code": "def solve():\n    pass",
        }
        for index in range(175)
    ]
    signatures = build_prompt_signatures(rows)
    assert len(signatures) == EXPECTED_SIGNATURES
    assert all("text" not in item and "prompt" not in item for item in signatures)


def test_materialized_report_is_sealed_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("LiveCodeBench acquisition report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    assert report["production_eligible"] is False
    assert report["candidate_registry"]["candidate_not_active"] is True
