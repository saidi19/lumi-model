from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_bigcodebench_promotion_preflight import (
    audit_and_build_outputs,
    load_preflight_plan,
    validate_promotion_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-bigcodebench-signature-promotion-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-bigcodebench-signature-promotion-preflight-v0.1.json"
PROMOTION = ROOT / "configs/lumi-4b-bigcodebench-signature-promotion-plan-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("BigCodeBench promotion-preflight plan is not materialized")
    return load_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_audits_candidate_without_promotion() -> None:
    plan = _load_plan()
    report, promotion = audit_and_build_outputs(plan, repository_root=ROOT)
    validate_report(report)
    validate_promotion_plan(promotion)
    assert report["audit"]["candidate_entries"] == 50_419
    assert report["audit"]["predecessor_entries_preserved_in_order"] is True
    assert report["boundaries"]["candidate_promoted"] is False
    assert promotion["authorization"]["promotion_authorized"] is False


def test_proposed_successors_are_immutable_and_hash_bound() -> None:
    plan = _load_plan()
    _report, promotion = audit_and_build_outputs(plan, repository_root=ROOT)
    registry = promotion["proposed_registry_successor"]
    quality = promotion["proposed_data_quality_successor"]
    assert registry["path"] == "data/benchmark-signatures-v0.2.jsonl.gz"
    assert registry["metadata"]["release"]["status"] == "immutable_registry_successor"
    assert "candidate_not_active" not in registry["metadata"]
    assert quality["path"] == "configs/data-quality-v0.2.json"
    assert quality["content"]["benchmark_decontamination"]["registry_sha256"] == (
        registry["file_sha256"]
    )


def test_materialized_outputs_match_reconstruction_when_present() -> None:
    plan = _load_plan()
    expected_report, expected_promotion = audit_and_build_outputs(
        plan,
        repository_root=ROOT,
    )
    if not REPORT.is_file() or not PROMOTION.is_file():
        pytest.skip("BigCodeBench promotion-preflight outputs are not materialized")
    assert json.loads(REPORT.read_text(encoding="utf-8")) == expected_report
    assert json.loads(PROMOTION.read_text(encoding="utf-8")) == expected_promotion
