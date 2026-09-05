from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_closure_source_promotion_preflight import (
    REPORT_STATUS,
    build_preflight_report,
    load_preflight_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-closure-source-promotion-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-closure-source-promotion-preflight-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("closure source-promotion preflight plan is not materialized")
    return load_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_cpu_only_non_authorizing_and_exactly_bounded() -> None:
    plan = _load_plan()
    assert plan["authorization"]["cpu_only"] is True
    assert plan["authorization"]["source_registry_promotion_authorized"] is False
    assert plan["authorization"]["dataset_seal_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["limits"]["candidate_records"] == 21_183


def test_preflight_fails_closed_and_preserves_ten_million_floor() -> None:
    plan = _load_plan()
    report = build_preflight_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["status"] == REPORT_STATUS
    assert report["checks"]["source_registry_successor_projected"] is False
    assert report["repair_projection"]["projected_total_train_supervised_tokens"] == 10_032_368
    assert report["repair_projection"]["ten_million_floor_preserved"] is True


def test_preflight_detects_exact_source_blockers() -> None:
    plan = _load_plan()
    report = build_preflight_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    blockers = {item["code"]: item for item in report["blockers"]}
    assert blockers["banglabridge_train_token_ceiling_exceeded"][
        "excess_train_supervised_tokens"
    ] == 201_756
    assert blockers[
        "banglabridge_original_and_adapted_provenance_hashes_missing"
    ]["rows_with_required_provenance"] == 0
    assert blockers["aya_constituent_upstream_rights_evidence_not_bound"][
        "constituent_count"
    ] == 3


def test_materialized_report_matches_reconstruction_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("closure source-promotion preflight report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = build_preflight_report(
        plan,
        plan_file_sha256=report["plan_file_sha256"],
        repository_root=ROOT,
    )
    assert report == expected

