from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_opencodeinstruct_source_promotion_preflight import (
    build_projected_successor,
    build_promotion_preflight_report,
    load_promotion_preflight_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-opencodeinstruct-source-promotion-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-opencodeinstruct-source-promotion-preflight-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("OpenCodeInstruct promotion-preflight plan is not materialized")
    return load_promotion_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_non_authorizing_and_exactly_bounded() -> None:
    plan = _load_plan()
    assert plan["authorization"]["source_registry_promotion_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["limits"]["candidate_records"] == 11_100
    assert plan["limits"]["train_supervised_tokens"] == 2_100_077
    assert plan["limits"]["validation_supervised_tokens"] == 110_619


def test_preflight_projects_one_source_and_replays_candidate() -> None:
    plan = _load_plan()
    report = build_promotion_preflight_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["candidate"]["accepted_records"] == 11_100
    assert report["candidate"]["benchmark_matches"] == 0
    assert report["projected_successor"]["predecessor_source_count"] == 13
    assert report["projected_successor"]["source_count"] == 14
    assert report["checks"]["candidate_registry_change_executed"] is False


def test_projection_rejects_existing_opencodeinstruct_source() -> None:
    plan = _load_plan()
    predecessor = json.loads(
        (ROOT / plan["bindings"]["predecessor_registry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    changed = copy.deepcopy(predecessor)
    changed["sources"][0] = copy.deepcopy(plan["source_after"])
    with pytest.raises(ValueError, match="already exists"):
        build_projected_successor(changed, source_after=plan["source_after"])


def test_materialized_report_matches_reconstruction_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("OpenCodeInstruct promotion-preflight report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = build_promotion_preflight_report(
        plan,
        plan_file_sha256=report["plan_file_sha256"],
        repository_root=ROOT,
    )
    assert report == expected
