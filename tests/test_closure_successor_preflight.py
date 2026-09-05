from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_closure_successor_preflight import (
    REPORT_STATUS,
    build_preflight_report,
    load_preflight_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-closure-successor-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-closure-successor-preflight-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("closure-successor preflight plan is not materialized")
    return load_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_nonmutating_and_exactly_bounded() -> None:
    plan = _load_plan()
    assert plan["authorization"]["cpu_only"] is True
    assert plan["authorization"]["registry_change_authorized"] is False
    assert plan["authorization"]["dataset_seal_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False
    assert plan["limits"]["candidate_records"] == 19_654


def test_projection_replays_repaired_bundle_and_stays_fail_closed() -> None:
    plan = _load_plan()
    report = build_preflight_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["status"] == REPORT_STATUS
    assert report["bundle"]["train_supervised_tokens"] == 2_375_699
    assert report["projection"]["train_supervised_tokens"] == 9_632_369
    assert report["projection"]["total_floor_gap"] == 367_631
    assert report["projection"]["source_family_cap_violations"] == ["nvidia_family"]


def test_auxiliary_transliteration_is_not_primary_behavior_proof() -> None:
    plan = _load_plan()
    report = build_preflight_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    assert report["bundle"]["auxiliary_transliteration_train_supervised_tokens"] == 300_000
    assert (
        report["projection"]["behavior_language_floor_gaps"]["banglish"]
        > (report["projection"]["accounting_language_floor_gaps"]["banglish"])
    )
    assert report["checks"]["banglatlit_tracked_as_auxiliary_not_primary_behavior_proof"] is True


def test_materialized_report_matches_reconstruction_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("closure-successor report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = build_preflight_report(
        plan,
        plan_file_sha256=report["plan_file_sha256"],
        repository_root=ROOT,
    )
    assert report == expected
