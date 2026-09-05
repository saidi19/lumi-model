from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_constituent_rights_audit import (
    build_report,
    build_verification,
    file_sha256,
    load_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-aya-constituent-rights-audit-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-aya-constituent-rights-audit-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-aya-constituent-rights-audit-verification-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("Aya constituent-rights audit plan is not materialized")
    return load_plan(PLAN, repository_root=ROOT)


def test_audit_is_bounded_cpu_only_and_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["authorization"]["cpu_only"] is True
    assert plan["authorization"]["gpu_authorized"] is False
    assert plan["authorization"]["candidate_modification_authorized"] is False
    assert plan["authorization"]["registry_change_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False
    assert len(plan["official_objects"]) == 9
    assert sum(item["bytes"] for item in plan["official_objects"]) == 3_824_046


def test_audit_classifies_all_rows_and_enforces_upstream_rights() -> None:
    plan = _load_plan()
    report = build_report(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["candidate"]["records"] == 5_079
    assert report["candidate"]["candidate_modified"] is False
    assert report["decision_summary"]["retained_with_obligations"]["records"] == 3_995
    assert report["decision_summary"]["quarantined"]["records"] == 1_084
    decisions = {item["family_id"]: item for item in report["constituents"]}
    assert decisions["flan_gem_wikilingua_translated"]["disposition"].startswith("quarantine")
    assert decisions["wiki_split_inst_translated"]["upstream_license"] == ("cc-by-sa-4.0")
    assert decisions["joke_explaination_inst_translated"]["upstream_license"] == ("mit")


def test_materialized_report_and_verification_reproduce_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("Aya constituent-rights outputs are not materialized")
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
