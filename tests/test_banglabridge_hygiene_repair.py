from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_banglabridge_hygiene_repair import (
    build_repair,
    load_repair_plan,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-banglabridge-hygiene-repair-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-banglabridge-hygiene-repair-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("BanglaBridge hygiene-repair plan is not materialized")
    return load_repair_plan(PLAN, repository_root=ROOT)


def test_repair_is_cpu_only_non_authorizing_and_bounded() -> None:
    plan = _load_plan()
    assert plan["authorization"]["cpu_only"] is True
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["registry_change_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False
    assert plan["limits"]["maximum_train_supervised_tokens"] == 50_000


def test_repair_preserves_reviewed_rows_and_fits_ceiling() -> None:
    plan = _load_plan()
    _, manifest, report = build_repair(
        plan,
        plan_file_sha256="0" * 64,
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["checks"]["owner_reviewed_rows_retained"] == 8
    assert report["checks"]["visible_messages_changed"] == 0
    assert report["output"]["train_supervised_tokens"] <= 50_000
    assert manifest["candidate"]["splits"]["validation"]["records"] == 17
    assert len(manifest["provenance_fields_added"]) == 4


def test_materialized_report_matches_reconstruction_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("BanglaBridge hygiene-repair report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    _, _, expected = build_repair(
        plan,
        plan_file_sha256=report["plan_file_sha256"],
        repository_root=ROOT,
    )
    assert report == expected

