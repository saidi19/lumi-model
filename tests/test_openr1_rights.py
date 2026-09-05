from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_openr1_rights import (
    EXPECTED_COUNTS,
    EXPECTED_RIGHTS_MAP,
    audit_attributions,
    load_rights_plan,
    validate_rights_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-openr1-rights-map-v0.1.json"
REPORT = ROOT / "evidence" / "lumi-4b-sft-openr1-rights-map-v0.1.json"


def _rows() -> list[dict]:
    rows = []
    index = 0
    for source, count in EXPECTED_COUNTS.items():
        for _ in range(count):
            rows.append(
                {
                    "record_id": f"visible-{index:04d}",
                    "rights_status": "unresolved_quarantined",
                    "upstream_problem_source": source,
                }
            )
            index += 1
    return rows


def test_exact_source_inventory_remains_fully_quarantined() -> None:
    audit = audit_attributions(_rows())
    assert audit["records"] == 2_656
    assert audit["source_labels"] == 7
    assert audit["rights_cleared_records"] == 0
    assert audit["unresolved_records"] == 2_656
    assert {row["source"]: row["records"] for row in audit["inventory"]} == (
        EXPECTED_COUNTS
    )


def test_unknown_source_or_cleared_status_fails_closed() -> None:
    rows = _rows()
    rows[0]["upstream_problem_source"] = "unknown"
    with pytest.raises(ValueError, match="source changed"):
        audit_attributions(rows)
    rows = _rows()
    rows[0]["rights_status"] = "cleared"
    with pytest.raises(ValueError, match="status changed"):
        audit_attributions(rows)


def test_rights_map_never_promotes_a_source_label() -> None:
    assert set(EXPECTED_RIGHTS_MAP) == set(EXPECTED_COUNTS)
    assert {
        value["decision"] for value in EXPECTED_RIGHTS_MAP.values()
    } == {"quarantine_pending_documented_downstream_rights"}


def test_exact_plan_and_report_when_materialized() -> None:
    if not PLAN.exists() or not REPORT.exists():
        pytest.skip("exact OpenR1 rights artifacts are not materialized")
    plan = load_rights_plan(PLAN, repository_root=ROOT)
    assert plan["rights_map"] == EXPECTED_RIGHTS_MAP
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_rights_report(report)
    assert report["promotion_eligible_records"] == 0
    assert report["source_audit"]["unresolved_records"] == 2_656

