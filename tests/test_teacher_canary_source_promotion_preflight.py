from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_teacher_canary_source_promotion_preflight import (
    EXPECTED_OWNER_RECEIPT_FILE_SHA256,
    build_projected_successor,
    build_promotion_preflight_report,
    file_sha256,
    load_promotion_preflight_plan,
    project_canonical_candidate,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs"
    / "lumi-4b-sft-first-party-teacher-source-promotion-preflight-v0.1.json"
)


def test_plan_is_bound_and_non_authorizing() -> None:
    plan = load_promotion_preflight_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    assert plan["limits"]["candidate_records"] == 80
    assert plan["limits"]["train_supervised_tokens"] == 2_002
    assert plan["limits"]["validation_supervised_tokens"] == 799
    assert plan["source_after"]["production_eligible"] is False


def test_preflight_projects_canonical_rows_and_registry_without_writes() -> None:
    plan = load_promotion_preflight_plan(PLAN, repository_root=ROOT)
    report = build_promotion_preflight_report(
        plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    validate_report(report)
    assert report["candidate"]["records"] == 80
    assert report["candidate"]["owner_sample_accepted"] == 32
    assert report["candidate"]["token_metrics"]["total_supervised_tokens"] == 2_801
    assert report["canonical_candidate_projection"]["filter_accepted"] == 80
    assert report["canonical_candidate_projection"]["filter_rejected"] == 0
    assert report["canonical_candidate_projection"]["materialized"] is False
    assert report["projected_successor"]["predecessor_source_count"] == 15
    assert report["projected_successor"]["source_count"] == 16
    assert report["checks"]["candidate_registry_change_executed"] is False


def test_canonical_projection_binds_owner_review_without_human_claim() -> None:
    plan = load_promotion_preflight_plan(PLAN, repository_root=ROOT)
    candidate_path = (ROOT / plan["bindings"]["candidate_rows"]["path"]).resolve()
    rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    projected = project_canonical_candidate(
        rows,
        owner_receipt_file_sha256=EXPECTED_OWNER_RECEIPT_FILE_SHA256,
    )
    assert len(projected) == 80
    assert all(row["schema_version"] == "lumi.sft.v0.1" for row in projected)
    assert all(row["provenance"]["ai_assisted"] is True for row in projected)
    assert all(row["provenance"]["human_authored"] is False for row in projected)
    assert all(
        row["provenance"]["owner_review_claim"]
        == "ai_generated_plus_owner_reviewed_not_human_written"
        for row in projected
    )


def test_projection_rejects_existing_teacher_source() -> None:
    plan = load_promotion_preflight_plan(PLAN, repository_root=ROOT)
    predecessor = json.loads(
        (ROOT / plan["bindings"]["predecessor_registry"]["path"]).read_text()
    )
    predecessor["sources"][0] = copy.deepcopy(plan["source_after"])
    with pytest.raises(ValueError, match="already exists"):
        build_projected_successor(predecessor, source_after=plan["source_after"])


def test_preflight_rejects_authorization_or_binding_drift(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["authorization"]["source_registry_promotion_authorized"] = True
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_promotion_preflight_plan(path, repository_root=ROOT)

    changed = copy.deepcopy(value)
    changed["bindings"]["candidate_rows"]["sha256"] = "0" * 64
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        load_promotion_preflight_plan(path, repository_root=ROOT)
