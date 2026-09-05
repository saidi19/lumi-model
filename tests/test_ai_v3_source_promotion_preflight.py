from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_ai_v3_source_promotion_preflight import (
    build_ai_v3_promotion_preflight_report,
    build_projected_successor,
    load_ai_v3_promotion_preflight_plan,
    verify_ai_v3_promotion_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-ai-v3-source-promotion-preflight-v0.1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_plan_is_bound_and_non_authorizing() -> None:
    plan = load_ai_v3_promotion_preflight_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    assert plan["limits"]["candidate_records"] == 320
    assert plan["limits"]["train_supervised_tokens"] == 9_314
    assert plan["limits"]["validation_supervised_tokens"] == 2_138
    assert plan["source_after"]["production_eligible"] is False
    assert "per_record_owner_review_provenance_not_materialized" in (
        plan["source_after"]["production_blockers"]
    )


def test_preflight_report_records_projection_without_authorizing_materialization() -> None:
    plan = load_ai_v3_promotion_preflight_plan(PLAN, repository_root=ROOT)
    report = build_ai_v3_promotion_preflight_report(
        plan,
        plan_file_sha256=_sha256(PLAN),
        repository_root=ROOT,
    )
    verify_ai_v3_promotion_preflight_report(report)
    assert report["candidate"]["accepted_records"] == 320
    assert report["candidate"]["owner_sample_accepted"] == 32
    assert report["candidate"]["per_record_human_review_status_counts"] == {
        "owner_spot_check_pending": 320
    }
    assert report["projected_successor"]["predecessor_source_count"] == 12
    assert report["projected_successor"]["source_count"] == 13
    assert report["checks"]["candidate_registry_change_executed"] is False
    assert report["production_eligible"] is False


def test_projection_rejects_existing_ai_v3_source() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    predecessor = json.loads(
        (ROOT / plan["bindings"]["predecessor_registry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    predecessor["sources"][0] = copy.deepcopy(plan["source_after"])
    with pytest.raises(ValueError, match="already exists"):
        build_projected_successor(predecessor, source_after=plan["source_after"])


def test_preflight_rejects_authorization_or_binding_drift(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["authorization"]["candidate_registry_change_authorized"] = True
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_ai_v3_promotion_preflight_plan(path, repository_root=ROOT)

    changed = copy.deepcopy(value)
    changed["bindings"]["accepted_records"]["sha256"] = "0" * 64
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        load_ai_v3_promotion_preflight_plan(path, repository_root=ROOT)
