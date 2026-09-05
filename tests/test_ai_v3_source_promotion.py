from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_ai_v3_source_promotion import (
    SUCCESSOR_SHA256,
    build_source_promotion_receipt,
    file_sha256,
    load_ai_v3_source_promotion_plan,
    reconstruct_authorized_successor,
    verify_source_promotion_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-ai-v3-source-promotion-v0.1.json"


def test_authorized_plan_is_exact_and_nonproduction() -> None:
    plan = load_ai_v3_source_promotion_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["candidate_source_registry_promotion_authorized"]
    assert plan["authorization"]["candidate_row_modification_authorized"] is False
    assert plan["authorization"]["production_dataset_promotion_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["owner_approval"]["maximum_minutes"] == 30
    assert plan["owner_approval"]["projected_successor_sha256"] == SUCCESSOR_SHA256


def test_authorized_successor_and_receipt_reconstruct_exactly() -> None:
    plan = load_ai_v3_source_promotion_plan(PLAN, repository_root=ROOT)
    predecessor, successor, preflight = reconstruct_authorized_successor(
        plan,
        repository_root=ROOT,
    )
    assert successor["sources"][:12] == predecessor["sources"]
    assert len(successor["sources"]) == 13
    assert successor["sources"][12] == plan["source_after"]
    receipt = build_source_promotion_receipt(
        plan,
        plan_file_sha256=file_sha256(PLAN),
        predecessor=predecessor,
        successor=successor,
        preflight_report=preflight,
    )
    verify_source_promotion_receipt(receipt)
    assert receipt["candidate_source_registry_promotions"] == 1
    assert receipt["production_dataset_promotions"] == 0
    assert receipt["record_provenance"]["per_record_human_review_status_counts"] == {
        "owner_spot_check_pending": 320
    }


def test_plan_rejects_authorization_and_successor_drift(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["authorization"]["paid_gpu_authorized"] = True
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_ai_v3_source_promotion_plan(path, repository_root=ROOT)

    changed = copy.deepcopy(value)
    changed["owner_approval"]["projected_successor_sha256"] = "0" * 64
    path = tmp_path / "successor.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="owner approval boundary"):
        load_ai_v3_source_promotion_plan(path, repository_root=ROOT)
