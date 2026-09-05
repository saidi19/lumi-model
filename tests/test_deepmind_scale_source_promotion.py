from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_deepmind_scale_generation import file_sha256
from lumi.posttraining_deepmind_scale_source_promotion import (
    SUCCESSOR_REGISTRY_SHA256,
    build_and_validate_successor,
    build_promotion_receipt,
    load_promotion_plan,
    resolve_binding,
    validate_promotion_receipt,
    verify_promotion_prerequisites,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-deepmind-scale-source-promotion-v0.1.json"
SUCCESSOR = ROOT / "configs/lumi-4b-sft-source-registry-v0.10.json"
RECEIPT = ROOT / "evidence/lumi-4b-sft-deepmind-scale-source-promotion-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_successor_and_receipt_reproduce() -> None:
    plan = load_promotion_plan(PLAN, repository_root=ROOT)
    preflight, owner_receipt, predecessor = verify_promotion_prerequisites(
        plan=plan, repository_root=ROOT
    )
    policy = _load(
        resolve_binding(
            ROOT, plan["bindings"]["data_contract"], name="data_contract"
        )
    )
    successor, source_before = build_and_validate_successor(
        plan=plan, predecessor=predecessor, policy=policy
    )
    assert _load(SUCCESSOR) == successor
    assert file_sha256(SUCCESSOR) == SUCCESSOR_REGISTRY_SHA256
    receipt = build_promotion_receipt(
        owner_receipt=owner_receipt,
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        preflight=preflight,
        source_before=source_before,
    )
    assert _load(RECEIPT) == receipt
    validate_promotion_receipt(receipt)


def test_only_deepmind_entry_changed_and_production_remains_blocked() -> None:
    predecessor = _load(ROOT / "configs/lumi-4b-sft-source-registry-v0.9.json")
    successor = _load(SUCCESSOR)
    changed = [
        before["source_id"]
        for before, after in zip(
            predecessor["sources"], successor["sources"], strict=True
        )
        if before != after
    ]
    assert changed == ["deepmind_mathematics_dataset"]
    deepmind = next(
        source
        for source in successor["sources"]
        if source["source_id"] == "deepmind_mathematics_dataset"
    )
    assert deepmind["production_eligible"] is False
    assert "source_share_would_exceed_0_35_cap_in_current_candidate_pool" in (
        deepmind["production_blockers"]
    )
    assert file_sha256(ROOT / "configs/lumi-4b-sft-source-registry-v0.9.json") == (
        "97b7a4399810f95d9c2e39dfe236a0129bd005abac81e067b67b8d0c0fb0baac"
    )
