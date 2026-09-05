from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_deepmind_scale_generation import file_sha256
from lumi.posttraining_deepmind_scale_promotion_preflight import (
    EXPECTED_PROJECTION,
    EXPECTED_SOURCE_AFTER,
    build_preflight_report,
    build_proposed_successor,
    load_preflight_plan,
    validate_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-deepmind-scale-promotion-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-deepmind-scale-promotion-preflight-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_successor_replaces_only_existing_deepmind_entry() -> None:
    predecessor = _load(ROOT / "configs/lumi-4b-sft-source-registry-v0.9.json")
    successor, before = build_proposed_successor(
        predecessor, source_after=EXPECTED_SOURCE_AFTER
    )
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(predecessor["sources"], successor["sources"], strict=True)
        )
        if left != right
    ]
    assert len(predecessor["sources"]) == len(successor["sources"]) == 13
    assert changed == [11]
    assert before["adapter"] == "deepmind_mathematics_fixed_seed_verified_v0.1"
    assert successor["sources"][11] == EXPECTED_SOURCE_AFTER


def test_materialized_preflight_reproduces_and_remains_nonproduction() -> None:
    plan = load_preflight_plan(PLAN, repository_root=ROOT)
    rebuilt = build_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert observed == rebuilt
    validate_preflight_report(observed)
    assert observed["projected_candidate_mix"] == EXPECTED_PROJECTION
    assert observed["projected_candidate_mix"]["source_cap_passed"] is False
    assert observed["checks"]["candidate_registry_promotions"] == 0
    assert observed["raw_training_text_retained"] is False
