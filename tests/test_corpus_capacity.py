from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.corpus_capacity import (
    build_corpus_capacity_report,
    evaluate_corpus_capacity,
)

ROOT = Path(__file__).parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        _load("configs/corpus-capacity-gate-v0.1.json"),
        _load("configs/data-mix-v0.1.json"),
        _load("configs/component-sources-v0.1.json"),
        _load("configs/common-pile-capacity-snapshot-v0.1.json"),
        _load("configs/tokenizer-final-ab-v0.1.json"),
    )


def test_planning_capacity_passes_but_bulk_data_and_training_stay_blocked() -> None:
    report = build_corpus_capacity_report(
        ROOT / "configs" / "corpus-capacity-gate-v0.1.json"
    )
    assert report["decision"] == {
        "bounded_source_canaries": "ready",
        "bulk_corpus_materialization": "blocked_pending_measured_capacity_and_owner_approval",
        "full_4b_training": "not_authorized",
        "metadata_capacity_estimate": "pass",
        "next_action": "bounded_source_acquisition_canaries",
        "paid_compute": "not_authorized",
    }
    summary = report["summary"]
    assert summary["final_training_tokens"] == 80_000_000_000
    assert summary["minimum_category_candidate_tokens"] == 120_000_000_000
    assert summary["planning_flexible_candidate_tokens"] == 20_000_000_000
    assert summary["published_utf8_bytes"] == 1_734_967_000_000
    assert summary["tree_compressed_bytes"] == 499_817_285_384
    assert summary["estimated_published_tokens"] > 400_000_000_000
    assert summary["required_final_retention_fraction_at_planning_target"] == pytest.approx(
        4 / 7
    )
    assert len(report["sources"]) == 23
    assert all(row["planning_estimate_passes"] for row in report["categories"])
    open_web = next(row for row in report["categories"] if row["category"] == "open_web")
    assert open_web["planning_estimate_headroom_fraction"] < 0.12


def test_snapshot_revision_drift_fails_closed() -> None:
    specification, mixture, inventory, snapshot, tokenizer = _inputs()
    changed = copy.deepcopy(snapshot)
    changed["sources"][0]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision drifted"):
        evaluate_corpus_capacity(specification, mixture, inventory, changed, tokenizer)


def test_missing_category_capacity_fails_the_planning_gate() -> None:
    specification, mixture, inventory, snapshot, tokenizer = _inputs()
    changed = copy.deepcopy(snapshot)
    cccc = next(row for row in changed["sources"] if row["component"] == "cccc")
    cccc["published_utf8_bytes"] = 1
    report = evaluate_corpus_capacity(specification, mixture, inventory, changed, tokenizer)
    assert report["decision"]["metadata_capacity_estimate"] == "fail"
    assert report["decision"]["bounded_source_canaries"] == "blocked"
    assert report["decision"]["full_4b_training"] == "not_authorized"


def test_planning_contract_cannot_authorize_paid_or_training_work() -> None:
    specification, mixture, inventory, snapshot, tokenizer = _inputs()
    changed = copy.deepcopy(specification)
    changed["authorization"]["full_4b_training"] = True
    with pytest.raises(ValueError, match="must not authorize"):
        evaluate_corpus_capacity(changed, mixture, inventory, snapshot, tokenizer)
