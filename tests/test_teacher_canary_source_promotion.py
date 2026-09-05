from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_teacher_canary_source_promotion import (
    CANONICAL_CANDIDATE_BYTES,
    CANONICAL_CANDIDATE_PATH,
    CANONICAL_CANDIDATE_SHA256,
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_PATH,
    SUCCESSOR_REGISTRY_SHA256,
    file_sha256,
    load_materialization_plan,
    reconstruct_materialized_values,
    validate_canonical_rows,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs"
    / "lumi-4b-sft-first-party-teacher-source-promotion-v0.1.json"
)


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("teacher-canary source-promotion plan is not materialized")
    return load_materialization_plan(PLAN, repository_root=ROOT)


def test_plan_is_exactly_bounded_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["authorization"]["canonical_candidate_materialization_authorized"]
    assert plan["authorization"]["candidate_registry_change_authorized"]
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["authorization"]["gpu_authorized"] is False
    assert plan["limits"]["candidate_records"] == 80
    assert plan["limits"]["candidate_bytes"] == 142_009
    assert plan["limits"]["registry_bytes"] == 18_859


def test_exact_candidate_and_successor_reconstruct() -> None:
    plan = _load_plan()
    candidate, successor, report = reconstruct_materialized_values(
        plan=plan,
        repository_root=ROOT,
    )
    validate_canonical_rows(candidate)
    assert len(candidate) == 80
    assert len(successor["sources"]) == 16
    assert successor["sources"][-1]["source_id"] == "lumi_first_party_teacher_canary_v0_1"
    assert report["canonical_candidate_projection"]["file_sha256"] == CANONICAL_CANDIDATE_SHA256
    assert report["projected_successor"]["sha256"] == SUCCESSOR_REGISTRY_SHA256


def test_materialized_outputs_are_byte_exact_when_present() -> None:
    _load_plan()
    candidate_path = (ROOT / CANONICAL_CANDIDATE_PATH).resolve()
    registry_path = (ROOT / SUCCESSOR_REGISTRY_PATH).resolve()
    if not candidate_path.is_file() or not registry_path.is_file():
        pytest.skip("teacher-canary materialized outputs are not present")
    assert file_sha256(candidate_path) == CANONICAL_CANDIDATE_SHA256
    assert candidate_path.stat().st_size == CANONICAL_CANDIDATE_BYTES
    assert file_sha256(registry_path) == SUCCESSOR_REGISTRY_SHA256
    assert registry_path.stat().st_size == SUCCESSOR_REGISTRY_BYTES
    assert len(candidate_path.read_text(encoding="utf-8").splitlines()) == 80
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(registry["sources"]) == 16
