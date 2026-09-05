from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.data_quality import sha256_file
from lumi.posttraining_oasst2_source_promotion import (
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_SHA256,
    build_and_validate_successor,
    build_promotion_receipt,
    load_promotion_plan,
    resolve_binding,
    validate_promotion_receipt,
    verify_promotion_prerequisites,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-oasst2-source-promotion-v0.1.json"
SUCCESSOR = ROOT / "configs/lumi-4b-sft-source-registry-v0.13.json"
RECEIPT = ROOT / "evidence/lumi-4b-sft-oasst2-source-promotion-v0.1.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("OASST2 source-promotion plan is not materialized")
    return load_promotion_plan(PLAN, repository_root=ROOT)


def test_promotion_is_exactly_bounded_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["authorization"]["candidate_registry_promotion_authorized"] is True
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["authorization"]["gpu_authorized"] is False
    assert plan["limits"]["candidate_records"] == 6_426
    assert plan["limits"]["successor_sources"] == 15


def test_successor_reconstructs_with_only_one_appended_source() -> None:
    plan = _load_plan()
    _, report, predecessor = verify_promotion_prerequisites(plan=plan, repository_root=ROOT)
    policy = _load(resolve_binding(ROOT, plan["bindings"]["data_contract"], name="data_contract"))
    successor = build_and_validate_successor(plan=plan, predecessor=predecessor, policy=policy)
    assert successor["sources"][:-1] == predecessor["sources"]
    assert successor["sources"][-1]["source_id"] == "openassistant_oasst2"
    assert report["projected_successor"]["sha256"] == SUCCESSOR_REGISTRY_SHA256


def test_materialized_successor_and_receipt_reproduce_when_present() -> None:
    plan = _load_plan()
    if not SUCCESSOR.is_file() or not RECEIPT.is_file():
        pytest.skip("OASST2 source promotion is not materialized")
    _, report, predecessor = verify_promotion_prerequisites(plan=plan, repository_root=ROOT)
    policy = _load(resolve_binding(ROOT, plan["bindings"]["data_contract"], name="data_contract"))
    expected_successor = build_and_validate_successor(
        plan=plan, predecessor=predecessor, policy=policy
    )
    assert _load(SUCCESSOR) == expected_successor
    assert sha256_file(SUCCESSOR) == SUCCESSOR_REGISTRY_SHA256
    assert SUCCESSOR.stat().st_size == SUCCESSOR_REGISTRY_BYTES
    expected_receipt = build_promotion_receipt(
        plan=plan,
        plan_file_sha256=sha256_file(PLAN),
        preflight_report=report,
    )
    observed_receipt = _load(RECEIPT)
    assert observed_receipt == expected_receipt
    validate_promotion_receipt(observed_receipt)
