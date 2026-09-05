from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_dmath_source_promotion import (
    EXPECTED_SOURCE_AFTER,
    build_successor_registry,
    load_source_promotion_plan,
    validate_successor_registry,
    verify_source_promotion_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-dmath-source-promotion-v0.1.json"
SUCCESSOR_PATH = ROOT / "configs/lumi-4b-sft-source-registry-v0.5.json"
RECEIPT_PATH = ROOT / "evidence/lumi-4b-sft-dmath-source-promotion-v0.1.json"


def _predecessor() -> dict:
    return {
        "authorization": {
            "bulk_data_download_authorized": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "metadata_and_source_admissibility_audit_only",
        },
        "observed_at": "2026-08-06",
        "schema_version": "lumi.sft.sources.v0.1",
        "sources": [{"source_id": f"source_{index}"} for index in range(10)],
    }


def test_successor_appends_only_dmath() -> None:
    predecessor = _predecessor()
    successor = build_successor_registry(
        predecessor,
        plan={"source_after": EXPECTED_SOURCE_AFTER},
    )
    assert successor["sources"][:10] == predecessor["sources"]
    assert successor["sources"][10] == EXPECTED_SOURCE_AFTER
    assert len(successor["sources"]) == 11


def test_successor_rejects_existing_dmath() -> None:
    predecessor = _predecessor()
    predecessor["sources"][3] = copy.deepcopy(EXPECTED_SOURCE_AFTER)
    with pytest.raises(ValueError, match="already exists"):
        build_successor_registry(
            predecessor,
            plan={"source_after": EXPECTED_SOURCE_AFTER},
        )


def test_exact_plan_loads_when_available() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DMath promotion plan is not materialized")
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["limits"]["added_sources"] == 1
    assert plan["authorization"]["paid_gpu_authorized"] is False


def test_actual_successor_and_receipt_are_nonproduction_when_available() -> None:
    if not (PLAN_PATH.exists() and SUCCESSOR_PATH.exists() and RECEIPT_PATH.exists()):
        pytest.skip("DMath promotion outputs are not materialized")
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    policy = load_sft_policy(ROOT / plan["bindings"]["data_contract"]["path"])
    predecessor = json.loads(
        (ROOT / plan["bindings"]["predecessor_registry"]["path"]).read_text()
    )
    successor = json.loads(SUCCESSOR_PATH.read_text())
    parsed = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor,
        successor_value=successor,
        successor_file_sha256="a" * 64,
    )
    assert parsed.by_id["dmath"].status == "canary_eligible"
    receipt = json.loads(RECEIPT_PATH.read_text())
    verify_source_promotion_receipt(receipt)
    assert receipt["production_eligible"] is False
    assert receipt["active_production_data_contract_updated"] is False
