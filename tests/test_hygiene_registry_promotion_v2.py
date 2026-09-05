from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit import resolve_binding
from lumi.posttraining_hygiene_registry_promotion_v2 import (
    EXPECTED_LIMITS,
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_SHA256,
    build_and_validate_registry_v14_successor,
    build_registry_v14_promotion_receipt,
    load_hygiene_registry_v14_promotion_plan,
    validate_registry_v14_promotion_receipt,
    verify_v14_promotion_prerequisites,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-hygiene-registry-v0.14-promotion-v0.1.json"
SUCCESSOR = ROOT / "configs/lumi-4b-sft-source-registry-v0.14.json"
RECEIPT = ROOT / "evidence/lumi-4b-sft-hygiene-registry-v0.14-promotion-v0.1.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("registry-v0.14 promotion plan is not materialized")
    return load_hygiene_registry_v14_promotion_plan(PLAN, repository_root=ROOT)


def test_registry_v14_promotion_is_bounded_cpu_only_and_nonproduction() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["source_registry_promotion_authorized"] is True
    assert plan["authorization"]["predecessor_registry_mutation_authorized"] is False
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_registry_v14_reconstructs_with_exactly_two_metadata_updates() -> None:
    plan = _load_plan()
    _, _, predecessor = verify_v14_promotion_prerequisites(
        plan=plan,
        repository_root=ROOT,
    )
    data_contract_path = resolve_binding(
        ROOT, plan["bindings"]["data_contract"], name="data_contract"
    )
    successor, changed = build_and_validate_registry_v14_successor(
        predecessor=predecessor,
        data_contract_path=data_contract_path,
    )
    assert [item["source_id"] for item in changed] == [
        "openassistant_oasst1",
        "openassistant_oasst2",
    ]
    assert len(successor["sources"]) == len(predecessor["sources"]) == 15


def test_registry_v14_and_receipt_reproduce_when_present() -> None:
    plan = _load_plan()
    if not SUCCESSOR.is_file() or not RECEIPT.is_file():
        pytest.skip("registry-v0.14 promotion is not materialized")
    preflight_report, _, predecessor = verify_v14_promotion_prerequisites(
        plan=plan,
        repository_root=ROOT,
    )
    data_contract_path = resolve_binding(
        ROOT, plan["bindings"]["data_contract"], name="data_contract"
    )
    expected_successor, changed = build_and_validate_registry_v14_successor(
        predecessor=predecessor,
        data_contract_path=data_contract_path,
    )
    assert _load(SUCCESSOR) == expected_successor
    assert file_sha256(SUCCESSOR) == SUCCESSOR_REGISTRY_SHA256
    assert SUCCESSOR.stat().st_size == SUCCESSOR_REGISTRY_BYTES
    expected_receipt = build_registry_v14_promotion_receipt(
        plan_file_sha256=file_sha256(PLAN),
        preflight_report=preflight_report,
        changed_sources=changed,
    )
    observed_receipt = _load(RECEIPT)
    assert observed_receipt == expected_receipt
    validate_registry_v14_promotion_receipt(observed_receipt)
