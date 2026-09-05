from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_aya_source_promotion import (
    EXPECTED_SOURCE_AFTER,
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    resolve_promotion_binding,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)
from lumi.posttraining_first_party import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-aya-source-promotion-v0.1.json"
PREDECESSOR = ROOT / "configs" / "lumi-4b-sft-source-registry-v0.3.json"
SUCCESSOR = ROOT / "configs" / "lumi-4b-sft-source-registry-v0.4.json"
RECEIPT = ROOT / "evidence" / "lumi-4b-sft-aya-source-promotion-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prerequisites(plan: dict) -> tuple[list[dict], dict, dict]:
    bindings = plan["bindings"]
    attribution = [
        json.loads(line)
        for line in resolve_promotion_binding(ROOT, bindings["attribution_manifest"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    materialization = _load(resolve_promotion_binding(ROOT, bindings["materialization_report"]))
    owner_receipt = _load(resolve_promotion_binding(ROOT, bindings["owner_review_receipt"]))
    return attribution, materialization, owner_receipt


def test_exact_reviewed_aya_prerequisites_pass() -> None:
    plan = load_source_promotion_plan(PLAN, repository_root=ROOT)
    attribution, materialization, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        attribution_rows=attribution,
        materialization_report=materialization,
        owner_review_receipt=owner_receipt,
    )
    assert plan["curated_metrics"]["accepted_records"] == 1478
    assert plan["curated_metrics"]["supervised_tokens"] == 577453
    assert plan["authorization"]["production_sft_authorized"] is False


def test_successor_changes_only_the_existing_aya_entry() -> None:
    plan = load_source_promotion_plan(PLAN, repository_root=ROOT)
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    assert file_sha256(PREDECESSOR) == (
        "e85f591f2a93b1384ed9ce4ee3da0e067bcfc07a1ade2e0b170eb875c7677914"
    )
    expected = build_successor_registry(predecessor_value, plan=plan)
    observed = _load(SUCCESSOR)
    assert observed == expected
    assert observed["sources"][1] == EXPECTED_SOURCE_AFTER
    assert observed["sources"][0] == predecessor_value["sources"][0]
    assert observed["sources"][2:] == predecessor_value["sources"][2:]
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=observed,
        successor_file_sha256=file_sha256(SUCCESSOR),
    )
    source = successor.by_id["coherelabs_aya_dataset"]
    assert source.adapter == "aya_bengali_repaired_4k_annotator_cap_v0.2"
    assert source.languages == frozenset({"bn"})
    assert source.task_categories == frozenset({"multilingual_bengali"})
    assert source.production_eligible is False
    assert len(successor.sources) == len(predecessor.sources) == 10


def test_promotion_receipt_is_reproducible_raw_free_and_nonproduction() -> None:
    plan = load_source_promotion_plan(PLAN, repository_root=ROOT)
    attribution, materialization, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        attribution_rows=attribution,
        materialization_report=materialization,
        owner_review_receipt=owner_receipt,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    successor_value = _load(SUCCESSOR)
    successor_payload = json_file_bytes(successor_value)
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=hashlib.sha256(successor_payload).hexdigest(),
    )
    expected = build_source_promotion_receipt(
        owner_review_receipt=owner_receipt,
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        predecessor=predecessor,
        successor=successor,
    )
    observed = _load(RECEIPT)
    assert observed == expected
    verify_source_promotion_receipt(observed)
    assert observed["candidate_source_registry_promotions"] == 1
    assert observed["production_dataset_promotions"] == 0
    assert observed["active_production_data_contract_updated"] is False


def test_rejected_or_tampered_owner_gate_cannot_be_promoted() -> None:
    plan = load_source_promotion_plan(PLAN, repository_root=ROOT)
    attribution, materialization, owner_receipt = _prerequisites(plan)
    changed = copy.deepcopy(owner_receipt)
    changed["rejected_records"] = 1
    changed["all_sampled_rows_accepted"] = False
    with pytest.raises(ValueError, match="receipt_sha256 changed"):
        verify_source_promotion_prerequisites(
            attribution_rows=attribution,
            materialization_report=materialization,
            owner_review_receipt=changed,
        )


def test_active_data_contract_remains_on_original_registry() -> None:
    contract = _load(ROOT / "configs" / "lumi-4b-sft-data-contract-v0.1.json")
    assert contract["source_registry"]["path"] == ("configs/lumi-4b-sft-source-registry-v0.1.json")
    assert contract["source_registry"]["sha256"] == (
        "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9"
    )
