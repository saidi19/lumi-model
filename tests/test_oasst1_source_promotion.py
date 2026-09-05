from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_oasst1_source_promotion import (
    EXPECTED_SOURCE_AFTER,
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-sft-oasst1-source-promotion-v0.1.json"
PREDECESSOR_PATH = ROOT / "configs" / "lumi-4b-sft-source-registry-v0.2.json"
SUCCESSOR_PATH = ROOT / "configs" / "lumi-4b-sft-source-registry-v0.3.json"
RECEIPT_PATH = ROOT / "evidence" / "lumi-4b-sft-oasst1-source-promotion-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prerequisites(plan: dict) -> tuple[dict, dict, dict]:
    bindings = plan["bindings"]
    report = _load((ROOT / bindings["curation_report"]["path"]).resolve())
    verification = _load(
        (ROOT / bindings["curation_verification"]["path"]).resolve()
    )
    owner_receipt = _load(
        (ROOT / bindings["owner_review_receipt"]["path"]).resolve()
    )
    return report, verification, owner_receipt


def test_exact_reviewed_oasst1_prerequisites_pass() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    report, verification, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        curation_report=report,
        curation_verification=verification,
        owner_review_receipt=owner_receipt,
        plan=plan,
    )
    assert plan["curated_metrics"]["accepted_records"] == 1745
    assert plan["curated_metrics"]["supervised_tokens"] == 477456
    assert plan["authorization"]["production_sft_authorized"] is False


def test_successor_changes_only_the_existing_oasst1_entry() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    assert file_sha256(PREDECESSOR_PATH) == (
        "f686668fbb05f0d7930b5047560543ce113fcfb97c423fbe840f65d8a1058cdd"
    )
    expected_successor = build_successor_registry(predecessor_value, plan=plan)
    successor_value = _load(SUCCESSOR_PATH)
    assert successor_value == expected_successor
    assert successor_value["sources"][0] == EXPECTED_SOURCE_AFTER
    assert successor_value["sources"][1:] == predecessor_value["sources"][1:]
    assert len(successor_value["sources"]) == len(predecessor.sources) == 10

    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=file_sha256(SUCCESSOR_PATH),
    )
    assert successor.registry_sha256 == (
        "e85f591f2a93b1384ed9ce4ee3da0e067bcfc07a1ade2e0b170eb875c7677914"
    )
    source = successor.by_id["openassistant_oasst1"]
    assert source.adapter == "oasst1_root_direct_rank0_quality_v0.3"
    assert source.status == "canary_eligible"
    assert source.languages == frozenset({"en"})
    assert source.production_eligible is False
    assert successor.record_reasons(
        dataset_id="OpenAssistant/oasst1",
        language="en",
        license_id="apache-2.0",
        revision="fdf72ae0827c1cda404aff25b6603abec9e3399b",
        source_id="openassistant_oasst1",
        split="train",
        task_category="coding",
    ) == ()


def test_promotion_receipt_is_reproducible_raw_free_and_nonproduction() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    report, verification, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        curation_report=report,
        curation_verification=verification,
        owner_review_receipt=owner_receipt,
        plan=plan,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    successor_value = _load(SUCCESSOR_PATH)
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
        plan_file_sha256=file_sha256(PLAN_PATH),
        predecessor=predecessor,
        successor=successor,
    )
    observed = _load(RECEIPT_PATH)
    assert observed == expected
    verify_source_promotion_receipt(observed)
    assert observed["candidate_source_registry_promotions"] == 1
    assert observed["production_dataset_promotions"] == 0
    assert observed["active_production_data_contract_updated"] is False
    assert observed["raw_training_text_retained"] is False


def test_rejected_or_tampered_owner_gate_cannot_be_promoted() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    report, verification, owner_receipt = _prerequisites(plan)
    changed = copy.deepcopy(owner_receipt)
    changed["rejected_records"] = 1
    changed["all_sampled_rows_accepted"] = False
    with pytest.raises(ValueError, match="receipt_sha256 changed"):
        verify_source_promotion_prerequisites(
            curation_report=report,
            curation_verification=verification,
            owner_review_receipt=changed,
            plan=plan,
        )


def test_active_data_contract_remains_on_the_original_registry() -> None:
    contract = _load(ROOT / "configs" / "lumi-4b-sft-data-contract-v0.1.json")
    assert contract["source_registry"] == {
        "path": "configs/lumi-4b-sft-source-registry-v0.1.json",
        "required_for_filtering_and_seal": True,
        "schema_version": "lumi.sft.sources.v0.1",
        "sha256": "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9",
    }
