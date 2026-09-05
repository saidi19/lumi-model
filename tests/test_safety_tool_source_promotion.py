from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_source_promotion import (
    EXPECTED_SOURCE_ENTRY,
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_promotion_policy_and_predecessor,
    load_source_promotion_plan,
    resolve_promotion_binding,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)
from lumi.posttraining_sources import build_sft_source_readiness_report

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-safety-tool-source-promotion-v0.1.json"
SUCCESSOR_PATH = ROOT / "configs" / "lumi-4b-sft-source-registry-v0.2.json"
RECEIPT_PATH = ROOT / "evidence" / "lumi-4b-safety-tool-source-promotion-v0.1.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prerequisites(plan: dict) -> tuple[dict, dict, dict]:
    bindings = plan["bindings"]
    candidate = _load_json(resolve_promotion_binding(ROOT, bindings["candidate_evidence"]))
    token_audit = _load_json(resolve_promotion_binding(ROOT, bindings["token_audit"]))
    owner_receipt = _load_json(
        resolve_promotion_binding(ROOT, bindings["owner_review_receipt"])
    )
    return candidate, token_audit, owner_receipt


def test_candidate_promotion_is_exact_append_only_successor() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    candidate, token_audit, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        candidate_evidence=candidate,
        owner_review_receipt=owner_receipt,
        plan=plan,
        token_audit=token_audit,
    )
    policy, predecessor, predecessor_value = load_promotion_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    assert predecessor.registry_sha256 == (
        "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9"
    )

    expected_successor = build_successor_registry(predecessor_value, plan=plan)
    successor_value = _load_json(SUCCESSOR_PATH)
    assert successor_value == expected_successor
    assert successor_value["sources"][:-1] == predecessor_value["sources"]
    assert successor_value["sources"][-1] == EXPECTED_SOURCE_ENTRY

    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=file_sha256(SUCCESSOR_PATH),
    )
    assert successor.registry_sha256 == (
        "f686668fbb05f0d7930b5047560543ce113fcfb97c423fbe840f65d8a1058cdd"
    )
    assert len(successor.sources) == 10
    source = successor.by_id["lumi_safety_tool_sft_v0_1"]
    assert source.status == "candidate_eligible"
    assert source.production_eligible is False
    assert successor.record_reasons(
        dataset_id=source.dataset_id,
        language="bn",
        license_id=source.license_id,
        revision=source.revision,
        source_id=source.source_id,
        split="train",
        task_category="safety_helpfulness",
    ) == ()


def test_candidate_promotion_closes_canary_gap_not_production_gap() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    policy, predecessor, predecessor_value = load_promotion_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    successor_value = _load_json(SUCCESSOR_PATH)
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=file_sha256(SUCCESSOR_PATH),
    )
    required_tasks = frozenset(plan["required_tasks"])
    before = build_sft_source_readiness_report(
        predecessor,
        required_tasks=required_tasks,
    )
    after = build_sft_source_readiness_report(successor, required_tasks=required_tasks)
    assert before["missing_canary_tasks"] == ["safety_helpfulness"]
    assert after["missing_canary_tasks"] == []
    assert set(after["missing_production_tasks"]) == required_tasks
    assert after["production_seal_ready"] is False


def test_generated_promotion_receipt_is_reproducible_and_fail_closed() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    candidate, token_audit, owner_receipt = _prerequisites(plan)
    verify_source_promotion_prerequisites(
        candidate_evidence=candidate,
        owner_review_receipt=owner_receipt,
        plan=plan,
        token_audit=token_audit,
    )
    policy, predecessor, predecessor_value = load_promotion_policy_and_predecessor(
        plan=plan,
        repository_root=ROOT,
    )
    successor_value = _load_json(SUCCESSOR_PATH)
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
        token_audit=token_audit,
    )
    observed = _load_json(RECEIPT_PATH)
    assert observed == expected
    verify_source_promotion_receipt(observed)
    assert observed["candidate_source_registry_promotions"] == 1
    assert observed["production_dataset_promotions"] == 0
    assert observed["active_production_data_contract_updated"] is False

    changed = copy.deepcopy(observed)
    changed["production_eligible"] = True
    with pytest.raises(ValueError, match="receipt_sha256 changed"):
        verify_source_promotion_receipt(changed)


def test_rejected_owner_gate_cannot_be_promoted() -> None:
    plan = load_source_promotion_plan(PLAN_PATH, repository_root=ROOT)
    candidate, token_audit, owner_receipt = _prerequisites(plan)
    changed = copy.deepcopy(owner_receipt)
    changed["rejected_records"] = 1
    changed["all_sampled_rows_accepted"] = False
    with pytest.raises(ValueError, match="receipt_sha256 changed"):
        verify_source_promotion_prerequisites(
            candidate_evidence=candidate,
            owner_review_receipt=changed,
            plan=plan,
            token_audit=token_audit,
        )
