from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_data import SFTPolicy, evaluate_sft_record
from lumi.posttraining_private_research_data_contract_v2 import (
    AI_SOURCE_CONTRACTS,
    EXCLUDED_SOURCE_IDS,
    INCLUDED_SOURCE_IDS,
    _synthetic_ai_record,
    build_contract,
    build_verification,
    sha256_value,
    validate_contract,
    validate_report,
    validate_verification,
)

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "configs/lumi-4b-sft-data-contract-v0.1.json"


def _predecessor() -> dict:
    return json.loads(PREDECESSOR.read_text(encoding="utf-8"))


def _contract() -> dict:
    return build_contract(_predecessor())


def test_v02_loads_exact_scoped_registry_partition() -> None:
    policy = SFTPolicy.from_mapping(_contract())
    learning = policy.learning_candidate
    assert learning is not None
    assert policy.source_registry_path == "configs/lumi-4b-sft-source-registry-v0.24.json"
    assert policy.source_registry_sha256 == (
        "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb"
    )
    assert learning["included_source_ids"] == frozenset(INCLUDED_SOURCE_IDS)
    assert learning["excluded_source_ids"] == frozenset(EXCLUDED_SOURCE_IDS)
    assert not learning["included_source_ids"] & learning["excluded_source_ids"]
    assert len(learning["included_source_ids"] | learning["excluded_source_ids"]) == 27


def test_v01_loader_remains_backward_compatible() -> None:
    policy = SFTPolicy.from_mapping(_predecessor())
    assert policy.learning_candidate is None
    assert policy.source_registry_path == "configs/lumi-4b-sft-source-registry-v0.1.json"
    assert policy.source_registry_sha256 == (
        "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9"
    )


def test_v02_preserves_every_v01_training_policy_field() -> None:
    before = SFTPolicy.from_mapping(_predecessor())
    after = SFTPolicy.from_mapping(_contract())
    for field in (
        "allowed_languages",
        "allowed_licenses",
        "allowed_splits",
        "benchmark_decontamination",
        "chat_template",
        "context_window",
        "maximum_message_characters",
        "maximum_messages",
        "minimum_quality_score",
        "packing_policy",
        "source_registry_required",
        "supervised_roles",
        "task_mix",
    ):
        assert getattr(after, field) == getattr(before, field)


@pytest.mark.parametrize("mutation", ["remove", "add", "overlap", "swap", "duplicate"])
def test_v02_rejects_source_partition_drift(mutation: str) -> None:
    contract = _contract()
    included = contract["learning_candidate"]["included_source_ids"]
    excluded = contract["learning_candidate"]["excluded_source_ids"]
    if mutation == "remove":
        included.pop()
    elif mutation == "add":
        included.append("unknown_source")
    elif mutation == "overlap":
        excluded.append(included[0])
    elif mutation == "swap":
        included[0], included[1] = included[1], included[0]
    else:
        included.append(included[-1])
    with pytest.raises(ValueError):
        SFTPolicy.from_mapping(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope", "production"),
        ("seal_allowed", False),
        ("production_dataset_seal_allowed", True),
        ("production_sft_allowed", True),
        ("external_distribution_allowed", True),
    ],
)
def test_v02_rejects_scope_drift(field: str, value: object) -> None:
    contract = _contract()
    contract["learning_candidate"][field] = value
    with pytest.raises(ValueError):
        SFTPolicy.from_mapping(contract)


@pytest.mark.parametrize(
    ("path", "sha256"),
    [
        ("/tmp/lumi-4b-sft-source-registry-v0.24.json", "a" * 64),
        ("../configs/lumi-4b-sft-source-registry-v0.24.json", "a" * 64),
        ("configs/lumi-4b-sft-source-registry-v0.23.json", "a" * 64),
        ("configs/lumi-4b-sft-source-registry-v0.25.json", "a" * 64),
        ("other/lumi-4b-sft-source-registry-v0.24.json", "a" * 64),
        ("configs/lumi-4b-sft-source-registry-v0.24.json", "a" * 64),
    ],
)
def test_v02_rejects_unbound_registry_path_or_hash(path: str, sha256: str) -> None:
    contract = _contract()
    contract["source_registry"].update(path=path, sha256=sha256)
    with pytest.raises(ValueError):
        SFTPolicy.from_mapping(contract)


def test_v02_rejects_extra_learning_candidate_permission() -> None:
    contract = _contract()
    contract["learning_candidate"]["unknown_permission"] = True
    with pytest.raises(ValueError, match="keys"):
        SFTPolicy.from_mapping(contract)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("learning_candidate", "seal_allowed"), 1),
        (("learning_candidate", "production_sft_allowed"), 0),
        (("learning_candidate", "registry_source_count"), 27.0),
        (("learning_candidate", "machine_only_gap_rows", "allowed"), 1),
        (
            (
                "learning_candidate",
                "machine_only_gap_rows",
                "human_authorship_claim_allowed",
            ),
            0,
        ),
        (
            (
                "learning_candidate",
                "ai_assisted_source_contracts",
                "lumi_ai_assisted_gap_closure_canary_v0_1",
                "model_assistance_used",
            ),
            1,
        ),
    ],
)
def test_v02_rejects_json_bool_int_type_confusion(path: tuple[str, ...], value: object) -> None:
    contract = _contract()
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        SFTPolicy.from_mapping(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registry", "../data/benchmark-signatures-v0.1.jsonl.gz"),
        ("registry_sha256", "a" * 64),
        ("anchor_ngram_tokens", 9),
        ("minimum_anchor_matches", 1),
    ],
)
def test_v02_rejects_benchmark_contract_drift(field: str, value: object) -> None:
    contract = _contract()
    contract["benchmark_decontamination"][field] = value
    with pytest.raises(ValueError):
        SFTPolicy.from_mapping(contract)


def test_record_validator_enforces_allowlist_and_ai_provenance() -> None:
    contract = _contract()
    policy = SFTPolicy.from_mapping(contract)
    source_id = "lumi_ai_assisted_gap_closure_canary_v0_1"
    accepted = evaluate_sft_record(_synthetic_ai_record(source_id, contract), policy=policy)
    assert accepted.accepted is True

    excluded = _synthetic_ai_record(source_id, contract)
    excluded["source"]["source_id"] = EXCLUDED_SOURCE_IDS[0]
    decision = evaluate_sft_record(excluded, policy=policy)
    assert decision.accepted is False
    assert "learning_candidate_source_not_included" in decision.reasons

    false_claim = _synthetic_ai_record(source_id, contract)
    false_claim["provenance"]["human_authorship_claimed"] = True
    decision = evaluate_sft_record(false_claim, policy=policy)
    assert decision.accepted is False
    assert "learning_candidate_ai_human_authorship_claim" in decision.reasons

    reviewed = _synthetic_ai_record(source_id, contract)
    reviewed["owner_review"]["human_review_completed"] = True
    decision = evaluate_sft_record(reviewed, policy=policy)
    assert decision.accepted is False
    assert "learning_candidate_ai_owner_review" in decision.reasons

    contradictory_owner = _synthetic_ai_record(source_id, contract)
    contradictory_owner["owner_review"]["human_review_claimed"] = True
    decision = evaluate_sft_record(contradictory_owner, policy=policy)
    assert decision.accepted is False
    assert "learning_candidate_ai_owner_review" in decision.reasons

    numeric_bool = _synthetic_ai_record(source_id, contract)
    numeric_bool["provenance"]["model_assistance_used"] = 1
    decision = evaluate_sft_record(numeric_bool, policy=policy)
    assert decision.accepted is False
    assert "learning_candidate_ai_provenance:model_assistance_used" in decision.reasons


def test_record_validator_fails_closed_for_unhashable_source_id() -> None:
    contract = _contract()
    policy = SFTPolicy.from_mapping(contract)
    record = _synthetic_ai_record("lumi_ai_assisted_gap_closure_canary_v0_1", contract)
    record["source"]["source_id"] = ["not", "hashable"]
    decision = evaluate_sft_record(record, policy=policy)
    assert decision.accepted is False
    assert "source_id" in decision.reasons
    assert "learning_candidate_source_not_included" in decision.reasons


def test_contract_validator_rejects_predecessor_policy_change() -> None:
    predecessor = _predecessor()
    contract = build_contract(predecessor)
    contract["filters"]["minimum_quality_score"] = 0.7
    with pytest.raises(ValueError):
        validate_contract(contract, predecessor=predecessor)


def test_report_and_verification_reject_production_flip() -> None:
    report = {
        "schema_version": "lumi.sft-private-research-data-contract-report.v0.2",
        "status": "PASS_private_research_nonproduction_data_contract_v0_2_active",
        "plan_file_sha256": "a" * 64,
        "production_eligible": True,
        "raw_training_text_retained": False,
        "data_contract": {"file_sha256": "c" * 64},
        "policy": {
            "included_source_ids": list(INCLUDED_SOURCE_IDS),
            "excluded_source_ids": list(EXCLUDED_SOURCE_IDS),
            "production_dataset_seal_allowed": False,
            "production_sft_allowed": False,
            "external_distribution_allowed": False,
        },
        "production": {
            "dataset_sealed": False,
            "production_eligible": False,
            "production_registry_created": False,
            "sft_training_started": False,
        },
        "enforcement": {
            "source_allowlist_enforced_by_record_validator": True,
            "two_machine_gap_source_provenance_contracts_enforced": True,
            "synthetic_ai_contract_checks": {key: True for key in AI_SOURCE_CONTRACTS},
        },
        "registry": {"production_eligible_source_count": 0},
        "resource_accounting": {
            "candidate_row_reads": 0,
            "data_contract_writes": 1,
            "factual_passes": 1,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "next_gate": {
            "action": "nonproduction_learning_candidate_seal_preflight_for_registry_v0_24_mix",
            "authorization_required": True,
        },
    }
    report["report_sha256"] = sha256_value(report)
    with pytest.raises(ValueError):
        validate_report(report)

    report["production_eligible"] = False
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    verification = build_verification(report=report, report_file_sha256="b" * 64)
    changed = copy.deepcopy(verification)
    changed["checks"]["production_flags_remain_false"] = False
    changed.pop("verification_sha256")
    changed["verification_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError):
        validate_verification(changed)
