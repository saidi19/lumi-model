from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_learning_candidate_seal_preflight_v2 import (
    AI_SOURCE_CONTRACTS,
    EVALUATION_SPLIT_PROHIBITIONS,
    EXCLUDED_SOURCE_IDS,
    EXPECTED_AUTHORIZATION_BOUNDARY,
    EXPECTED_BASE_AND_TOKENIZER,
    EXPECTED_BINDING_CHAIN,
    EXPECTED_CONTRACT_BOUNDARY,
    EXPECTED_DECISION,
    EXPECTED_HYGIENE,
    EXPECTED_INVENTORY,
    EXPECTED_NEXT_GATE,
    EXPECTED_OBSERVED_AT,
    EXPECTED_REGISTRY,
    EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE,
    EXPECTED_SUCCESSOR,
    EXPECTED_SUCCESSOR_LIMITS,
    INCLUDED_SOURCE_IDS,
    REPORT_SCHEMA_VERSION,
    REPORT_STATUS,
    _assert_raw_text_free,
    _reserve_output,
    _resolve_binding,
    build_verification,
    sha256_value,
    validate_report,
    validate_verification,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def report() -> dict:
    value = {
        "authorization_boundary": copy.deepcopy(EXPECTED_AUTHORIZATION_BOUNDARY),
        "base_and_tokenizer": copy.deepcopy(EXPECTED_BASE_AND_TOKENIZER),
        "binding_chain": copy.deepcopy(EXPECTED_BINDING_CHAIN),
        "contract_boundary": copy.deepcopy(EXPECTED_CONTRACT_BOUNDARY),
        "decision": copy.deepcopy(EXPECTED_DECISION),
        "hygiene": copy.deepcopy(EXPECTED_HYGIENE),
        "inventory": copy.deepcopy(EXPECTED_INVENTORY),
        "next_gate": copy.deepcopy(EXPECTED_NEXT_GATE),
        "observed_at": EXPECTED_OBSERVED_AT,
        "plan_file_sha256": "b" * 64,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry": copy.deepcopy(EXPECTED_REGISTRY),
        "resource_accounting": copy.deepcopy(EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "successor": copy.deepcopy(EXPECTED_SUCCESSOR),
    }
    value["report_sha256"] = sha256_value(value)
    validate_report(value)
    return value


def _resign(report: dict) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_value(report)


def test_exact_current_inventory_is_bound_without_candidate_reads(report: dict) -> None:
    assert report["inventory"] == EXPECTED_INVENTORY
    assert report["hygiene"] == EXPECTED_HYGIENE
    assert report["resource_accounting"]["candidate_row_reads"] == 0
    assert report["resource_accounting"]["source_row_reads"] == 0
    assert report["resource_accounting"]["test_object_reads"] == 0


def test_contract_permission_is_not_materialization_authorization(report: dict) -> None:
    assert report["contract_boundary"]["learning_candidate_seal_allowed"] is True
    assert report["authorization_boundary"]["candidate_materialization_authorized"] is False
    assert report["authorization_boundary"]["dataset_seal_authorized"] is False
    assert report["authorization_boundary"]["successor_authorization_required"] is True
    assert report["decision"]["candidate_materialization_performed"] is False
    assert report["production_eligible"] is False


def test_exact_registry_partition_and_evaluation_exclusions(report: dict) -> None:
    registry = report["registry"]
    assert registry["included_source_ids"] == list(INCLUDED_SOURCE_IDS)
    assert registry["excluded_source_ids"] == list(EXCLUDED_SOURCE_IDS)
    assert not set(INCLUDED_SOURCE_IDS) & set(EXCLUDED_SOURCE_IDS)
    assert len(set(INCLUDED_SOURCE_IDS) | set(EXCLUDED_SOURCE_IDS)) == 27
    assert report["contract_boundary"]["evaluation_split_prohibitions"] == (
        EVALUATION_SPLIT_PROHIBITIONS
    )


def test_two_machine_gap_sources_keep_truthful_nonhuman_contract(report: dict) -> None:
    contracts = report["contract_boundary"]["machine_gap_source_contracts"]
    assert contracts == AI_SOURCE_CONTRACTS
    assert set(contracts) == {
        "lumi_ai_assisted_gap_closure_canary_v0_1",
        "lumi_ai_assisted_gap_closure_scaled_v0_2",
    }
    for contract in contracts.values():
        assert contract["model_assistance_used"] is True
        assert contract["human_written_from_scratch"] is False
        assert contract["human_review_completed"] is False
        assert contract["runtime_model_calls"] == 0
        assert contract["runtime_external_model_api_calls"] == 0


def test_successor_limits_are_bounded_and_nontraining(report: dict) -> None:
    limits = report["successor"]["limits"]
    assert limits == EXPECTED_SUCCESSOR_LIMITS
    assert limits["maximum_candidate_row_reads"] == 329_200
    assert limits["maximum_output_audit_row_reads"] == 329_200
    assert limits["maximum_total_row_reads"] == 658_400
    assert limits["maximum_gpu_use"] == 0
    assert limits["maximum_inference_calls"] == 0
    assert limits["maximum_network_bytes"] == 0
    assert limits["maximum_training_actions"] == 0


def test_predecessor_seal_chain_is_not_reused(plan: dict, report: dict) -> None:
    serialized_bindings = str(plan["bindings"])
    assert "v0.17" not in serialized_bindings
    assert "v0.12" not in serialized_bindings
    assert "264981" not in serialized_bindings
    assert "runtime-recovery" not in serialized_bindings
    assert "closure-retirement" not in serialized_bindings
    assert report["inventory"]["records"] == 329_200


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("inventory", "records"), 329_199),
        (("inventory", "train_supervised_tokens"), 11_855_216),
        (("hygiene", "near_duplicate_edges"), 1),
        (("hygiene", "task_floor_gap_total"), 1),
        (("registry", "source_count"), 26),
        (("registry", "production_eligible_source_count"), 1),
        (("base_and_tokenizer", "base_checkpoint_step"), 76_295),
        (("binding_chain", "source_registry_sha256"), "a" * 64),
        (("next_gate", "authorization_required"), False),
        (("authorization_boundary", "candidate_materialization_authorized"), True),
        (("contract_boundary", "production_sft_allowed"), True),
        (("successor", "limits", "maximum_candidate_row_reads"), 329_199),
        (("production_eligible",), True),
    ],
)
def test_report_rejects_boundary_drift(report: dict, path: tuple[str, ...], value: object) -> None:
    changed = copy.deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign(changed)
    with pytest.raises(ValueError):
        validate_report(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("inventory", "records"), 329_200.0),
        (("production_eligible",), 0),
        (("contract_boundary", "learning_candidate_seal_allowed"), 1),
    ],
)
def test_report_rejects_bool_int_type_aliases(
    report: dict, path: tuple[str, ...], value: object
) -> None:
    changed = copy.deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign(changed)
    with pytest.raises(ValueError):
        validate_report(changed)


def test_report_rejects_source_partition_reordering(report: dict) -> None:
    changed = copy.deepcopy(report)
    changed["registry"]["included_source_ids"][0:2] = reversed(
        changed["registry"]["included_source_ids"][0:2]
    )
    _resign(changed)
    with pytest.raises(ValueError):
        validate_report(changed)


def test_verification_requires_every_check_and_zero_action(report: dict) -> None:
    verification = build_verification(report=report, report_file_sha256="a" * 64)
    validate_verification(verification)
    changed = copy.deepcopy(verification)
    changed["checks"]["raw_training_text_absent"] = False
    changed.pop("verification_sha256")
    changed["verification_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError):
        validate_verification(changed)

    changed = copy.deepcopy(verification)
    changed["next_gate"]["authorization_required"] = False
    changed.pop("verification_sha256")
    changed["verification_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError):
        validate_verification(changed)


@pytest.mark.parametrize(
    "raw_key",
    [
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "response",
        "text",
        "user_prompt",
    ],
)
def test_raw_text_keys_fail_closed(raw_key: str) -> None:
    with pytest.raises(ValueError, match="raw candidate text"):
        _assert_raw_text_free({"nested": [{raw_key: "forbidden"}]})


def test_binding_rejects_path_or_hash_drift() -> None:
    with pytest.raises(ValueError):
        _resolve_binding(
            ROOT,
            {
                "path": "../configs/lumi-4b-sft-source-registry-v0.24.json",
                "sha256": "a" * 64,
            },
            "source_registry",
        )


def test_pass_output_reservation_is_durable_and_nonretryable(tmp_path: Path) -> None:
    output = tmp_path / "pass-output.json"
    with _reserve_output(output) as handle:
        assert output.is_file()
        handle.write(b"{}\n")
    with pytest.raises(FileExistsError):
        _reserve_output(output)
