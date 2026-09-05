from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_deepmind_adapter_preflight import (
    DEEPMIND_OBJECTS,
    GENERATOR_MODULE_PATHS,
    NEXT_AUTHORIZATION_STATEMENT,
    SEED_CONTRACT,
    build_preflight_report,
    derive_seed,
    load_adapter_preflight_plan,
    seed_contract_evidence,
    validate_preflight_report,
    validate_source_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_adapter_preflight_plan_loads() -> None:
    plan = load_adapter_preflight_plan(
        ROOT / "configs/lumi-4b-sft-deepmind-adapter-preflight-v0.1.json",
        repository_root=ROOT,
    )
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["generator_execution_authorized"] is False
    assert plan["next_gate"]["exact_authorization_statement"] == (NEXT_AUTHORIZATION_STATEMENT)


def test_source_inventory_and_generator_modules_are_exact() -> None:
    inventory = validate_source_inventory(DEEPMIND_OBJECTS)
    assert inventory == {
        "bytes": 234_672,
        "generator_module_count": 8,
        "inventory_sha256": ("47e472f97f7092cf264edb727bd0821db59869133f56b12cddf3f8a4ef1a6ce9"),
        "objects": 30,
    }


def test_seed_contract_is_deterministic_and_split_disjoint() -> None:
    module = GENERATOR_MODULE_PATHS[0]
    train_namespace = SEED_CONTRACT["split_namespaces"]["train"]
    validation_namespace = SEED_CONTRACT["split_namespaces"]["validation"]
    assert derive_seed(train_namespace, module, 7) == derive_seed(train_namespace, module, 7)
    assert derive_seed(train_namespace, module, 7) != derive_seed(validation_namespace, module, 7)
    evidence = seed_contract_evidence(sample_count=16)
    assert evidence["sampled_seeds_per_split"] == 128
    assert evidence["sampled_train_validation_seed_overlap"] == 0


def test_report_freezes_current_gaps_without_authorizing_generation() -> None:
    plan = {
        "next_gate": {
            "authorization_required": True,
            "exact_authorization_statement": NEXT_AUTHORIZATION_STATEMENT,
            "limits": {
                "exact_source_bytes": 234_672,
                "exact_source_objects": 30,
                "maximum_canary_generated_records": 2_000,
                "maximum_decoded_bytes": 8_388_608,
                "maximum_network_bytes": 1_048_576,
                "maximum_runtime_minutes": 180,
            },
        },
        "observed_at": "2026-08-06",
    }
    report = build_preflight_report(
        plan=plan,
        plan_file_sha256="a" * 64,
        bound_evidence={
            "dmath_candidate_train_supervised_tokens": 48_480,
            "dmath_owner_review_complete": False,
            "math_minimum_gap_before_dmath": 2_199_880,
            "planning_train_supervised_tokens_before_dmath": 1_002_200,
            "projected_10m_floor_gap_after_dmath": 8_949_320,
            "projected_planning_train_supervised_tokens_after_dmath": 1_050_680,
        },
    )
    validate_preflight_report(report)
    assert report["remaining_gap"] == {
        "projected_10m_floor_gap_after_dmath": 8_949_320,
        "projected_reasoning_math_minimum_gap_after_dmath": 2_151_400,
        "token_yield_pilot_required_before_scale_count": True,
    }
    assert report["production"]["generated_records"] == 0
    assert report["production"]["production_eligible"] is False


def test_seed_contract_rejects_unknown_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        derive_seed("test", GENERATOR_MODULE_PATHS[0], 0)
