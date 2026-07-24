from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.segment2_throughput_gate import (
    BASELINE_LABEL,
    evaluate_segment2_throughput_gate,
    validate_gate_contract,
    validate_gate_decision,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/4b-segment2-throughput-gate-v0.1.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _evidence(
    rates: dict[str, float] | None = None,
    *,
    heldout_losses: dict[str, float] | None = None,
    training_losses: dict[str, float] | None = None,
) -> dict:
    contract = _contract()
    rates = rates or {
        BASELINE_LABEL: 70_000,
        "batch2_gc": 82_000,
        "batch1_no_gc": 88_000,
        "batch2_no_gc": 101_000,
        "batch2_no_gc_compile": 112_000,
    }
    heldout_losses = heldout_losses or {label: 2.30 for label in rates}
    training_losses = training_losses or {label: 2.25 for label in rates}
    source = contract["source"]
    results = []
    for profile in contract["profiles"]:
        label = profile["label"]
        results.append(
            {
                "checkpoint_restore_pass": True,
                "document_isolation_pass": True,
                "label": label,
                "measured_steps": contract["benchmark"]["measured_steps_per_candidate"],
                "metrics": {
                    "heldout_loss": heldout_losses[label],
                    "median_tokens_per_second": rates[label],
                    "p10_tokens_per_second": rates[label] * 0.98,
                    "peak_memory_bytes": 60_000_000_000,
                    "training_loss": training_losses[label],
                },
                "non_finite_count": 0,
                "profile": profile,
                "resume_canary_pass": True,
                "source_checkpoint": {
                    "checkpoint_manifest_sha256": "c" * 64,
                    "run_fingerprint": source["run_fingerprint"],
                    "source_commit": source["source_commit"],
                    "step": source["segment_1_stop_step"],
                    "tokens": source["segment_1_stop_tokens"],
                },
                "status": "pass",
                "warmup_steps": contract["benchmark"]["warmup_steps_per_candidate"],
            }
        )
    return {
        "observed_cost_usd": 45.0,
        "observed_runtime_minutes": 84.0,
        "results": results,
        "schema_version": 1,
        "status": "complete",
    }


def test_contract_is_planning_only_and_preserves_global_batch() -> None:
    contract = validate_gate_contract(_contract())

    assert contract["approval"] == {
        "paid_gpu": False,
        "production_training": False,
    }
    assert all(
        contract["hardware"]["world_size"]
        * profile["per_rank_batch_size"]
        * profile["gradient_accumulation_steps"]
        == contract["benchmark"]["global_sequences_per_step"]
        for profile in contract["profiles"]
    )

    changed = copy.deepcopy(contract)
    changed["approval"]["paid_gpu"] = True
    with pytest.raises(ValueError, match="cannot authorize paid work"):
        validate_gate_contract(changed)


def test_gate_selects_fastest_materially_safe_profile() -> None:
    decision = evaluate_segment2_throughput_gate(_contract(), _evidence())

    assert decision["winner"] == "batch2_no_gc_compile"
    assert decision["reason"] == "material_safe_throughput_gain"
    assert (
        decision["selected_projected_remaining_cost_usd"]
        < decision["baseline_projected_remaining_cost_usd"]
    )
    assert validate_gate_decision(decision) == decision


def test_gate_keeps_baseline_when_candidates_are_only_marginally_faster() -> None:
    rates = {
        BASELINE_LABEL: 70_000,
        "batch2_gc": 75_000,
        "batch1_no_gc": 77_000,
        "batch2_no_gc": 79_000,
        "batch2_no_gc_compile": 80_000,
    }

    decision = evaluate_segment2_throughput_gate(
        _contract(),
        _evidence(rates),
    )

    assert decision["winner"] == BASELINE_LABEL
    assert decision["reason"] == "no_candidate_cleared_fail_closed_thresholds"


def test_gate_rejects_fast_profile_with_loss_regression() -> None:
    heldout = {
        BASELINE_LABEL: 2.30,
        "batch2_gc": 2.30,
        "batch1_no_gc": 2.30,
        "batch2_no_gc": 2.30,
        "batch2_no_gc_compile": 2.40,
    }

    decision = evaluate_segment2_throughput_gate(
        _contract(),
        _evidence(heldout_losses=heldout),
    )

    assert decision["winner"] == "batch2_no_gc"
    assert (
        "heldout_loss_regression"
        in decision["comparisons_to_baseline"]["batch2_no_gc_compile"]["reasons"]
    )


def test_gate_allows_preserved_oom_for_nonbaseline_candidate() -> None:
    evidence = _evidence()
    failed = evidence["results"][-1]
    failed.clear()
    failed.update(
        {
            "error": "CUDA out of memory",
            "label": "batch2_no_gc_compile",
            "profile": _contract()["profiles"][-1],
            "source_checkpoint": {
                "checkpoint_manifest_sha256": "c" * 64,
                "run_fingerprint": _contract()["source"]["run_fingerprint"],
                "source_commit": _contract()["source"]["source_commit"],
                "step": 31_500,
                "tokens": 33_030_144_000,
            },
            "status": "oom",
        }
    )

    decision = evaluate_segment2_throughput_gate(_contract(), evidence)

    assert decision["winner"] == "batch2_no_gc"
    assert decision["comparisons_to_baseline"]["batch2_no_gc_compile"] == {
        "eligible": False,
        "reason": "oom",
    }


def test_gate_rejects_mixed_checkpoint_evidence_and_cost_overrun() -> None:
    evidence = _evidence()
    evidence["results"][1]["source_checkpoint"]["checkpoint_manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="one checkpoint"):
        evaluate_segment2_throughput_gate(_contract(), evidence)

    evidence = _evidence()
    evidence["observed_cost_usd"] = 64.0
    with pytest.raises(ValueError, match="cost cap"):
        evaluate_segment2_throughput_gate(_contract(), evidence)


def test_gate_decision_detects_tampering() -> None:
    decision = evaluate_segment2_throughput_gate(_contract(), _evidence())
    decision["winner"] = BASELINE_LABEL

    with pytest.raises(ValueError, match="hash changed"):
        validate_gate_decision(decision)
