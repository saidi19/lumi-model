from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.pre_segment3_throughput_screen import (
    BASELINE_LABEL,
    REQUIRED_LABELS,
    _validate_contract,
    decide,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/4b-pre-segment3-throughput-gate-v0.1.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _result(label: str, *, median: float, p10: float, loss: float) -> dict:
    profile = next(item for item in _contract()["profiles"] if item["label"] == label)
    return {
        "label": label,
        "metrics": {
            "mean_gradient_norm": 0.1,
            "median_tokens_per_second": median,
            "p10_tokens_per_second": p10,
            "peak_memory_bytes": 70_000_000_000,
            "training_loss": loss,
        },
        "profile": profile,
        "status": "pass",
    }


def test_contract_pins_exact_profile_matrix() -> None:
    contract = _validate_contract(_contract())
    assert tuple(item["label"] for item in contract["profiles"]) == REQUIRED_LABELS
    assert contract["profiles"][0]["label"] == BASELINE_LABEL


def test_contract_rejects_global_batch_change() -> None:
    contract = copy.deepcopy(_contract())
    contract["profiles"][1]["gradient_accumulation_steps"] = 31
    with pytest.raises(ValueError, match="global batch"):
        _validate_contract(contract)


def test_decision_selects_only_material_safe_gain() -> None:
    results = [_result(BASELINE_LABEL, median=100_000, p10=98_000, loss=2.0)]
    for label in REQUIRED_LABELS[1:]:
        results.append(_result(label, median=110_000, p10=109_000, loss=2.0))
    winner = REQUIRED_LABELS[-1]
    results[-1] = _result(winner, median=121_000, p10=120_000, loss=2.005)
    decision = decide(contract=_contract(), results=results, runtime_minutes=60)
    assert decision["winner"] == winner
    assert decision["production_training_authorized"] is False
    assert decision["exact_segment2_checkpoint_verification_required"] is True


def test_decision_fails_closed_on_loss_regression() -> None:
    results = [_result(BASELINE_LABEL, median=100_000, p10=98_000, loss=2.0)]
    for label in REQUIRED_LABELS[1:]:
        results.append(_result(label, median=125_000, p10=122_000, loss=2.1))
    decision = decide(contract=_contract(), results=results, runtime_minutes=60)
    assert decision["winner"] == BASELINE_LABEL
