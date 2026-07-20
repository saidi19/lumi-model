from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.mlx_topology_gate import (
    FINAL_WIDTH_FIELDS,
    build_final_width_fixture,
    load_gate_inputs,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs" / "mlx-final-topology-gate-v0.1.json"


def test_final_width_fixture_changes_only_depth() -> None:
    _, _, source = load_gate_inputs(PLAN)

    fixture = build_final_width_fixture(source, num_hidden_layers=1)

    assert source["num_hidden_layers"] == 36
    assert fixture["num_hidden_layers"] == 1
    assert {field: fixture[field] for field in FINAL_WIDTH_FIELDS} == {
        field: source[field] for field in FINAL_WIDTH_FIELDS
    }
    assert fixture["hidden_size"] == 3072
    assert fixture["intermediate_size"] == 8384
    assert fixture["num_attention_heads"] == 64
    assert fixture["num_key_value_heads"] == 8
    assert fixture["head_dim"] == 64
    assert fixture["num_attention_heads"] * fixture["head_dim"] == 4096
    assert fixture["num_key_value_heads"] * fixture["head_dim"] == 512


def test_final_width_fixture_rejects_invalid_depth() -> None:
    _, _, source = load_gate_inputs(PLAN)

    with pytest.raises(ValueError, match="positive integer"):
        build_final_width_fixture(source, num_hidden_layers=0)


def test_final_width_gate_plan_keeps_strict_numeric_and_exact_decisions() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    thresholds = plan["parity"]["thresholds"]

    assert thresholds["max_absolute_error"] <= 0.02
    assert thresholds["max_mean_absolute_error"] <= 0.003
    assert thresholds["max_normalized_root_mean_square_error"] <= 0.002
    assert thresholds["min_cosine_similarity"] >= 0.999998
    assert thresholds["max_cached_logprob_error"] <= 0.02
    assert plan["parity"]["generation_max_new_tokens"] >= 4


def test_gate_plan_rejects_source_config_drift(tmp_path: Path) -> None:
    source = json.loads((ROOT / "configs" / "lumi-4b-v0.2.json").read_text(encoding="utf-8"))
    source["hidden_size"] = 2048
    (tmp_path / "lumi-4b-v0.2.json").write_text(json.dumps(source), encoding="utf-8")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    (tmp_path / PLAN.name).write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="config hash mismatch"):
        load_gate_inputs(tmp_path / PLAN.name)
