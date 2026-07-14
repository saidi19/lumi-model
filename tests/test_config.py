from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi import LumiConfig, LumiForCausalLM, parameter_breakdown


def test_analytical_count_matches_instantiated_model(tiny_config: LumiConfig) -> None:
    model = LumiForCausalLM(tiny_config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_breakdown(tiny_config).total == actual


def test_lumi_4b_candidate_parameter_count() -> None:
    path = Path(__file__).parents[1] / "configs" / "lumi-4b.json"
    with path.open(encoding="utf-8") as handle:
        config = LumiConfig(**json.load(handle))
    assert parameter_breakdown(config).total == 4_002_376_704


def test_special_token_ids_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        LumiConfig(pad_token_id=2, bos_token_id=1, eos_token_id=2)


def test_gqa_head_counts_must_divide() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        LumiConfig(num_attention_heads=10, num_key_value_heads=3)
