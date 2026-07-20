from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from lumi.mlx_conversion import MLX_MODEL_FILENAME, build_mlx_config


def _source_config() -> dict:
    root = Path(__file__).parents[1]
    return json.loads((root / "configs" / "lumi-150m.json").read_text(encoding="utf-8"))


def test_mlx_config_preserves_lumi_architecture_identity() -> None:
    source = _source_config()
    source["quantization"] = {"bits": 8}

    converted = build_mlx_config(source, dtype="float32")

    assert converted["model_type"] == "lumi"
    assert converted["model_file"] == MLX_MODEL_FILENAME
    assert converted["rope_traditional"] is False
    assert converted["head_dim"] == 64
    assert converted["num_attention_heads"] == 16
    assert converted["num_key_value_heads"] == 4
    assert converted["dtype"] == "float32"
    assert "quantization" not in converted
    assert source["quantization"] == {"bits": 8}


def test_mlx_config_preserves_final_4b_wide_query_projection() -> None:
    root = Path(__file__).parents[1]
    source = json.loads((root / "configs" / "lumi-4b-v0.2.json").read_text(encoding="utf-8"))

    converted = build_mlx_config(source, dtype="float32")

    assert converted["hidden_size"] == 3072
    assert converted["num_attention_heads"] == 64
    assert converted["num_key_value_heads"] == 8
    assert converted["head_dim"] == 64
    assert converted["num_attention_heads"] * converted["head_dim"] == 4096
    assert converted["num_key_value_heads"] * converted["head_dim"] == 512


def test_mlx_config_rejects_architecture_drift() -> None:
    source = _source_config()
    source["model_type"] = "llama"
    with pytest.raises(ValueError, match="expected a Lumi checkpoint"):
        build_mlx_config(source, dtype="float32")

    source = _source_config()
    source["num_key_value_heads"] = 3
    with pytest.raises(ValueError, match="must be divisible"):
        build_mlx_config(source, dtype="float32")


def test_mlx_config_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        build_mlx_config(_source_config(), dtype="float64")


def test_optional_mlx_module_is_not_imported_by_the_base_package() -> None:
    root = Path(__file__).parents[1]
    package_init = (root / "src" / "lumi" / "__init__.py").read_text(encoding="utf-8")
    assert "mlx_model" not in package_init
    assert "mlx_conversion" not in package_init

    model_source = (root / "src" / "lumi" / "mlx_model.py").read_text(encoding="utf-8")
    parsed = ast.parse(model_source)
    classes = {node.name for node in parsed.body if isinstance(node, ast.ClassDef)}
    assert {"Model", "ModelArgs", "LumiModel", "LumiDecoderLayer"} <= classes
    assert "import torch" not in model_source
