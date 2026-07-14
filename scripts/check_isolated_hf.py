#!/usr/bin/env python3
"""Export a tiny self-contained Lumi checkpoint and verify it in isolated Python."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from lumi import LumiConfig, LumiForCausalLM


def main() -> None:
    torch.manual_seed(17)
    config = LumiConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    model = LumiForCausalLM(config).eval()
    input_ids = torch.tensor([[config.bos_token_id, 12, 25, config.doc_token_id]])
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    loader = Path(__file__).with_name("isolated_hf_loader.py").resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        payload = root / "expected.pt"
        model.save_pretrained(checkpoint)
        required_files = {
            "config.json",
            "configuration_lumi.py",
            "modeling_lumi.py",
            "model.safetensors",
        }
        missing_files = required_files.difference(path.name for path in checkpoint.iterdir())
        if missing_files:
            raise RuntimeError(f"exported checkpoint is missing files: {sorted(missing_files)}")
        with (checkpoint / "config.json").open(encoding="utf-8") as handle:
            exported_config = json.load(handle)
        required_auto_classes = {"AutoConfig", "AutoModel", "AutoModelForCausalLM"}
        actual_auto_classes = set(exported_config.get("auto_map", {}))
        if not required_auto_classes.issubset(actual_auto_classes):
            raise RuntimeError(
                f"exported checkpoint has incomplete auto_map: {sorted(actual_auto_classes)}"
            )
        torch.save(
            {"input_ids": input_ids, "expected_logits": expected_logits},
            payload,
        )
        child_environment = os.environ.copy()
        child_environment["HF_HOME"] = str(root / "huggingface-cache")
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(loader),
                "--checkpoint",
                str(checkpoint),
                "--payload",
                str(payload),
            ],
            cwd=root,
            env=child_environment,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated checkpoint verification failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
