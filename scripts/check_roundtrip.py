#!/usr/bin/env python3
"""Verify save/load and short untrained generation for the smoke model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from lumi import LumiConfig, LumiForCausalLM


def main() -> None:
    with Path("configs/lumi-smoke.json").open(encoding="utf-8") as handle:
        config = LumiConfig(**json.load(handle))
    model = LumiForCausalLM(config).eval()
    input_ids = torch.tensor([[config.bos_token_id, 17, 23, 9]])

    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = AutoModelForCausalLM.from_pretrained(directory).eval()
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits
            generated = loaded.generate(input_ids=input_ids, max_new_tokens=2, use_cache=False)

    torch.testing.assert_close(actual, expected)
    if generated.shape != (1, input_ids.shape[1] + 2):
        raise RuntimeError(f"unexpected generated shape: {tuple(generated.shape)}")
    print("PASS: save/load logits and generation round trip")


if __name__ == "__main__":
    main()
