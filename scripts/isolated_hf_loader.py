#!/usr/bin/env python3
"""Load and verify a Lumi checkpoint without importing the local Lumi package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    return parser.parse_args()


def _assert_dynamic_class(model) -> None:
    module_name = model.__class__.__module__
    if not module_name.startswith("transformers_modules."):
        raise RuntimeError(f"checkpoint used a non-isolated model class: {module_name}")


def main() -> None:
    args = parse_args()
    if "lumi" in sys.modules:
        raise RuntimeError("the local lumi package was imported before checkpoint loading")

    payload = torch.load(args.payload, map_location="cpu", weights_only=True)
    input_ids = payload["input_ids"]
    expected_logits = payload["expected_logits"]

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    _assert_dynamic_class(model)
    with torch.no_grad():
        actual_logits = model(input_ids=input_ids).logits
        generated = model.generate(input_ids=input_ids, max_new_tokens=2, use_cache=False)
    torch.testing.assert_close(actual_logits, expected_logits)
    if generated.shape != (input_ids.shape[0], input_ids.shape[1] + 2):
        raise RuntimeError(f"unexpected generated shape: {tuple(generated.shape)}")

    base_model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    _assert_dynamic_class(base_model)
    with torch.no_grad():
        hidden_states = base_model(input_ids=input_ids).last_hidden_state
    if hidden_states.shape[:2] != input_ids.shape:
        raise RuntimeError(f"unexpected base-model output shape: {tuple(hidden_states.shape)}")

    print("PASS: isolated AutoModel and AutoModelForCausalLM load/generate")


if __name__ == "__main__":
    main()
