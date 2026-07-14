#!/usr/bin/env python3
"""Run a cheap synthetic optimization check; this is not a quality pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lumi import LumiConfig, LumiForCausalLM


def make_batch(batch_size: int, sequence_length: int, vocab_size: int) -> torch.Tensor:
    starts = torch.randint(3, min(vocab_size, 128), (batch_size, 1))
    offsets = torch.arange(sequence_length).unsqueeze(0)
    return 3 + ((starts + offsets - 3) % (vocab_size - 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lumi-smoke.json"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(7)
    with args.config.open(encoding="utf-8") as handle:
        config = LumiConfig(**json.load(handle))
    model = LumiForCausalLM(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    losses: list[float] = []
    for step in range(args.steps):
        input_ids = make_batch(args.batch_size, args.sequence_length, config.vocab_size)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step={step + 1:03d} loss={losses[-1]:.6f}")

    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"smoke loss did not improve: first={losses[0]:.6f}, last={losses[-1]:.6f}"
        )
    print(f"PASS: loss {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
