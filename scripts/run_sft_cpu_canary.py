#!/usr/bin/env python3
"""Run or fresh-resume the CPU-only Lumi assistant-loss SFT correctness canary."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_sft_runtime import run_sft_cpu_canary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-shards", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=2)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = run_sft_cpu_canary(
        encoded_shards=args.encoded_shards,
        source_checkpoint=args.source_checkpoint,
        source_checkpoint_manifest_sha256=args.source_checkpoint_manifest_sha256,
        output=args.output,
        target_steps=args.target_steps,
        stop_after_step=args.stop_after_step,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        maximum_gradient_norm=args.maximum_gradient_norm,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        resume=args.resume,
    )
    print(
        f"PASS: SFT CPU canary status={state['status']} step={state['step']} "
        f"loss={state['last_loss']:.6f} checkpoint={state['latest_checkpoint']}"
    )


if __name__ == "__main__":
    main()
