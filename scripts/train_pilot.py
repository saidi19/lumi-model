#!/usr/bin/env python3
"""Train or resume a correctness-first Lumi real-data pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.pilot_training import train_packed_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = train_packed_pilot(
        manifest_path=args.manifest,
        config_path=args.config,
        output=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        checkpoint_steps=args.checkpoint_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        keep_checkpoints=args.keep_checkpoints,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        schedule_steps=args.schedule_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        device=args.device,
        resume=args.resume,
    )
    print(
        f"PASS: step={state['step']} tokens={state['tokens_seen']} "
        f"loss={state['last_loss']:.6f} checkpoint={state['latest_checkpoint']}"
    )


if __name__ == "__main__":
    main()
