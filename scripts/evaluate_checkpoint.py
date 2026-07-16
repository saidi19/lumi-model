#!/usr/bin/env python3
"""Evaluate a Lumi checkpoint on a sealed held-out token artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    evaluation = plan["evaluation"]
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        tokenizer_path=args.tokenizer,
        output=args.output,
        prompts=evaluation["prompts"],
        generation_max_new_tokens=evaluation["generation_max_new_tokens"],
        batch_size=evaluation["batch_size"],
        checkpoint_batches=evaluation["checkpoint_batches"],
        device=args.device,
        resume=args.resume,
    )
    print(
        f"PASS: loss={report['loss']['mean']:.6f} "
        f"perplexity={report['loss']['perplexity']:.3f} "
        f"tokens_per_second={report['runtime']['tokens_per_second']:.1f} "
        f"device={report['runtime']['device']}"
    )


if __name__ == "__main__":
    main()
