#!/usr/bin/env python3
"""Run Lumi's pinned local initialization and optimizer A/B gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.init_optimizer_gate import run_init_optimizer_ab_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_init_optimizer_ab_gate(
        specification_path=args.specification,
        training_manifest_path=args.training_manifest,
        heldout_manifest_path=args.heldout_manifest,
        output=args.output,
        canary_only=args.canary_only,
    )
    if args.canary_only:
        canary = result["resume_canary"]
        print(
            f"PASS: resume_canary step={canary['step']} tokens={canary['tokens_seen']} "
            f"checkpoint={canary['checkpoint']}"
        )
        return
    decision = result["decision"]
    print(
        f"PASS: winner={decision['winner']} init={decision['initialization_strategy']} "
        f"optimizer={decision['optimizer_profile']}"
    )


if __name__ == "__main__":
    main()
