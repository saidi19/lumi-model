#!/usr/bin/env python3
"""Validate an exact isolated Lumi-1B milestone training state."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.scaling_run import load_isolated_scaling_run_plan, validate_training_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_isolated_scaling_run_plan(args.plan)
    state = validate_training_state(
        plan=plan,
        training_output=args.training_output,
        expected_step=args.expected_step,
    )
    print(
        f"PASS: step={state['step']} tokens={state['tokens_seen']} "
        f"checkpoint={state['latest_checkpoint']}"
    )


if __name__ == "__main__":
    main()
