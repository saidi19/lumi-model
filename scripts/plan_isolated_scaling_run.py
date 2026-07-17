#!/usr/bin/env python3
"""Validate and project the fresh Lumi-1B isolated 1.4B-token run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.scaling_run import plan_isolated_scaling_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/isolated-scaling-run-1b-v0.1.json"),
    )
    parser.add_argument("--hourly-rate-usd", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = plan_isolated_scaling_run(
        args.plan, hourly_rate_usd=args.hourly_rate_usd
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing != report:
                raise ValueError(f"existing projection changed: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(f".{args.output.name}.tmp")
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(args.output)
    print(rendered, end="")
    final = report["milestones"][-1]["projection"]
    print(
        "PASS: planning-only "
        f"compute_hours={final['compute_hours']:.3f} "
        f"lifecycle_cost_usd={final['lifecycle_cost_with_allowance_usd']:.2f} "
        "paid_gpu_launch=false"
    )


if __name__ == "__main__":
    main()
