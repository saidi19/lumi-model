#!/usr/bin/env python3
"""Write a bounded owner authorization from a sealed CUDA smoke plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.distributed_smoke import write_owner_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--max-runtime-minutes", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = write_owner_authorization(
        plan_path=args.plan,
        output=args.output,
        provider=args.provider,
        instance_type=args.instance_type,
        hourly_rate_usd=args.hourly_rate_usd,
        max_runtime_minutes=args.max_runtime_minutes,
    )
    print(
        "PASS: owner_approved "
        f"fingerprint={authorization['run_fingerprint']} "
        f"through_step={authorization['approved_through_step']} "
        "full_4b_training=false"
    )


if __name__ == "__main__":
    main()
