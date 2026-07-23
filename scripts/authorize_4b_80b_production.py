#!/usr/bin/env python3
"""Write one exact owner authorization for one planned 4B/80B segment."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.production_pretraining import write_owner_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--max-runtime-minutes", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = write_owner_authorization(
        plan_path=args.plan,
        output=args.output,
        segment_index=args.segment_index,
        provider=args.provider,
        instance_type=args.instance_type,
        hourly_rate_usd=args.hourly_rate_usd,
        max_runtime_minutes=args.max_runtime_minutes,
        max_cost_usd=args.max_cost_usd,
    )
    print(
        "PASS: owner_approved "
        f"segment={authorization['segment_index']}/{authorization['segment_count']} "
        f"through_step={authorization['approved_through_step']} "
        f"token_cap={authorization['approved_token_cap']}"
    )


if __name__ == "__main__":
    main()
