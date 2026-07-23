#!/usr/bin/env python3
"""Validate one Lumi-4B/80B segment authorization against its sealed plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.production_pretraining import validate_owner_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = validate_owner_authorization(
        plan_path=args.plan,
        authorization_path=args.authorization,
    )
    print(
        "PASS: owner_approved "
        f"segment={authorization['segment_index']}/{authorization['segment_count']} "
        f"from_step={authorization['approved_from_step']} "
        f"through_step={authorization['approved_through_step']}"
    )


if __name__ == "__main__":
    main()
