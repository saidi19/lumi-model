#!/usr/bin/env python3
"""Fail unless the paid distributed smoke authorization matches its plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.distributed_smoke import validate_owner_authorization


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
        f"fingerprint={authorization['run_fingerprint']} "
        f"through_step={authorization['approved_through_step']} "
        f"max_runtime_minutes={authorization['max_runtime_minutes']} "
        "full_4b_training=false"
    )


if __name__ == "__main__":
    main()
