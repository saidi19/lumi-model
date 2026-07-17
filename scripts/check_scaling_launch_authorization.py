#!/usr/bin/env python3
"""Fail unless a Lumi paid-run authorization matches the exact plan and commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.scaling_run import validate_launch_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = validate_launch_authorization(
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_commit=args.source_commit,
    )
    rendered = json.dumps(authorization, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(
        "PASS: owner-approved "
        f"through_step={authorization['approved_through_step']} "
        f"provider={authorization['provider']} "
        f"instance={authorization['instance_type']}"
    )


if __name__ == "__main__":
    main()
