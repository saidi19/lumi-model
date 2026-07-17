#!/usr/bin/env python3
"""Validate and seal a completed Lumi-1B isolated 1.4B-token run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.scaling_run import seal_isolated_scaling_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = seal_isolated_scaling_run(
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_commit=args.source_commit,
        workspace=args.workspace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"PASS: step={result['training']['step']} "
        f"tokens={result['training']['processed_tokens']} "
        f"manifest_sha256={result['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
