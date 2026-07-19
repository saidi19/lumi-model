#!/usr/bin/env python3
"""Provision or resume Lumi's approved production corpus S3 namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_s3_production import provision_production_s3

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "production-acquisition-v0.1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = provision_production_s3(
        args.contract,
        args.output,
        repository_root=ROOT,
        tokenizer_path=args.tokenizer,
        training_exclusion_ledger=args.training_exclusion_ledger,
        profile=args.profile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
