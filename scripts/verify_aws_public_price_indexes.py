#!/usr/bin/env python3
"""Verify the exact public AWS Price List files used by the S3 overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_s3_provider import verify_public_price_indexes

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "configs" / "aws-s3-us-east-1-pricing-v0.1.json",
    )
    parser.add_argument("--s3-index", type=Path, required=True)
    parser.add_argument("--data-transfer-index", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_public_price_indexes(
        args.snapshot,
        args.s3_index,
        args.data_transfer_index,
    )
    print("PASS: public AWS Price List files match the curated pricing snapshot")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
