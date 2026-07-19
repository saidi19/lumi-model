#!/usr/bin/env python3
"""Independently reproduce a sealed AWS S3 provider plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_s3_provider import verify_aws_s3_provider_report

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "aws-s3-provider-v0.1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_aws_s3_provider_report(
        args.contract,
        args.report,
        args.manifest,
        repository_root=ROOT,
    )
    print("PASS: AWS S3 provider report hash and deterministic reproduction agree")
    print(json.dumps({"decision": report["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
