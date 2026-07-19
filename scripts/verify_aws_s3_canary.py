#!/usr/bin/env python3
"""Verify sealed bounded AWS S3 canary evidence after cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_s3_canary import verify_canary_evidence

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "aws-s3-provider-v0.1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_canary_evidence(
        args.contract,
        args.output,
        repository_root=ROOT,
    )
    print("PASS: bounded AWS S3 canary evidence and cleanup invariants agree")
    print(json.dumps({"decision": report["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
