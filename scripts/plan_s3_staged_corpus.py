#!/usr/bin/env python3
"""Build and verify Lumi's local-only S3-staged corpus transport plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.s3_staged_corpus import (
    build_s3_staging_plan,
    validate_s3_staging_plan,
    write_s3_staging_plan,
)

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "s3-staged-corpus-v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_s3_staging_plan(args.contract, repository_root=ROOT)
    write_s3_staging_plan(args.output, plan)
    print(json.dumps(validate_s3_staging_plan(plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
