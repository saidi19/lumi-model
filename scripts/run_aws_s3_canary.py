#!/usr/bin/env python3
"""Run or resume the bounded Lumi AWS S3 I/O canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_s3_canary import cleanup_canary, resume_canary, start_canary

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "resume", "cleanup"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="lumi-s3")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "aws-s3-provider-v0.1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "start":
        result = start_canary(
            args.contract,
            args.output,
            profile=args.profile,
            repository_root=ROOT,
            run_id=args.run_id,
        )
    elif args.phase == "resume":
        result = resume_canary(
            args.contract,
            args.output,
            profile=args.profile,
            repository_root=ROOT,
        )
    else:
        result = cleanup_canary(args.output, profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
