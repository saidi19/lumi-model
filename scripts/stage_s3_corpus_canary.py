#!/usr/bin/env python3
"""Stage Lumi's exact eight-source S3 canary inside one bounded CodeBuild."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from lumi.s3_staged_corpus import (
    stage_s3_staging_plan,
    validate_s3_staging_plan,
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-fingerprint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _write_result(path: Path, result: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite staging result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if SHA256_PATTERN.fullmatch(args.expected_plan_fingerprint) is None:
        raise ValueError("expected S3 staging plan fingerprint is invalid")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    verified = validate_s3_staging_plan(plan)
    if verified["plan_fingerprint"] != args.expected_plan_fingerprint:
        raise ValueError("authorized S3 staging plan fingerprint changed")
    if os.environ.get("AWS_DEFAULT_REGION") != "us-east-1":
        raise ValueError("S3 staging CodeBuild must run in us-east-1")

    import boto3

    scratch_cap = int(verified["source_bytes"]) + 128 * 1024 * 1024
    result = stage_s3_staging_plan(
        plan,
        args.scratch,
        scratch_cap=scratch_cap,
        s3_client=boto3.client("s3", region_name="us-east-1"),
        max_workers=args.workers,
    )
    _write_result(args.output, result)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
