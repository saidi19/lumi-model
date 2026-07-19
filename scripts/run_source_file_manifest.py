#!/usr/bin/env python3
"""Build the resumable metadata-only Lumi source-file manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.source_file_manifest import run_source_file_manifest

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "source-file-manifest-v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_source_file_manifest(
        args.contract,
        args.output,
        repository_root=ROOT,
        resume=args.resume,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print("PASS: all pinned source-file metadata is complete and sealed")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
