#!/usr/bin/env python3
"""Build and verify Lumi's non-authorizing Glue canary package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.glue_canary import build_glue_canary_package, verify_glue_canary_package

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "distributed-production-corpus-v0.1.json",
    )
    parser.add_argument("--anchor-export-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def _source_commit(raw: str | None) -> str:
    if raw is not None:
        return raw
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    build_glue_canary_package(
        contract_path=args.contract,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
        anchor_export_directory=args.anchor_export_directory,
        output_directory=args.output_directory,
    )
    verified = verify_glue_canary_package(args.output_directory)
    print(json.dumps(verified, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
