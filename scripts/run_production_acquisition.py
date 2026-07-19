#!/usr/bin/env python3
"""Run or resume Lumi's authorized production corpus acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.production_corpus import run_production_acquisition

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--maximum-runtime-seconds", type=int)
    parser.add_argument("--batch-records", type=int, default=32)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "production-acquisition-worker-v0.1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_production_acquisition(
        args.contract,
        repository_root=ROOT,
        tokenizer_path=args.tokenizer,
        training_exclusion_ledger=args.training_exclusion_ledger,
        production_report=args.production_report,
        state_directory=args.state_directory,
        profile=args.profile,
        maximum_runtime_seconds=args.maximum_runtime_seconds,
        batch_records=args.batch_records,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
