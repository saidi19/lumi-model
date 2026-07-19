#!/usr/bin/env python3
"""Build Lumi's exact production corpus-worker release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.production_corpus import build_worker_release

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "production-acquisition-worker-v0.1.json",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = build_worker_release(
        args.contract,
        repository_root=ROOT,
        tokenizer_path=args.tokenizer,
        training_exclusion_ledger=args.training_exclusion_ledger,
        production_report=args.production_report,
    )
    rendered = json.dumps(release, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
