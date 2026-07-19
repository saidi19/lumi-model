#!/usr/bin/env python3
"""Run the bounded Lumi corpus retention canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.corpus_retention import run_retention_canary

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "corpus-retention-canary-v0.1.json",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-forever", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_retention_canary(
        args.contract,
        args.tokenizer,
        args.output,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        retry_forever=args.retry_forever,
    )
    print("PASS: corpus retention canary is complete")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
