#!/usr/bin/env python3
"""Inspect or enforce the final Lumi serving readiness contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lumi.serving_contract import load_serving_contract, serving_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = serving_readiness(load_serving_contract(args.contract))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and not report["ready"]:
        print("BLOCKED: serving gates are incomplete", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
