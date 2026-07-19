#!/usr/bin/env python3
"""Independently verify a sealed Lumi corpus retention canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.corpus_retention import verify_retention_canary

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_retention_canary(args.output, repository_root=ROOT)
    print("PASS: retention canary manifest, database, and report agree")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
