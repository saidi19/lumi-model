#!/usr/bin/env python3
"""Reproduce and independently verify a sealed Lumi corpus-storage plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.corpus_storage import verify_corpus_storage_report

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "corpus-object-storage-v0.1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = verify_corpus_storage_report(
        args.contract,
        args.report,
        args.manifest,
        repository_root=ROOT,
    )
    print("PASS: storage-plan hash and deterministic reproduction agree")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
