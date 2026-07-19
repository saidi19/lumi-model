#!/usr/bin/env python3
"""Independently verify the sealed Lumi source-file manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.source_file_manifest import verify_source_file_manifest

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "source-file-manifest-v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_source_file_manifest(
        args.output,
        contract_path=args.contract,
        repository_root=ROOT,
    )
    print("PASS: source-file hashes, SQLite state, snapshot, and report agree")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
