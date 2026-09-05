#!/usr/bin/env python3
"""Finalize the private owner spot-check for the Lumi SFT closure canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_owner_review import read_json, validate_owner_decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sample-sha256", required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--evidence-receipt", type=Path, required=True)
    return parser.parse_args()


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite owner-review receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    decisions = read_json(args.decisions)
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    if not isinstance(sample, list) or not all(isinstance(row, dict) for row in sample):
        raise ValueError("owner-review sample must be a JSON array of objects")
    receipt = validate_owner_decisions(
        decisions,
        sample_rows=sample,
        sample_sha256=args.sample_sha256,
    )
    _write_new(args.private_receipt, receipt)
    _write_new(args.evidence_receipt, receipt)
    print(receipt["status"])
    print(receipt["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
