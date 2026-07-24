#!/usr/bin/env python3
"""Evaluate sealed Lumi-4B Segment-2 throughput evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.segment2_throughput_gate import evaluate_segment2_throughput_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = evaluate_segment2_throughput_files(
        contract_path=args.contract,
        evidence_path=args.evidence,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite decision: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: winner={decision['winner']} "
        f"checkpoint={decision['checkpoint_manifest_sha256']} "
        f"decision={decision['decision_sha256']}"
    )


if __name__ == "__main__":
    main()
