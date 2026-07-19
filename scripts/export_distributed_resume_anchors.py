#!/usr/bin/env python3
"""Export and independently verify Lumi's distributed resume anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.distributed_baseline import export_resume_anchors, verify_resume_anchor_export

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--baseline-contract",
        type=Path,
        default=ROOT / "configs" / "distributed-baseline-v0.1.json",
    )
    parser.add_argument(
        "--quality-contract",
        type=Path,
        default=ROOT / "configs" / "data-quality-v0.1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_resume_anchors(
        baseline_contract_path=args.baseline_contract,
        quality_contract_path=args.quality_contract,
        evidence_directory=args.evidence_directory,
        output_directory=args.output_directory,
    )
    verified = verify_resume_anchor_export(
        args.output_directory,
        args.quality_contract,
    )
    print(json.dumps(verified, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
