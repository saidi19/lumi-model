#!/usr/bin/env python3
"""Build the immutable Segment-2 checkpoint plan before owner approval."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from lumi.segment3_checkpoint_verification import (
    build_checkpoint_gate_plan,
    write_checkpoint_gate_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance-boot-utc", required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_checkpoint_gate_plan(
        source_checkpoint=args.source_checkpoint,
        manifest_path=args.manifest,
        heldout_manifest_path=args.heldout_manifest,
        tokenizer_path=args.tokenizer,
        config_path=args.config,
        evaluation_plan_path=args.evaluation_plan,
        source_commit=args.source_commit,
        instance_boot_utc=args.instance_boot_utc,
        created_at=(
            None
            if args.created_at is None
            else datetime.fromisoformat(args.created_at)
        ),
    )
    write_checkpoint_gate_plan(plan=plan, output=args.output)
    print(f"PASS: plan={plan['plan_sha256']}", flush=True)


if __name__ == "__main__":
    main()
