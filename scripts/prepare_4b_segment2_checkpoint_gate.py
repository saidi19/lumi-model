#!/usr/bin/env python3
"""Seal the exact-checkpoint pre-Segment-2 verification bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.segment2_checkpoint_verification import (
    build_checkpoint_gate_bundle,
    write_checkpoint_gate_bundle,
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan, authorization = build_checkpoint_gate_bundle(
        source_checkpoint=args.source_checkpoint,
        manifest_path=args.manifest,
        heldout_manifest_path=args.heldout_manifest,
        tokenizer_path=args.tokenizer,
        config_path=args.config,
        evaluation_plan_path=args.evaluation_plan,
        source_commit=args.source_commit,
        instance_boot_utc=args.instance_boot_utc,
    )
    write_checkpoint_gate_bundle(
        plan=plan,
        authorization=authorization,
        output=args.output,
    )
    print(
        f"PASS: plan={plan['plan_sha256']} "
        f"authorization={authorization['authorization_sha256']} "
        f"deadline={authorization['deadline_utc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
