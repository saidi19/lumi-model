#!/usr/bin/env python3
"""Seal the exact pre-Segment-2 throughput-screen plan and authorization."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.pre_segment2_throughput_screen import (
    build_pre_segment2_screen_authorization,
    build_pre_segment2_screen_plan,
    write_pre_segment2_screen_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance-boot-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_pre_segment2_screen_plan(
        contract_path=args.contract,
        manifest_path=args.manifest,
        heldout_manifest_path=args.heldout_manifest,
        tokenizer_path=args.tokenizer,
        config_path=args.config,
        source_commit=args.source_commit,
        instance_boot_utc=args.instance_boot_utc,
    )
    authorization = build_pre_segment2_screen_authorization(plan)
    write_pre_segment2_screen_bundle(
        plan=plan,
        authorization=authorization,
        output=args.output,
    )
    print(
        f"PASS: plan={plan['plan_sha256']} "
        f"authorization={authorization['authorization_sha256']} "
        f"deadline={authorization['deadline_utc']}"
    )


if __name__ == "__main__":
    main()
