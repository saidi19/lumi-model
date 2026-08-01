#!/usr/bin/env python3
"""Seal the owner-approved pre-Segment-3 optimization screen."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.pre_segment3_throughput_screen import (
    build_authorization,
    build_plan,
    write_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance-boot-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        contract_path=args.contract,
        manifest_path=args.manifest,
        heldout_manifest_path=args.heldout_manifest,
        tokenizer_root=args.tokenizer_root,
        config_path=args.config,
        source_commit=args.source_commit,
        instance_boot_utc=args.instance_boot_utc,
    )
    authorization = build_authorization(plan)
    write_bundle(plan=plan, authorization=authorization, output=args.output)
    print(f"plan_sha256={plan['plan_sha256']}")
    print(f"authorization_sha256={authorization['authorization_sha256']}")
    print(f"deadline_utc={authorization['deadline_utc']}")


if __name__ == "__main__":
    main()
