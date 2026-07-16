#!/usr/bin/env python3
"""Verify the sealed Lumi pilot bundle before paid GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.cloud_pilot import preflight_cloud_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless CUDA and BF16 are available",
    )
    parser.add_argument(
        "--skip-shard-hashes",
        action="store_true",
        help="verify shard structure and sizes without rehashing shard contents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = preflight_cloud_pilot(
        plan_path=args.plan,
        manifest_path=args.manifest,
        tokenizer_path=args.tokenizer,
        require_cuda=args.require_cuda,
        verify_shards=not args.skip_shard_hashes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"PASS: parameters={report['parameters']} "
        f"tokens={report['artifact']['tokens']} "
        f"cuda={report['runtime']['cuda_available']}"
    )


if __name__ == "__main__":
    main()
