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
        "--workspace",
        type=Path,
        help="existing directory whose free space is checked for 1B checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path written atomically",
    )
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
        workspace_path=args.workspace,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    print(
        f"PASS: parameters={report['parameters']} "
        f"tokens={report['artifact']['tokens']} "
        f"cuda={report['runtime']['cuda_available']} "
        f"runtime_ready={report['runtime'].get('runtime_ready', 'not-enforced')}"
    )


if __name__ == "__main__":
    main()
