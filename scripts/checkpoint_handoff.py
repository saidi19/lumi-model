#!/usr/bin/env python3
"""Seal or verify a portable Lumi distributed checkpoint handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.production_training import seal_checkpoint_handoff, verify_checkpoint_handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--checkpoint", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--source-commit", required=True)
    seal.add_argument("--run-fingerprint", required=True)
    seal.add_argument("--world-size", type=int, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--handoff", type=Path, required=True)
    verify.add_argument("--source-commit", required=True)
    verify.add_argument("--run-fingerprint", required=True)
    verify.add_argument("--world-size", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seal":
        result = seal_checkpoint_handoff(
            args.checkpoint,
            args.output,
            source_commit=args.source_commit,
            run_fingerprint=args.run_fingerprint,
            world_size=args.world_size,
        )
    else:
        result = verify_checkpoint_handoff(
            args.checkpoint,
            args.handoff,
            expected_source_commit=args.source_commit,
            expected_run_fingerprint=args.run_fingerprint,
            expected_world_size=args.world_size,
        )
    print(
        json.dumps(
            {
                "files": len(result["files"]),
                "handoff_sha256": result["handoff_sha256"],
                "status": "PASS",
                "step": result["checkpoint"]["step"],
                "tokens_seen": result["checkpoint"]["tokens_seen"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
