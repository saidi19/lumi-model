#!/usr/bin/env python3
"""Preserve model-only milestone files before old trainer checkpoints are pruned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.scaling_run import snapshot_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = snapshot_model_checkpoint(args.checkpoint, args.output)
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    print(
        f"PASS: source={snapshot['source_checkpoint']} "
        f"files={len(snapshot['files'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
