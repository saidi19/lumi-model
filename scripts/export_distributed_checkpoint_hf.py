#!/usr/bin/env python3
"""Export one verified Lumi distributed checkpoint for held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.distributed_export import export_distributed_checkpoint_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_distributed_checkpoint_hf(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        dtype=args.dtype,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
