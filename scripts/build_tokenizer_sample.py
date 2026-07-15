#!/usr/bin/env python3
"""Build a small provenance-tracked tokenizer sample from pinned sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_pipeline import build_sample, load_sample_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "tokenizer-canary-v0.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_sample(
        load_sample_config(args.config, quality_path=args.quality_config),
        args.output,
        target_bytes=args.target_bytes,
        resume=args.resume,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    summary = {
        "documents": manifest["statistics"]["documents"],
        "output": str(args.output),
        "sha256": manifest["output"]["sha256"],
        "text_utf8_bytes": manifest["output"]["text_utf8_bytes"],
    }
    print("PASS: data sample is complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
