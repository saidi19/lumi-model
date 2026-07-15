#!/usr/bin/env python3
"""Build the quality-gated Lumi development sample from all 23 components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_pipeline import build_sample
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs" / "development-sample-v0.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = compose_development_config(args.plan, repository_root=REPOSITORY_ROOT)
    manifest = build_sample(
        config,
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
        "sources_used": sum(source["used"] for source in manifest["source_files"]),
        "text_utf8_bytes": manifest["output"]["text_utf8_bytes"],
    }
    print("PASS: all-component development sample is complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
