#!/usr/bin/env python3
"""Run a metadata-only SFT readiness audit or fresh structural replay; never launch."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_gpu_smoke_readiness_v1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run(args.repository_root.resolve(), args.expected_plan_sha256, verify=args.verify)
    print(result["status"])


if __name__ == "__main__":
    main()
