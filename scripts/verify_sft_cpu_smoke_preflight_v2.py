#!/usr/bin/env python3
"""Independently verify the metadata-only v0.2 CPU SFT smoke preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_sft_cpu_smoke_preflight_v2 import run_verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verification = run_verification(
        plan_path=args.plan,
        report_path=args.report,
        repository_root=args.repository_root,
    )
    print(f"PASS: {verification['status']} rows=0 training=0")


if __name__ == "__main__":
    main()
