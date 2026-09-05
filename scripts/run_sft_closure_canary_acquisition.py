#!/usr/bin/env python3
"""Write the raw-text-free Lumi SFT closure-canary acquisition receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_closure_canary_acquisition import (
    build_acquisition_report,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--benchmark-registry", type=Path, required=True)
    parser.add_argument("--data-quality", type=Path, required=True)
    parser.add_argument("--tokenizers-overlay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_acquisition_report(
        repository_root=args.repository_root,
        plan_path=args.plan,
        private_root=args.private_root,
        tokenizer_path=args.tokenizer,
        benchmark_registry_path=args.benchmark_registry,
        data_quality_path=args.data_quality,
        tokenizers_overlay=args.tokenizers_overlay,
    )
    write_report(args.output, report)
    print(report["status"])
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
