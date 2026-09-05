#!/usr/bin/env python3
"""Run the bounded CPU-only Lumi SFT closure-canary curation."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_closure_canary_curation import run_curation, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--data-quality", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, _ = run_curation(
        repository_root=args.repository_root,
        raw_root=args.raw_root,
        private_root=args.private_root,
        plan_path=args.plan,
        tokenizer_path=args.tokenizer,
        chat_template_path=args.chat_template,
        data_contract_path=args.data_contract,
        data_quality_path=args.data_quality,
    )
    write_report(args.report, report)
    print(report["status"])
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
