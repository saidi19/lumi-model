#!/usr/bin/env python3
"""Build the deterministic, planning-only Lumi corpus capacity report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.corpus_capacity import build_corpus_capacity_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/corpus-capacity-gate-v0.1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_corpus_capacity_report(args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    report_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            f"{report_sha256}  {args.output.name}\n",
            encoding="utf-8",
        )
    summary = report["summary"]
    print(
        "PASS: planning metadata verified; "
        f"estimated_tokens={summary['estimated_published_tokens']} "
        f"compressed_bytes={summary['tree_compressed_bytes']} "
        f"report_sha256={report_sha256} "
        "bulk_materialization=blocked training=not_authorized"
    )


if __name__ == "__main__":
    main()
