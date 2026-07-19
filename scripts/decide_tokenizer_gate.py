#!/usr/bin/env python3
"""Evaluate the pinned Lumi final-tokenizer structural and efficiency gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from lumi.tokenizer_gate import evaluate_tokenizer_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--reserved-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write a valid fail decision without returning a nonzero process status",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite gate report: {args.output}")
    report = evaluate_tokenizer_gate(
        _read_json(args.config),
        _read_json(args.comparison),
        _read_json(args.reserved_audit),
    )
    report["created_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    report["inputs"] = {
        "comparison": str(args.comparison.resolve()),
        "config": str(args.config.resolve()),
        "reserved_audit": str(args.reserved_audit.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{report['status'].upper()}: {report['decision']}")
    if report["status"] != "pass" and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
