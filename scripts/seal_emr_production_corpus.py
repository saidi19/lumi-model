#!/usr/bin/env python3
"""Seal verified Lumi EMR production outputs without calling AWS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.emr_production_seal import build_final_corpus_seal, write_final_corpus_seal


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-checkpoint", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--dedup-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-writer-result", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    seal = build_final_corpus_seal(
        _read(args.plan),
        _read(args.accepted_checkpoint),
        _read(args.authorization),
        _read(args.dedup_result),
        [_read(path) for path in args.shard_writer_result],
    )
    report = write_final_corpus_seal(seal, args.output)
    print(json.dumps({**report, "status": seal["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
