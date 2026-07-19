#!/usr/bin/env python3
"""Build Lumi's local-only distributed corpus migration plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.distributed_corpus import build_distributed_corpus_plan

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "distributed-production-corpus-v0.1.json",
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _source_commit(raw: str | None) -> str:
    if raw is not None:
        return raw
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    plan = build_distributed_corpus_plan(
        args.contract,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
    )
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
