#!/usr/bin/env python3
"""Build the non-authorizing Lumi Glue final-corpus canary package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lumi.glue_final_corpus import (
    build_glue_final_corpus_canary_package,
    expected_glue_final_corpus_canary_approval,
)

ROOT = Path(__file__).parents[1]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "glue-final-corpus-canary-v0.1.json",
    )
    parser.add_argument("--final-plan", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def _source_commit(value: str | None) -> str:
    if value:
        return value
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = _args()
    release = build_glue_final_corpus_canary_package(
        contract_path=args.contract,
        final_plan_path=args.final_plan,
        repository_root=ROOT,
        source_commit=_source_commit(args.source_commit),
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "approval_text": expected_glue_final_corpus_canary_approval(release),
                "maximum_usd": release["job"]["maximum_compute_usd"],
                "plan_fingerprint": release["input"]["plan_fingerprint"],
                "release_fingerprint": release["release_fingerprint"],
                "status": "canary_package_ready_not_authorized",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
