#!/usr/bin/env python3
"""Structurally verify the two-pass AI-assisted canary without a third row read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_ai_assisted_gap_closure_canary_v1 import verify_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/lumi-4b-sft-ai-assisted-gap-closure-canary-v0.1.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "artifacts/lumi-sft-ai-assisted-gap-closure-canary-v0.1/attempt-001"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-v0.1.json"),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path(
            "evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-verification-v0.1.json"
        ),
    )
    args = parser.parse_args()

    root = args.repository_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    verification_path = resolve(args.verification)
    if verification_path.exists():
        raise FileExistsError("versioned verification already exists; overwrite forbidden")
    result = verify_report(
        plan_path=resolve(args.plan),
        repository_root=root,
        artifact_root=resolve(args.artifact_root),
        report_path=resolve(args.report),
    )
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

