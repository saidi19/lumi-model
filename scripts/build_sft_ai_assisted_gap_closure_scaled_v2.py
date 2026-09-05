#!/usr/bin/env python3
"""Preflight or execute the scaled local AI-assisted gap-closure candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_ai_assisted_gap_closure_scaled_v2 import preflight, run_scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--plan", type=Path,
        default=Path("configs/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json"),
    )
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("artifacts/lumi-sft-ai-assisted-gap-closure-scaled-v0.2/attempt-001"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    if args.preflight_only:
        result = preflight(plan_path=resolve(args.plan), repository_root=root)
    else:
        result = run_scaled(
            plan_path=resolve(args.plan),
            repository_root=root,
            artifact_root=resolve(args.artifact_root),
            report_path=resolve(args.report),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
