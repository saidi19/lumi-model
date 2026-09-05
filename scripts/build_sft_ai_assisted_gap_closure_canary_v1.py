#!/usr/bin/env python3
"""Preflight or execute the local AI-assisted gap-closure canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_ai_assisted_gap_closure_canary_v1 import preflight, run_canary


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
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan = args.plan if args.plan.is_absolute() else root / args.plan
    if args.preflight_only:
        result = preflight(plan_path=plan, repository_root=root)
    else:
        artifact = (
            args.artifact_root
            if args.artifact_root.is_absolute()
            else root / args.artifact_root
        )
        report = args.report if args.report.is_absolute() else root / args.report
        result = run_canary(
            plan_path=plan,
            repository_root=root,
            artifact_root=artifact,
            report_path=report,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

