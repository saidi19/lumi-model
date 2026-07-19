#!/usr/bin/env python3
"""Provision the approved bounded AWS CPU corpus-acquisition worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.aws_production_worker import (
    build_worker_launch_plan,
    provision_production_worker,
    wait_for_worker_online,
)

ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-release", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--pricing",
        type=Path,
        default=ROOT / "configs" / "aws-production-worker-pricing-v0.1.json",
    )
    parser.add_argument("--skip-ssm-wait", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_worker_launch_plan(
        args.worker_release,
        production_report_path=args.production_report,
        pricing_path=args.pricing,
        source_bundle_path=args.source_bundle,
        source_commit=args.source_commit,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_directory / "launch-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = provision_production_worker(
        plan,
        worker_release_path=args.worker_release,
        production_report_path=args.production_report,
        source_bundle_path=args.source_bundle,
        output_directory=args.output_directory,
        profile=args.profile,
    )
    result: dict[str, object] = {"launch": report}
    if not args.skip_ssm_wait:
        result["ssm"] = wait_for_worker_online(report, profile=args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
