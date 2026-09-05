#!/usr/bin/env python3
"""Create the exact private PyArrow overlay plan without downloading a wheel."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_closure_decoder_preflight import build_overlay_plan, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--acquisition-verification", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan, report = build_overlay_plan(
        metadata_path=args.metadata,
        acquisition_plan_path=args.acquisition_plan,
        acquisition_report_path=args.acquisition_report,
        acquisition_verification_path=args.acquisition_verification,
        python_executable=args.python,
    )
    write_json(args.plan_output, plan)
    write_json(args.report_output, report)
    print(plan["plan_sha256"])
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
