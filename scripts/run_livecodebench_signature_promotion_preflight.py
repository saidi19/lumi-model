#!/usr/bin/env python3
"""Materialize the raw-text-free LiveCodeBench promotion preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_bigcodebench_signatures import write_json_atomic
from lumi.posttraining_livecodebench_promotion_preflight import (
    audit_and_build_outputs,
    load_preflight_plan,
    validate_promotion_plan,
    validate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    report, promotion_plan = audit_and_build_outputs(plan, repository_root=root)
    validate_report(report)
    validate_promotion_plan(promotion_plan)
    report_path = (root / plan["output"]["report_path"]).resolve()
    promotion_path = (root / plan["output"]["promotion_plan_path"]).resolve()
    if not report_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("LiveCodeBench promotion-preflight report escaped evidence/")
    if not promotion_path.is_relative_to((root / "configs").resolve()):
        raise ValueError("LiveCodeBench promotion plan escaped configs/")
    write_json_atomic(promotion_path, promotion_plan)
    write_json_atomic(report_path, report)
    print(
        "PASS: LiveCodeBench signature promotion preflight "
        f"candidate_entries={report['audit']['candidate_entries']} "
        f"proposed_registry={promotion_plan['proposed_registry_successor']['file_sha256']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
