#!/usr/bin/env python3
"""Run the bounded local BigCodeBench re-audit of OpenCodeInstruct."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_bigcodebench_signatures import write_json_atomic
from lumi.posttraining_opencodeinstruct_bigcodebench_reaudit import (
    build_report,
    load_reaudit_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_reaudit_plan(plan_path.resolve(), repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    if report_path.exists():
        raise FileExistsError("refusing to overwrite OpenCodeInstruct BigCodeBench report")
    report = build_report(plan, repository_root=root)
    write_json_atomic(report_path, report)
    if report_path.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("OpenCodeInstruct BigCodeBench report exceeded write limit")
    print(
        "PASS: OpenCodeInstruct BigCodeBench v0.2 re-audit complete "
        f"matches={report['results']['matched_records']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
