#!/usr/bin/env python3
"""Materialize the non-executing gpt-oss Banglish-repair successor report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary_v2 import (
    build_successor_report,
    load_successor_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_successor_plan(plan_path, repository_root=root)
    report = build_successor_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if args.print_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output = (root / str(plan["output"]["report_path"])).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("gpt-oss successor report escaped evidence/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite successor report: {output}")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: non-executing successor report={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
