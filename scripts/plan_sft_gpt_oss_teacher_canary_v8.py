#!/usr/bin/env python3
"""Materialize the non-executing v0.8 full-canary plan report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary_v8 import build_plan_report, load_full_canary_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    root, plan_path = args.repository_root.resolve(), args.plan.resolve()
    plan = load_full_canary_plan(plan_path, repository_root=root)
    report = build_plan_report(plan=plan, plan_file_sha256=file_sha256(plan_path), repository_root=root)
    if args.print_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()) or output.exists():
        raise ValueError("unsafe or existing v0.8 report path")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: non-executing v0.8 report={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
