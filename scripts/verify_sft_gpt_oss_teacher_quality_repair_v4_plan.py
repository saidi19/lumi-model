#!/usr/bin/env python3
"""Independently reproduce the gpt-oss prompt-quality repair plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_quality_repair_v4 import build_plan_report, build_plan_verification, load_quality_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    root, plan_path = args.repository_root.resolve(), args.plan.resolve()
    plan = load_quality_plan(plan_path, repository_root=root)
    report_path = (root / str(plan["output"]["report_path"])).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report != build_plan_report(plan=plan, plan_file_sha256=file_sha256(plan_path), repository_root=root):
        raise ValueError("gpt-oss quality-repair report does not reproduce")
    verification = build_plan_verification(report=report, report_file_sha256=file_sha256(report_path))
    if args.print_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output = (root / str(plan["output"]["verification_path"])).resolve()
    if not output.is_relative_to((root / "evidence").resolve()) or output.exists():
        raise ValueError("unsafe or existing gpt-oss quality-repair verification path")
    output.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: quality-repair verification={verification['verification_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
