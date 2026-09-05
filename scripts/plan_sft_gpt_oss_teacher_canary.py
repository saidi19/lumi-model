#!/usr/bin/env python3
"""Materialize the non-executing host-bound gpt-oss canary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary import (
    build_host_bound_report,
    load_host_bound_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_host_bound_plan(plan_path, repository_root=root)
    report = build_host_bound_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / str(plan["output"]["report_path"])).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("gpt-oss host-bound report escaped evidence/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite host-bound report: {output}")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: exact H100 host-bound plan sealed without execution; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

