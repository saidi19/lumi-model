#!/usr/bin/env python3
"""Independently reproduce and verify the host-bound gpt-oss canary plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_teacher_canary import (
    build_host_bound_report,
    build_host_bound_verification,
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
    report_path = (root / str(plan["output"]["report_path"])).resolve()
    observed = json.loads(report_path.read_text(encoding="utf-8"))
    rebuilt = build_host_bound_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if observed != rebuilt:
        raise ValueError("gpt-oss host-bound report does not reproduce")
    verification = build_host_bound_verification(
        report=observed,
        report_file_sha256=file_sha256(report_path),
    )
    output = (root / str(plan["output"]["verification_path"])).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("gpt-oss host-bound verification escaped evidence/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite host-bound verification: {output}")
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: exact H100 host-bound plan independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

