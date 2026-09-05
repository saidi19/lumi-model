#!/usr/bin/env python3
"""Materialize the non-authorizing targeted Lumi SFT closure plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_targeted_closure_plan import (
    build_report,
    load_plan,
    sha256_file,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path.resolve(), repository_root=root)
    report = build_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("targeted-closure report escaped evidence/")
    if output.exists():
        raise FileExistsError("refusing to overwrite targeted-closure report")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("targeted-closure report exceeded write limit")
    print(
        "PASS: sealed non-authorizing targeted closure plan "
        f"target={report['target_closure']['target_train_supervised_tokens']} "
        f"projected_total={report['target_closure']['projected_train_supervised_tokens']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
