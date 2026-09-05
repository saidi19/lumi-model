#!/usr/bin/env python3
"""Materialize the registry-v0.22 targeted-gap selection rebase."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_targeted_gap_closure_source_selection_v2 import (
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
        raise ValueError("targeted-gap report escaped evidence/")
    if output.exists():
        raise FileExistsError("refusing to overwrite targeted-gap v0.2 report")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("targeted-gap report exceeded write limit")
    print(
        "PASS: registry-v0.22 targeted-gap rebase selected zero-inference canary "
        f"remaining_gap={report['current_verified_state']['strict_targeted_gap_lower_bound']} "
        f"target={report['replacement_capacity_target']['retained_train_supervised_tokens']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
