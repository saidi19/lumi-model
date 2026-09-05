#!/usr/bin/env python3
"""Materialize the non-authorizing v0.21 targeted-gap source selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_targeted_gap_closure_source_selection_v1 import (
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
        raise FileExistsError("refusing to overwrite targeted-gap report")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("targeted-gap report exceeded write limit")
    print(
        "PASS: local source-selection preflight stopped fail-closed "
        f"unconditional_gap={report['source_selection']['unconditional_remaining_strict_gap']} "
        f"conditional_gap={report['source_selection']['conditional_candidate']['strict_gap_after_clearance']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
