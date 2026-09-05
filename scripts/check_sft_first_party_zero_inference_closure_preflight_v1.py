#!/usr/bin/env python3
"""Materialize the plan-only first-party zero-inference closure preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_first_party_zero_inference_closure_preflight_v1 import (
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
        raise ValueError("zero-inference preflight report escaped evidence/")
    if output.exists():
        raise FileExistsError("refusing to overwrite zero-inference preflight report")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("zero-inference preflight report exceeded write limit")
    print(
        "PASS: plan-only first-party zero-inference closure preflight ready "
        "target=1200000 candidate_construction=false "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
