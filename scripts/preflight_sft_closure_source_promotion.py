#!/usr/bin/env python3
"""Run the non-mutating closure-candidate registry admissibility preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_closure_source_promotion_preflight import (
    build_preflight_report,
    load_preflight_plan,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    report = build_preflight_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("closure source-promotion preflight escaped evidence/")
    if output.exists():
        raise FileExistsError("refusing to overwrite closure source-promotion preflight")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_report_bytes"]:
        raise ValueError("closure source-promotion preflight exceeded write limit")
    print(
        "PASS: closure registry preflight failed closed as required "
        f"blockers={len(report['blockers'])} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
