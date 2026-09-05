#!/usr/bin/env python3
"""Independently replay and verify the targeted Lumi SFT closure plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_targeted_closure_plan import (
    build_report,
    build_verification,
    load_plan,
    sha256_file,
    validate_report,
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
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite targeted-closure verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    reconstructed = build_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    if report != reconstructed:
        raise ValueError("targeted-closure plan replay changed")
    verification = build_verification(
        plan=plan,
        report=report,
        report_file_sha256=sha256_file(report_path),
    )
    write_json_atomic(verification_path, verification)
    if verification_path.stat().st_size > plan["limits"]["maximum_verification_bytes"]:
        raise ValueError("targeted-closure verification exceeded write limit")
    print(
        "PASS: independently verified targeted closure plan "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
