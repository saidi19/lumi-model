#!/usr/bin/env python3
"""Independently replay the CPU-only local Aya capacity scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_aya_capacity_scan import (
    build_report,
    build_verification,
    file_sha256,
    load_plan,
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
        raise FileExistsError("refusing to overwrite Aya capacity verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)

    def progress(rows: int) -> None:
        print(f"REPLAY: Aya rows verified {rows}/1200429", flush=True)

    replay = build_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
        progress=progress,
    )
    if report != replay:
        raise ValueError("Aya capacity full replay changed")
    verification = build_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    write_json_atomic(verification_path, verification)
    combined = report_path.stat().st_size + verification_path.stat().st_size
    if combined > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("Aya capacity outputs exceeded private-write ceiling")
    print(
        "PASS: independently replayed Aya capacity scan "
        f"verification={verification['verification_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
