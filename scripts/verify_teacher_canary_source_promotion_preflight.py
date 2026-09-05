#!/usr/bin/env python3
"""Independently replay and verify the teacher-canary promotion preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_teacher_canary_source_promotion_preflight import (
    build_promotion_preflight_report,
    build_verification,
    file_sha256,
    load_promotion_preflight_plan,
    validate_verification,
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite teacher-canary verification: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_promotion_preflight_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["preflight_report_path"]).resolve()
    report = _read_object(report_path)
    replay = build_promotion_preflight_report(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    verification = build_verification(
        report,
        report_file_sha256=file_sha256(report_path),
        recomputed_report=replay,
        plan=plan,
        repository_root=root,
    )
    validate_verification(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("teacher-canary verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: teacher-canary preflight independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
