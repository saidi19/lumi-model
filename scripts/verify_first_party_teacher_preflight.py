#!/usr/bin/env python3
"""Independently reconstruct and verify the local teacher canary preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_first_party_teacher_preflight import (
    build_first_party_teacher_preflight_report,
    build_first_party_teacher_preflight_verification,
    file_sha256,
    load_first_party_teacher_preflight_plan,
    validate_first_party_teacher_preflight_report,
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite teacher verification: {path}")
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
    plan = load_first_party_teacher_preflight_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = _load_json(report_path)
    validate_first_party_teacher_preflight_report(report)
    rebuilt = build_first_party_teacher_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    if rebuilt != report:
        raise ValueError("teacher preflight report reconstruction changed")
    verification = build_first_party_teacher_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if not verification_path.is_relative_to(root / "evidence"):
        raise ValueError("teacher preflight verification must remain under evidence")
    _write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified teacher preflight "
        f"candidate_records={verification['checks']['candidate_records_generated']} "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

