#!/usr/bin/env python3
"""Build the raw-text-free local teacher canary preflight report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_first_party_teacher_preflight import (
    build_first_party_teacher_preflight_report,
    file_sha256,
    load_first_party_teacher_preflight_plan,
    validate_first_party_teacher_preflight_report,
)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite teacher preflight report: {path}")
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
    report = build_first_party_teacher_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_first_party_teacher_preflight_report(report)
    report_path = (root / plan["output"]["report_path"]).resolve()
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("teacher preflight report must remain under evidence")
    _write_json_atomic(report_path, report)
    print(
        "PASS: teacher preflight metadata-only "
        f"model={report['model']['metadata']['model_id']} "
        f"candidate_records={report['authorization']['candidate_records_generated']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

