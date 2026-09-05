#!/usr/bin/env python3
"""Run the local no-download DeepMind Mathematics adapter preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_deepmind_adapter_preflight import (
    build_preflight_report,
    file_sha256,
    load_adapter_preflight_plan,
    load_bound_evidence,
    validate_preflight_report,
)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind preflight report: {path}")
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
    plan = load_adapter_preflight_plan(plan_path, repository_root=root)
    bound_evidence = load_bound_evidence(plan, repository_root=root)
    report = build_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        bound_evidence=bound_evidence,
    )
    validate_preflight_report(report)
    report_path = (root / plan["output"]["report_path"]).resolve()
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind preflight report must remain under evidence")
    _write_json_atomic(report_path, report)
    print(
        "PASS: DeepMind local no-download adapter preflight "
        f"objects={report['inventory']['objects']} bytes={report['inventory']['bytes']} "
        f"generated_records=0 math_gap="
        f"{report['remaining_gap']['projected_reasoning_math_minimum_gap_after_dmath']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
