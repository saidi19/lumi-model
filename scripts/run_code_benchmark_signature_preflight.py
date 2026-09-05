#!/usr/bin/env python3
"""Materialize the raw-text-free coding benchmark signature preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_code_benchmark_preflight import (
    build_preflight_outputs,
    load_preflight_plan,
    validate_acquisition_plan,
    validate_report,
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark preflight output: {path}")
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
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    report, acquisition_plan = build_preflight_outputs(plan, repository_root=root)
    validate_report(report)
    validate_acquisition_plan(acquisition_plan)
    report_path = (root / plan["output"]["report_path"]).resolve()
    acquisition_path = (root / plan["output"]["acquisition_plan_path"]).resolve()
    if not report_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("benchmark preflight report must stay under evidence/")
    if not acquisition_path.is_relative_to((root / "configs").resolve()):
        raise ValueError("benchmark acquisition plan must stay under configs/")
    if report_path.exists() or acquisition_path.exists():
        raise FileExistsError("refusing to overwrite benchmark preflight outputs")
    _write_json_atomic(acquisition_path, acquisition_plan)
    _write_json_atomic(report_path, report)
    print(
        "PASS: code benchmark metadata preflight "
        f"metadata_bytes={report['captured_metadata']['payload_bytes']} "
        f"bigcodebench={report['decisions']['bigcodebench']} "
        f"livecodebench={report['decisions']['livecodebench']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
