#!/usr/bin/env python3
"""Seal the no-download rights-clear math replacement preflight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_math_replacement_preflight import (
    file_sha256,
    json_file_bytes,
    load_math_replacement_plan,
    rebuild_math_replacement_preflight,
    validate_math_replacement_report,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite math preflight report: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
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
    plan = load_math_replacement_plan(plan_path, repository_root=root)
    report = rebuild_math_replacement_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    validate_math_replacement_report(report)
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("math replacement report must remain under evidence/")
    _write_bytes_atomic(output, json_file_bytes(report))
    print(
        "PASS: rights-clear math replacement preflight; "
        f"registered={report['mix_gap']['registered_reviewed_train_supervised_tokens']} "
        f"planning={report['mix_gap']['planning_train_supervised_tokens']} "
        f"math_gap={report['mix_gap']['reasoning_math_minimum_gap_after_reviewed_draft']} "
        f"next=dmath report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
