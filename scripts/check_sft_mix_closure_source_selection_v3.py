#!/usr/bin/env python3
"""Build the registry-v0.15 Lumi SFT 10M closure update report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_mix_closure_source_selection_v3 import (
    json_file_bytes,
    load_mix_closure_source_selection_plan_v3,
    rebuild_mix_closure_source_selection_v3,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v0.3 closure report: {path}")
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
    plan_path = plan_path.resolve()
    plan = load_mix_closure_source_selection_plan_v3(
        plan_path, repository_root=root
    )
    report = rebuild_mix_closure_source_selection_v3(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("v0.3 closure report escaped evidence/")
    _write_atomic(output, json_file_bytes(report))
    projected = report["projected_mix"]
    print(
        "PASS: registry-v0.15 10M closure update verified; "
        f"projected={projected['train_supervised_tokens']} "
        f"task_buffer={projected['buffer_above_minimum_task_closure']} "
        f"floor_buffer={projected['buffer_above_10m_floor']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
