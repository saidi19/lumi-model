#!/usr/bin/env python3
"""Verify the staged Lumi SFT benchmark schedule and emit readiness evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_benchmark_gates import (
    build_benchmark_gate_readiness_report,
    load_benchmark_gate_plan,
    verify_benchmark_gate_readiness_report,
)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_benchmark_gate_plan(plan_path, repository_root=root)
    report = build_benchmark_gate_readiness_report(plan, repository_root=root)
    verify_benchmark_gate_readiness_report(report)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        if not output.resolve().is_relative_to(root / "evidence"):
            raise ValueError("benchmark readiness output must stay under evidence/")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite benchmark evidence: {output}")
        _write_json_atomic(output, report)
    print(
        "PASS: benchmark schedule verified; "
        f"pending_adapters={len(report['adapter_build_pending'])}; "
        f"candidate_tokens={report['candidate_pool']['accepted_train_supervised_tokens']}; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
