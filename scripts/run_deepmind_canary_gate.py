#!/usr/bin/env python3
"""Run the fail-closed pre-network DeepMind canary dependency gate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_deepmind_canary_gate import (
    audit_runtime_dependencies,
    build_blocked_gate_report,
    file_sha256,
    json_file_bytes,
    load_canary_gate_plan,
    validate_blocked_gate_report,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind gate output: {path}")
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
    plan = load_canary_gate_plan(plan_path, repository_root=root)
    dependency_audit = audit_runtime_dependencies()
    report = build_blocked_gate_report(
        dependency_audit=dependency_audit,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_blocked_gate_report(report)
    output_path = (root / plan["output"]["report_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind canary-gate report must stay under evidence/")
    _write_bytes_atomic(output_path, json_file_bytes(report))
    print(
        "BLOCKED-SAFE: DeepMind canary stopped before network; "
        f"missing={','.join(dependency_audit['missing_modules'])}; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
