#!/usr/bin/env python3
"""Create the no-install DeepMind dependency lock proposal."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_deepmind_dependency_resolution import (
    audit_runtime,
    build_resolution_report,
    file_sha256,
    json_file_bytes,
    load_resolution_plan,
    validate_resolution_report,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dependency output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--absl-metadata", type=Path, required=True)
    parser.add_argument("--six-metadata", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_resolution_plan(plan_path, repository_root=root)
    report = build_resolution_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        setup_payload=args.setup.read_bytes(),
        pypi_payloads={
            "absl-py": args.absl_metadata.read_bytes(),
            "six": args.six_metadata.read_bytes(),
        },
        runtime_audit=audit_runtime(),
    )
    validate_resolution_report(report)
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind dependency report must stay under evidence/")
    _write_bytes_atomic(output, json_file_bytes(report))
    print(
        "PASS: no-install DeepMind dependency lock proposal sealed; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
