#!/usr/bin/env python3
"""Independently replay the Lumi safety/tool source URI repair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_safety_uri_repair import (
    build_safety_uri_repair_verification,
    file_sha256,
    json_file_bytes,
    load_safety_uri_repair_plan,
    rebuild_safety_uri_repair_outputs,
    resolve_output,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite safety URI verification: {path}")
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
    plan = load_safety_uri_repair_plan(plan_path, repository_root=root)
    outputs = rebuild_safety_uri_repair_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    comparisons = {
        "artifact_manifest_path": outputs["artifact_manifest_bytes"],
        "private_safety_candidate_path": outputs["private_safety_candidate_bytes"],
        "report_path": json_file_bytes(outputs["report"]),
    }
    for name, expected in comparisons.items():
        path = resolve_output(root, plan["output"][name], name=name)
        if path.read_bytes() != expected:
            raise ValueError(f"safety URI repair output does not reproduce: {name}")
    report_path = resolve_output(root, plan["output"]["report_path"], name="report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = build_safety_uri_repair_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    verification_path = resolve_output(
        root,
        plan["output"]["verification_path"],
        name="verification",
    )
    _write_atomic(verification_path, json_file_bytes(verification))
    print(
        "PASS: safety URI repair outputs independently reproduced; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
