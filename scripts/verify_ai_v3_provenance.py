#!/usr/bin/env python3
"""Independently replay the AI v0.3 owner-review provenance materialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_ai_v3_provenance_materialization import (
    build_ai_v3_provenance_outputs,
    build_ai_v3_provenance_verification,
    json_file_bytes,
    load_ai_v3_provenance_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite AI v0.3 verification: {path}")
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
    plan = load_ai_v3_provenance_plan(plan_path, repository_root=root)
    expected_candidate, expected_manifest, expected_report = (
        build_ai_v3_provenance_outputs(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            repository_root=root,
        )
    )
    candidate_path = (root / plan["output"]["candidate_path"]).resolve()
    manifest_path = (root / plan["output"]["manifest_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    if candidate_path.read_bytes() != expected_candidate:
        raise ValueError("AI v0.3 private provenance candidate does not reproduce")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != expected_manifest:
        raise ValueError("AI v0.3 provenance manifest does not reproduce")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report != expected_report:
        raise ValueError("AI v0.3 provenance report does not reproduce")
    verification = build_ai_v3_provenance_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: AI v0.3 provenance materialization independently reproduced; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
