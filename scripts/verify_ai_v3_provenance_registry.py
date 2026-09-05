#!/usr/bin/env python3
"""Independently reconstruct the AI v0.3 provenance-bound registry successor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_ai_v3_provenance_materialization import json_file_bytes
from lumi.posttraining_ai_v3_provenance_registry import (
    build_ai_v3_provenance_registry_outputs,
    build_ai_v3_provenance_registry_verification,
    load_ai_v3_provenance_registry_plan,
    resolve_output,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registry verification: {path}")
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
    plan = load_ai_v3_provenance_registry_plan(plan_path, repository_root=root)
    expected_registry, expected_report = build_ai_v3_provenance_registry_outputs(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    registry_path = resolve_output(
        root,
        plan["output"]["successor_registry_path"],
        name="successor_registry_path",
    )
    report_path = resolve_output(
        root,
        plan["output"]["report_path"],
        name="report_path",
    )
    if registry_path.read_bytes() != expected_registry:
        raise ValueError("AI v0.3 successor registry does not reproduce")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report != expected_report:
        raise ValueError("AI v0.3 provenance-registry report does not reproduce")
    verification = build_ai_v3_provenance_registry_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    verification_path = resolve_output(
        root,
        plan["output"]["verification_path"],
        name="verification_path",
    )
    _write_atomic(verification_path, json_file_bytes(verification))
    print(
        "PASS: AI v0.3 provenance registry independently reproduced; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
