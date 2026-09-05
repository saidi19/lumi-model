#!/usr/bin/env python3
"""Independently replay the blocked DeepMind canary dependency gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_canary_gate import (
    VERIFICATION_SCHEMA_VERSION,
    audit_runtime_dependencies,
    build_blocked_gate_report,
    file_sha256,
    load_canary_gate_plan,
    sha256_value,
    validate_blocked_gate_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind verification: {path}")
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
    plan = load_canary_gate_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = _read_json(report_path)
    validate_blocked_gate_report(report)
    replayed = build_blocked_gate_report(
        dependency_audit=audit_runtime_dependencies(),
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    if report != replayed:
        raise ValueError("DeepMind blocked gate did not replay exactly")

    verification: dict[str, Any] = {
        "checks": {
            "canary_generation_attempted": False,
            "dependency_blocker_replayed": True,
            "network_bytes_observed": 0,
            "package_installation_attempted": False,
            "production_promotions": 0,
            "raw_training_text_present": False,
            "report_reconstructed": True,
            "source_objects_fetched": 0,
        },
        "missing_modules": report["dependency_audit"]["missing_modules"],
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_fail_closed_before_deepmind_network_access",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output_path = (root / plan["output"]["verification_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind verification must stay under evidence/")
    _write_json_atomic(output_path, verification)
    print(
        "PASS: independently verified pre-network DeepMind stop; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
