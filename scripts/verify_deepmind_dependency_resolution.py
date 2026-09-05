#!/usr/bin/env python3
"""Independently replay the no-install DeepMind dependency resolution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_dependency_resolution import (
    VERIFICATION_SCHEMA_VERSION,
    audit_runtime,
    build_resolution_report,
    file_sha256,
    load_resolution_plan,
    sha256_value,
    validate_resolution_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dependency verification: {path}")
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
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--absl-metadata", type=Path, required=True)
    parser.add_argument("--six-metadata", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_resolution_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = _read_json(report_path)
    validate_resolution_report(report)
    replayed = build_resolution_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        setup_payload=args.setup.read_bytes(),
        pypi_payloads={
            "absl-py": args.absl_metadata.read_bytes(),
            "six": args.six_metadata.read_bytes(),
        },
        runtime_audit=audit_runtime(),
    )
    if report != replayed:
        raise ValueError("DeepMind dependency resolution did not replay exactly")
    verification: dict[str, Any] = {
        "checks": {
            "canary_records_generated": 0,
            "declared_requirements_ast_replayed": True,
            "exact_pypi_metadata_replayed": True,
            "exact_setup_blob_replayed": True,
            "packages_installed": 0,
            "report_reconstructed": True,
            "runtime_compatibility_verified": True,
            "wheel_hashes_verified_from_pypi_metadata": True,
        },
        "observed_at": plan["observed_at"],
        "packages_to_install": report["next_gate"]["packages_to_install"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_no_install_deepmind_dependency_lock_proposal",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind dependency verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified no-install dependency lock; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
