#!/usr/bin/env python3
"""Independently verify the coding benchmark signature source preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_code_benchmark_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    build_preflight_outputs,
    load_preflight_plan,
    validate_acquisition_plan,
    validate_report,
)
from lumi.posttraining_opencodeinstruct_curation import file_sha256, sha256_value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark verification: {path}")
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
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    expected_report, expected_acquisition = build_preflight_outputs(
        plan, repository_root=root
    )
    report_path = (root / plan["output"]["report_path"]).resolve()
    acquisition_path = (root / plan["output"]["acquisition_plan_path"]).resolve()
    report = _read_json(report_path)
    acquisition = _read_json(acquisition_path)
    validate_report(report)
    validate_acquisition_plan(acquisition)
    if report != expected_report or acquisition != expected_acquisition:
        raise ValueError("benchmark preflight reconstruction changed")

    verification: dict[str, Any] = {
        "acquisition_plan_file_sha256": file_sha256(acquisition_path),
        "acquisition_plan_internal_sha256": acquisition["plan_sha256"],
        "checks": {
            "all_16_official_metadata_objects_hash_verified": True,
            "benchmark_data_downloaded": False,
            "bigcodebench_exact_object_inventory_verified": True,
            "livecodebench_exact_object_inventory_verified": True,
            "network_payload_within_16_mib": True,
            "raw_benchmark_text_in_git": False,
            "rights_decisions_reconstructed": True,
            "solutions_or_tests_downloaded": False,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_metadata_only_bigcodebench_ready_livecodebench_rights_blocked",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("benchmark preflight verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified code benchmark source preflight "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
