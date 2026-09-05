#!/usr/bin/env python3
"""Independently reproduce the no-download math replacement preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_math_replacement_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    file_sha256,
    json_file_bytes,
    load_math_replacement_plan,
    rebuild_math_replacement_preflight,
    sha256_value,
    validate_math_replacement_report,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite math preflight verification: {path}")
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
    plan = load_math_replacement_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["preflight_report_path"]).resolve()
    observed = _load_json(report_path)
    validate_math_replacement_report(observed)
    rebuilt = rebuild_math_replacement_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if rebuilt != observed:
        raise ValueError("math replacement preflight did not reproduce")
    verification = {
        "checks": {
            "all_binding_hashes_passed": True,
            "dmath_exact_boundary_reproduced": True,
            "deepmind_fallback_boundary_reproduced": True,
            "frozen_token_gap_reproduced": True,
            "no_acquisition_or_generation_authorized": True,
            "no_raw_training_text_retained": True,
            "openr1_zero_rights_clear_records_reproduced": True,
            "production_promotions_zero": True,
        },
        "observed_at": "2026-08-06",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": observed["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_math_replacement_preflight_dmath_is_next_gate",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("math replacement verification must remain under evidence/")
    _write_bytes_atomic(output, json_file_bytes(verification))
    print(
        "PASS: math replacement preflight independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
