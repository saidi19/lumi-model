#!/usr/bin/env python3
"""Independently verify the exact DeepMind dependency overlay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_dependency_overlay import (
    EXPECTED_INSTALL_POLICY,
    VERIFICATION_SCHEMA_VERSION,
    WHEEL_BYTES,
    build_overlay_receipt,
    file_sha256,
    load_overlay_plan,
    overlay_inventory,
    run_import_smoke,
    sha256_value,
    validate_overlay_receipt,
    validate_wheelhouse,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite overlay verification: {path}")
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
    plan = load_overlay_plan(plan_path, repository_root=root)
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    receipt = _read_json(receipt_path)
    validate_overlay_receipt(receipt)
    wheel_inventory = validate_wheelhouse(plan["output"]["wheelhouse_path"])
    inventory = overlay_inventory(plan["output"]["overlay_path"])
    smoke = run_import_smoke(
        overlay_path=Path(plan["output"]["overlay_path"]),
        python_executable=sys.executable,
    )
    replay_command = [*EXPECTED_INSTALL_POLICY]
    replayed = build_overlay_receipt(
        command=replay_command,
        inventory=inventory,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        smoke=smoke,
        wheel_inventory=wheel_inventory,
    )
    if receipt != replayed:
        raise ValueError("DeepMind overlay receipt did not replay exactly")
    verification: dict[str, Any] = {
        "checks": {
            "canary_records_generated": 0,
            "exact_distribution_set_verified": True,
            "exact_wheel_bytes_verified": WHEEL_BYTES,
            "fresh_process_import_smoke_passed": True,
            "overlay_inventory_replayed": True,
            "paths_within_overlay": True,
            "production_sft_authorized": False,
            "receipt_reconstructed": True,
        },
        "observed_at": plan["observed_at"],
        "overlay_inventory": inventory,
        "production_eligible": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_exact_deepmind_dependency_overlay",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind overlay verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified DeepMind dependency overlay; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
