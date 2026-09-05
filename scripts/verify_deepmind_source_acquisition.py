#!/usr/bin/env python3
"""Independently verify the exact DeepMind source acquisition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_source_acquisition import (
    VERIFICATION_SCHEMA_VERSION,
    build_acquisition_receipt,
    file_sha256,
    inspect_source_root,
    load_acquisition_plan,
    sha256_value,
    validate_acquisition_receipt,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acquisition verification: {path}")
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
    plan = load_acquisition_plan(plan_path, repository_root=root)
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    receipt = _read_json(receipt_path)
    validate_acquisition_receipt(receipt)
    acquisition = inspect_source_root(plan["output"]["source_root"])
    replayed = build_acquisition_receipt(
        acquisition=acquisition,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    if receipt != replayed:
        raise ValueError("DeepMind source-acquisition receipt did not replay exactly")
    verification: dict[str, Any] = {
        "checks": {
            "canary_records_generated": 0,
            "exact_file_set_verified": True,
            "exact_git_blob_sha1_verified": True,
            "exact_source_bytes_verified": acquisition["network_payload_bytes"],
            "exact_source_objects_verified": acquisition["objects"],
            "production_sft_authorized": False,
            "receipt_reconstructed": True,
            "source_registry_promotions": 0,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_inventory_sha256": acquisition["inventory_sha256"],
        "status": "verified_exact_deepmind_source_acquisition_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind acquisition verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified exact DeepMind source; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
