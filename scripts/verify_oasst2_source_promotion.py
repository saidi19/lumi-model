#!/usr/bin/env python3
"""Independently reconstruct the OASST2 registry v0.13 promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_oasst2_curation import sha256_value
from lumi.posttraining_oasst2_source_promotion import (
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_SHA256,
    VERIFICATION_SCHEMA_VERSION,
    build_and_validate_successor,
    build_promotion_receipt,
    json_file_bytes,
    load_promotion_plan,
    resolve_binding,
    validate_promotion_receipt,
    verify_promotion_prerequisites,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 verification: {path}")
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
    plan_path = plan_path.resolve()
    plan = load_promotion_plan(plan_path, repository_root=root)
    _, preflight_report, predecessor = verify_promotion_prerequisites(
        plan=plan, repository_root=root
    )
    policy = _load(resolve_binding(root, plan["bindings"]["data_contract"], name="data_contract"))
    expected_successor = build_and_validate_successor(
        plan=plan, predecessor=predecessor, policy=policy
    )
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    observed_successor = _load(successor_path)
    if observed_successor != expected_successor:
        raise ValueError("OASST2 successor registry does not reproduce")
    if (
        sha256_file(successor_path) != SUCCESSOR_REGISTRY_SHA256
        or successor_path.stat().st_size != SUCCESSOR_REGISTRY_BYTES
    ):
        raise ValueError("OASST2 successor file identity changed")

    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    observed_receipt = _load(receipt_path)
    validate_promotion_receipt(observed_receipt)
    expected_receipt = build_promotion_receipt(
        plan=plan,
        plan_file_sha256=sha256_file(plan_path),
        preflight_report=preflight_report,
    )
    if observed_receipt != expected_receipt:
        raise ValueError("OASST2 promotion receipt does not reproduce")

    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    contract_path = resolve_binding(root, plan["bindings"]["data_contract"], name="data_contract")
    verification = {
        "checks": {
            "active_data_contract_unchanged": True,
            "added_source_ids": ["openassistant_oasst2"],
            "predecessor_registry_unchanged": True,
            "promotion_receipt_reproduced": True,
            "raw_training_text_retained": False,
            "successor_registry_reproduced": True,
        },
        "observed_at": "2026-08-08",
        "predecessor_registry_file_sha256": sha256_file(predecessor_path),
        "production_eligible": False,
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_internal_sha256": observed_receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_oasst2_registry_v0_13_nonproduction",
        "successor_registry_bytes": successor_path.stat().st_size,
        "successor_registry_file_sha256": sha256_file(successor_path),
        "unchanged_data_contract_file_sha256": sha256_file(contract_path),
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("OASST2 verification escaped evidence/")
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: OASST2 registry v0.13 independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
