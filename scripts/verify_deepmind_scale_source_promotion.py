#!/usr/bin/env python3
"""Independently verify the immutable DeepMind scale registry promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_deepmind_scale_generation import file_sha256, sha256_value
from lumi.posttraining_deepmind_scale_source_promotion import (
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
        raise FileExistsError(f"refusing to overwrite DeepMind verification: {path}")
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
    plan = load_promotion_plan(plan_path, repository_root=root)
    preflight, owner_receipt, predecessor = verify_promotion_prerequisites(
        plan=plan, repository_root=root
    )
    policy = _load(
        resolve_binding(
            root, plan["bindings"]["data_contract"], name="data_contract"
        )
    )
    expected_successor, source_before = build_and_validate_successor(
        plan=plan, predecessor=predecessor, policy=policy
    )
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    observed_successor = _load(successor_path)
    if observed_successor != expected_successor:
        raise ValueError("DeepMind scale successor registry does not reproduce")
    if file_sha256(successor_path) != SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("DeepMind scale successor file SHA-256 changed")
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    observed_receipt = _load(receipt_path)
    validate_promotion_receipt(observed_receipt)
    expected_receipt = build_promotion_receipt(
        owner_receipt=owner_receipt,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        preflight=preflight,
        source_before=source_before,
    )
    if observed_receipt != expected_receipt:
        raise ValueError("DeepMind scale promotion receipt does not reproduce")
    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    contract_path = resolve_binding(
        root, plan["bindings"]["data_contract"], name="data_contract"
    )
    verification = {
        "checks": {
            "active_data_contract_unchanged": True,
            "changed_source_ids": ["deepmind_mathematics_dataset"],
            "predecessor_registry_unchanged": True,
            "promotion_receipt_reproduced": True,
            "raw_training_text_retained": False,
            "successor_registry_reproduced": True,
        },
        "observed_at": "2026-08-07",
        "predecessor_registry_file_sha256": file_sha256(predecessor_path),
        "production_eligible": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": observed_receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_deepmind_scale_registry_v0_10_nonproduction",
        "successor_registry_file_sha256": file_sha256(successor_path),
        "unchanged_data_contract_file_sha256": file_sha256(contract_path),
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind scale verification escaped evidence/")
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: DeepMind scale registry v0.10 independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
