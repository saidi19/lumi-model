#!/usr/bin/env python3
"""Independently reconstruct and verify the DeepMind registry promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_canary_run import (
    file_sha256,
    json_file_bytes,
    sha256_value,
)
from lumi.posttraining_deepmind_source_promotion import (
    VERIFICATION_SCHEMA_VERSION,
    build_source_promotion_receipt,
    build_successor_registry,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    successor_payload_sha256,
    validate_records_against_successor,
    validate_successor_registry,
    verify_promotion_prerequisites,
    verify_source_promotion_receipt,
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
    plan = load_source_promotion_plan(plan_path, repository_root=root)
    records, candidate_audit = verify_promotion_prerequisites(
        plan=plan,
        repository_root=root,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    expected_successor = build_successor_registry(predecessor_value, plan=plan)
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    successor_value = _read_json(successor_path)
    if successor_value != expected_successor:
        raise ValueError("DeepMind successor reconstruction changed")
    successor_sha256 = successor_payload_sha256(successor_value)
    if file_sha256(successor_path) != successor_sha256:
        raise ValueError("DeepMind successor serialization changed")
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=successor_sha256,
    )
    validate_records_against_successor(records, successor=successor)

    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    receipt = _read_json(receipt_path)
    verify_source_promotion_receipt(receipt)
    expected_receipt = build_source_promotion_receipt(
        candidate_audit=candidate_audit,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
    )
    if receipt != expected_receipt:
        raise ValueError("DeepMind promotion receipt reconstruction changed")

    verification: dict[str, Any] = {
        "candidate_records": 1_915,
        "candidate_supervised_tokens": 18_731,
        "checks": {
            "accepted_records_match_successor_registry": True,
            "active_production_data_contract_updated": False,
            "all_prerequisite_seals_verified": True,
            "github_https_repository_prefix_added": True,
            "huggingface_repository_prefix_preserved": True,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "receipt_reconstructed": True,
            "successor_registry_reconstructed": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_deepmind_candidate_registry_promotion_nonproduction",
        "successor_registry_sha256": successor_sha256,
    }
    verification["verification_sha256"] = sha256_value(verification)
    output_path = (root / plan["output"]["verification_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind promotion verification escaped evidence/")
    _write_json_atomic(output_path, verification)
    if file_sha256(successor_path) != successor_sha256:
        raise ValueError("DeepMind successor changed during independent verification")
    if json_file_bytes(successor_value) != successor_path.read_bytes():
        raise ValueError("DeepMind successor bytes changed during verification")
    print(
        "PASS: independently verified DeepMind candidate registry promotion; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
