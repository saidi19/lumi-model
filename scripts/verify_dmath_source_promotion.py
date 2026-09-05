#!/usr/bin/env python3
"""Independently replay and verify the DMath registry promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_dmath_source_promotion import (
    VERIFICATION_SCHEMA_VERSION,
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    resolve_promotion_binding,
    sha256_value,
    validate_records_against_successor,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)
from lumi.posttraining_first_party import file_sha256


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DMath verification: {path}")
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
    bindings = plan["bindings"]
    records = _read_jsonl(
        resolve_promotion_binding(root, bindings["accepted_records"])
    )
    report = _read_json(resolve_promotion_binding(root, bindings["curation_report"]))
    curation_verification = _read_json(
        resolve_promotion_binding(root, bindings["curation_verification"])
    )
    owner_receipt = _read_json(
        resolve_promotion_binding(root, bindings["owner_review_receipt"])
    )
    owner_verification = _read_json(
        resolve_promotion_binding(root, bindings["owner_review_verification"])
    )
    verify_source_promotion_prerequisites(
        accepted_records=records,
        curation_report=report,
        curation_verification=curation_verification,
        owner_review_receipt=owner_receipt,
        owner_review_verification=owner_verification,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    expected_successor = build_successor_registry(predecessor_value, plan=plan)
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    successor_value = _read_json(successor_path)
    if successor_value != expected_successor:
        raise ValueError("DMath successor registry reconstruction changed")
    successor_sha256 = hashlib.sha256(json_file_bytes(successor_value)).hexdigest()
    if file_sha256(successor_path) != successor_sha256:
        raise ValueError("DMath successor registry serialization changed")
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
        owner_review_receipt=owner_receipt,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
    )
    if receipt != expected_receipt:
        raise ValueError("DMath promotion receipt reconstruction changed")

    verification: dict[str, Any] = {
        "checks": {
            "accepted_records_match_successor_registry": True,
            "active_production_data_contract_updated": False,
            "all_prerequisite_seals_verified": True,
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
        "status": "verified_dmath_candidate_registry_promotion_nonproduction",
        "successor_registry_sha256": successor_sha256,
    }
    verification["verification_sha256"] = sha256_value(verification)
    output_path = (root / plan["output"]["verification_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DMath promotion verification must stay under evidence/")
    _write_json_atomic(output_path, verification)
    print(
        "PASS: independently verified DMath candidate registry promotion; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
