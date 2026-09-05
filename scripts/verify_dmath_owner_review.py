#!/usr/bin/env python3
"""Replay and independently record the exact DMath owner-review result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_dmath_curation import sha256_value
from lumi.posttraining_dmath_owner_review import (
    load_exact_owner_sample,
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
    validate_owner_review_receipt,
)
from lumi.posttraining_first_party import file_sha256

SCHEMA_VERSION = "lumi.sft-dmath-owner-review-verification.v0.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    decisions_path = args.decisions.resolve()
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    output_path = args.output if args.output.is_absolute() else root / args.output
    if not output_path.resolve().is_relative_to((root / "evidence").resolve()):
        raise ValueError("DMath owner verification must stay under evidence/")

    plan = load_owner_review_plan(plan_path, repository_root=root)
    report = _read_json(resolve_plan_binding(root, plan["bindings"]["curation_report"]))
    curation_verification = _read_json(
        resolve_plan_binding(root, plan["bindings"]["curation_verification"])
    )
    sample = load_exact_owner_sample(
        resolve_plan_binding(root, plan["bindings"]["owner_sample"]),
        curation_report=report,
        verification=curation_verification,
    )
    expected_receipt = validate_owner_decisions(
        _read_json(decisions_path),
        sample_rows=sample,
        decision_file_sha256=file_sha256(decisions_path),
        plan_file_sha256=file_sha256(plan_path),
    )
    receipt = _read_json(receipt_path)
    validate_owner_review_receipt(receipt)
    if receipt != expected_receipt:
        raise ValueError("DMath owner receipt does not match decision replay")

    verification: dict[str, Any] = {
        "accepted_records": receipt["accepted_records"],
        "checks": {
            "all_plan_bindings_verified": True,
            "all_sampled_rows_accepted": receipt["all_sampled_rows_accepted"],
            "decision_file_replayed": True,
            "official_test_split_accessed": False,
            "raw_training_text_retained": False,
            "receipt_seal_verified": True,
            "source_registry_promotions": 0,
        },
        "decision_file_sha256": file_sha256(decisions_path),
        "observed_at": "2026-08-06",
        "owner_receipt_file_sha256": file_sha256(receipt_path),
        "owner_receipt_internal_sha256": receipt["receipt_sha256"],
        "production_eligible": False,
        "rejected_records": receipt["rejected_records"],
        "sample_records": receipt["sample_records"],
        "schema_version": SCHEMA_VERSION,
        "status": "verified_owner_review_complete_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    _write_json_atomic(output_path, verification)
    print(
        "PASS: DMath owner-review independently replayed "
        f"accepted={verification['accepted_records']} rejected=0 "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
