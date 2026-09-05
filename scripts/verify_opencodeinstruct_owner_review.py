#!/usr/bin/env python3
"""Independently verify the OpenCodeInstruct owner-review receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_opencodeinstruct_curation import file_sha256, sha256_value
from lumi.posttraining_opencodeinstruct_owner_review import (
    load_owner_review_plan,
    read_json,
    resolve_binding,
    validate_bound_candidate,
    validate_owner_decisions,
    validate_owner_review_receipt,
)

VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-owner-review-verification.v0.1"
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite OpenCodeInstruct owner verification: {path}"
        )
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
    plan_path = plan_path.resolve()
    plan = load_owner_review_plan(plan_path, repository_root=root)
    sample, _report, _candidate_verification = validate_bound_candidate(
        plan, repository_root=root
    )
    decisions_path = resolve_binding(
        root, plan["bindings"]["decisions"], name="decisions"
    )
    reconstructed = validate_owner_decisions(
        read_json(decisions_path),
        sample_rows=sample,
        plan=plan,
        decision_file_sha256=file_sha256(decisions_path),
        plan_file_sha256=file_sha256(plan_path),
    )
    private_receipt_path = (root / plan["output"]["private_receipt_path"]).resolve()
    repository_receipt_path = (
        root / plan["output"]["repository_receipt_path"]
    ).resolve()
    private_receipt = read_json(private_receipt_path)
    repository_receipt = read_json(repository_receipt_path)
    validate_owner_review_receipt(private_receipt)
    validate_owner_review_receipt(repository_receipt)
    if private_receipt != repository_receipt or repository_receipt != reconstructed:
        raise ValueError("OpenCodeInstruct owner-review receipt reconstruction changed")

    verification: dict[str, Any] = {
        "accepted_records": repository_receipt["accepted_records"],
        "checks": {
            "candidate_bindings_verified": True,
            "decision_ids_exactly_match_owner_sample": True,
            "decision_schema_and_timestamp_verified": True,
            "private_and_repository_receipts_identical": True,
            "raw_training_text_retained": False,
            "receipt_reconstructed_independently": True,
            "sample_records": len(sample),
        },
        "missing_required_benchmark_registries": repository_receipt[
            "missing_required_benchmark_registries"
        ],
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "production_promotions": 0,
        "receipt_file_sha256": file_sha256(repository_receipt_path),
        "receipt_internal_sha256": repository_receipt["receipt_sha256"],
        "rejected_records": repository_receipt["rejected_records"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_owner_review_complete_nonproduction_promotion_separate",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["repository_verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("OpenCodeInstruct verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified OpenCodeInstruct owner review "
        f"accepted={verification['accepted_records']} "
        f"rejected={verification['rejected_records']} "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
