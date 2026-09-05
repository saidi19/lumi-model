#!/usr/bin/env python3
"""Independently verify the Lumi SFT closure-canary owner spot-check."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_closure_canary_curation import file_sha256, sha256_value
from lumi.posttraining_owner_review import read_json, validate_owner_decisions

SCHEMA_VERSION = "lumi.sft-closure-canary-owner-review-verification.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--evidence-receipt", type=Path, required=True)
    parser.add_argument("--curation-report", type=Path, required=True)
    parser.add_argument("--curation-verification", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _verify_internal(value: Mapping[str, Any], field: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != sha256_value(unsigned):
        raise ValueError(f"{field} mismatch")


def _load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("owner-review sample must be a JSON array of objects")
    return value


def main() -> int:
    args = parse_args()
    decisions = read_json(args.decisions)
    sample = _load_array(args.sample)
    manifest = read_json(args.private_manifest)
    sample_sha256 = manifest["owner_review"]["sample_sha256"]
    reconstructed = validate_owner_decisions(
        decisions,
        sample_rows=sample,
        sample_sha256=sample_sha256,
    )
    private_receipt = read_json(args.private_receipt)
    evidence_receipt = read_json(args.evidence_receipt)
    if reconstructed != private_receipt or reconstructed != evidence_receipt:
        raise ValueError("owner-review receipt reconstruction changed")
    if reconstructed["sample_records"] != 32:
        raise ValueError("owner-review sample count changed")
    if reconstructed["accepted_records"] != 32 or reconstructed["rejected_records"] != 0:
        raise ValueError("owner-review decisions did not pass")
    if reconstructed["production_eligible"] is not False:
        raise ValueError("spot-check cannot grant production eligibility")

    _verify_internal(manifest, "manifest_sha256")
    curation = read_json(args.curation_report)
    _verify_internal(curation, "report_sha256")
    curation_verification = read_json(args.curation_verification)
    _verify_internal(curation_verification, "verification_sha256")
    if curation["bindings"]["private_manifest_file_sha256"] != file_sha256(
        args.private_manifest
    ):
        raise ValueError("curation manifest binding changed")
    if curation_verification["bindings"]["curation_report_file_sha256"] != file_sha256(
        args.curation_report
    ):
        raise ValueError("curation verification binding changed")

    verification: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_owner_spot_check_passed_not_production_eligible",
        "bindings": {
            "curation_report_file_sha256": file_sha256(args.curation_report),
            "curation_report_internal_sha256": curation["report_sha256"],
            "curation_verification_file_sha256": file_sha256(
                args.curation_verification
            ),
            "curation_verification_internal_sha256": curation_verification[
                "verification_sha256"
            ],
            "decision_file_sha256": file_sha256(args.decisions),
            "evidence_receipt_file_sha256": file_sha256(args.evidence_receipt),
            "owner_receipt_internal_sha256": reconstructed["receipt_sha256"],
            "private_manifest_file_sha256": file_sha256(args.private_manifest),
            "private_manifest_internal_sha256": manifest["manifest_sha256"],
            "private_receipt_file_sha256": file_sha256(args.private_receipt),
            "sample_file_sha256": file_sha256(args.sample),
            "sample_internal_sha256": sample_sha256,
        },
        "checks": {
            "accepted_records": reconstructed["accepted_records"],
            "all_sampled_rows_accepted": reconstructed["all_sampled_rows_accepted"],
            "decision_ids_unique_and_complete": True,
            "production_eligible": False,
            "rejected_records": reconstructed["rejected_records"],
            "sample_records": reconstructed["sample_records"],
        },
        "reviewer_id_sha256": hashlib.sha256(
            reconstructed["reviewer_id"].encode()
        ).hexdigest(),
        "raw_training_text_in_verification": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(verification["status"])
    print(verification["verification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
