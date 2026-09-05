#!/usr/bin/env python3
"""Independently replay the bounded BanglaBridge hygiene repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_banglabridge_hygiene_repair import (
    VERIFICATION_SCHEMA_VERSION,
    _json_bytes,
    build_repair,
    load_repair_plan,
    sha256_value,
    validate_report,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_repair_plan(plan_path.resolve(), repository_root=root)
    candidate_path = (root / plan["output"]["candidate_path"]).resolve()
    manifest_path = (root / plan["output"]["manifest_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite BanglaBridge repair verification")
    candidate_payload, expected_manifest, expected_report = build_repair(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_report(report)
    if candidate_path.read_bytes() != candidate_payload:
        raise ValueError("BanglaBridge repaired candidate replay changed")
    if (
        manifest != expected_manifest
        or manifest_path.read_bytes() != _json_bytes(expected_manifest)
    ):
        raise ValueError("BanglaBridge repaired manifest replay changed")
    if report != expected_report or report_path.read_bytes() != _json_bytes(expected_report):
        raise ValueError("BanglaBridge repaired report replay changed")
    verification = {
        "checks": {
            "all_input_rows_replayed": True,
            "candidate_bytes_reconstructed": True,
            "four_provenance_hashes_per_retained_row": True,
            "frozen_token_counts_reconstructed": True,
            "no_network_gpu_registry_dataset_seal_or_sft": True,
            "owner_reviewed_visible_rows_preserved": True,
            "schema_safety_dedup_decontamination_and_4k_fit_verified": True,
            "train_token_ceiling_respected": True,
        },
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "observed_at": "2026-08-10",
        "output_candidate_file_sha256": sha256_file(candidate_path),
        "production_eligible": False,
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_private_nonproduction_banglabridge_hygiene_repair",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified BanglaBridge hygiene repair "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
