#!/usr/bin/env python3
"""Independently reconstruct the OpenCodeInstruct v0.11 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import sha256_value, write_json_atomic
from lumi.posttraining_opencodeinstruct_source_promotion_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    build_promotion_preflight_report,
    load_promotion_preflight_plan,
    validate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_promotion_preflight_plan(plan_path.resolve(), repository_root=root)
    report_path = (root / plan["output"]["preflight_report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite OpenCodeInstruct preflight verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    reconstructed = build_promotion_preflight_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    if report != reconstructed:
        raise ValueError("OpenCodeInstruct source-promotion preflight replay changed")
    verification = {
        "checks": {
            "all_11100_records_replayed": True,
            "attribution_and_cc_by_4_0_binding_verified": True,
            "benchmark_zero_match_evidence_verified": True,
            "candidate_registry_not_materialized": True,
            "owner_review_and_token_evidence_verified": True,
            "predecessor_registry_unchanged": True,
            "projected_successor_hash_reconstructed": True,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "projected_successor_sha256": report["projected_successor"]["sha256"],
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_preflight_source_registry_successor_not_materialized",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified OpenCodeInstruct v0.11 promotion preflight "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
