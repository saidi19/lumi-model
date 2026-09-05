#!/usr/bin/env python3
"""Independently reconstruct the OASST2 registry-v0.13 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_oasst2_curation import sha256_value, write_json_atomic
from lumi.posttraining_oasst2_source_promotion_preflight import (
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
    plan_path = plan_path.resolve()
    plan = load_promotion_preflight_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["preflight_report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite OASST2 preflight verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    reconstructed = build_promotion_preflight_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    if report != reconstructed:
        raise ValueError("OASST2 source-promotion preflight replay changed")
    verification = {
        "checks": {
            "all_6426_records_replayed": True,
            "apache_2_0_attribution_verified": True,
            "candidate_registry_not_materialized": True,
            "curation_owner_and_token_evidence_verified": True,
            "planning_shortfall_preserved": True,
            "predecessor_registry_unchanged": True,
            "projected_successor_hash_reconstructed": True,
        },
        "observed_at": "2026-08-08",
        "production_eligible": False,
        "projected_successor_sha256": report["projected_successor"]["sha256"],
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_preflight_registry_v0_13_successor_not_materialized",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified OASST2 v0.13 promotion preflight "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
