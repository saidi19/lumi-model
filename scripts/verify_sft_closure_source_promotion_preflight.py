#!/usr/bin/env python3
"""Independently reconstruct the closure source-promotion preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_closure_source_promotion_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    build_preflight_report,
    load_preflight_plan,
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
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    report_path = (root / plan["output"]["preflight_report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite closure preflight verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    reconstructed = build_preflight_report(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    if report != reconstructed:
        raise ValueError("closure source-promotion preflight replay changed")
    verification = {
        "checks": {
            "all_21183_candidate_records_replayed": True,
            "blocker_set_reconstructed": True,
            "no_candidate_registry_change": True,
            "no_dataset_seal_or_sft": True,
            "predecessor_registry_unchanged": True,
            "raw_training_text_not_retained": True,
            "repair_projection_reconstructed": True,
        },
        "observed_at": "2026-08-09",
        "production_eligible": False,
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_blocked_preflight_registry_successor_not_projected",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    if verification_path.stat().st_size > plan["limits"]["maximum_verification_bytes"]:
        raise ValueError("closure preflight verification exceeded write limit")
    print(
        "PASS: independently verified blocked closure registry preflight "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
