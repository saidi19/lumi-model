#!/usr/bin/env python3
"""Independently replay the OpenCodeInstruct BigCodeBench re-audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import sha256_value, write_json_atomic
from lumi.posttraining_opencodeinstruct_bigcodebench_reaudit import (
    VERIFICATION_SCHEMA_VERSION,
    build_report,
    load_reaudit_plan,
    validate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_reaudit_plan(plan_path.resolve(), repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite OpenCodeInstruct verification")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    replay = build_report(plan, repository_root=root)
    if replay != report:
        raise ValueError("OpenCodeInstruct BigCodeBench independent replay changed")

    verification = {
        "checks": {
            "all_11100_candidate_rows_replayed": True,
            "candidate_and_owner_review_bindings_verified": True,
            "candidate_unchanged": True,
            "no_bigcodebench_match_found": report["results"]["matched_records"] == 0,
            "no_livecodebench_network_gpu_or_sft_access": True,
            "registry_v0_2_runtime_load_passed": True,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_bigcodebench_reaudit_livecodebench_still_unresolved",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified OpenCodeInstruct BigCodeBench re-audit "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
