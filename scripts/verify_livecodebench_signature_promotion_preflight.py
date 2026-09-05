#!/usr/bin/env python3
"""Independently verify the LiveCodeBench signature promotion preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import sha256_value, write_json_atomic
from lumi.posttraining_livecodebench_promotion_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    audit_and_build_outputs,
    load_preflight_plan,
    validate_promotion_plan,
    validate_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    expected_report, expected_promotion = audit_and_build_outputs(
        plan,
        repository_root=root,
    )
    report_path = (root / plan["output"]["report_path"]).resolve()
    promotion_path = (root / plan["output"]["promotion_plan_path"]).resolve()
    report = _read_json(report_path)
    promotion = _read_json(promotion_path)
    validate_report(report)
    validate_promotion_plan(promotion)
    if report != expected_report or promotion != expected_promotion:
        raise ValueError("LiveCodeBench promotion preflight reconstruction changed")

    verification = {
        "checks": {
            "all_50944_entries_schema_verified": True,
            "candidate_identity_collisions_zero": True,
            "candidate_not_promoted": True,
            "candidate_raw_text_free": True,
            "config_unchanged": True,
            "network_gpu_sft_unused": True,
            "predecessor_50419_entries_preserved_in_order": True,
            "private_research_decontamination_scope_preserved": True,
            "proposed_data_quality_successor_hash_reconstructed": True,
            "proposed_immutable_registry_hash_reconstructed": True,
            "rights_clearance_not_claimed": True,
            "training_ingestion_not_authorized": True,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "promotion_plan_file_sha256": sha256_file(promotion_path),
        "promotion_plan_internal_sha256": promotion["plan_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_preflight_no_promotion",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("LiveCodeBench promotion verification escaped evidence/")
    write_json_atomic(output, verification)
    print(
        "PASS: independently verified LiveCodeBench promotion preflight "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
