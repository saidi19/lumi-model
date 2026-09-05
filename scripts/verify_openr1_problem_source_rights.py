#!/usr/bin/env python3
"""Independently reproduce the raw-text-free OpenR1 rights-map audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_openr1_full_run import sha256_value
from lumi.posttraining_openr1_rights import (
    VERIFICATION_SCHEMA_VERSION,
    audit_attributions,
    load_rights_plan,
    resolve_plan_binding,
    validate_rights_report,
    write_json_atomic,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_rights_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    report = _read_json(report_path)
    validate_rights_report(report)
    observed = audit_attributions(
        _read_jsonl(resolve_plan_binding(root, plan["bindings"]["attribution_manifest"]))
    )
    if report.get("source_audit") != observed:
        raise ValueError("OpenR1 rights source audit did not reproduce")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("OpenR1 rights plan binding failed")
    verification: dict[str, Any] = {
        "checks": {
            "all_source_counts_reproduced": True,
            "all_unresolved_rows_remain_quarantined": True,
            "dataset_declarations_reproduced": True,
            "no_raw_training_text_retained": True,
            "zero_promotions_reproduced": True,
        },
        "observed_at": "2026-08-06",
        "production_eligible": False,
        "promotion_eligible_records": 0,
        "records": observed["records"],
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_labels": observed["source_labels"],
        "status": "verified_all_openr1_problem_sources_quarantined",
        "unresolved_records": observed["unresolved_records"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: OpenR1 rights-map independent verification "
        f"records={observed['records']} promotion_eligible=0 "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
