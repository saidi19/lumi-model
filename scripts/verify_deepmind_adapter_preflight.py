#!/usr/bin/env python3
"""Independently verify the no-download DeepMind adapter preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_deepmind_adapter_preflight import (
    DEEPMIND_OBJECTS,
    NEXT_AUTHORIZATION_STATEMENT,
    VERIFICATION_SCHEMA_VERSION,
    build_preflight_report,
    file_sha256,
    load_adapter_preflight_plan,
    load_bound_evidence,
    sha256_value,
    validate_preflight_report,
    validate_source_inventory,
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind verification: {path}")
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
    plan = load_adapter_preflight_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = _load_json(report_path)
    validate_preflight_report(report)
    reconstructed = build_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        bound_evidence=load_bound_evidence(plan, repository_root=root),
    )
    if reconstructed != report:
        raise ValueError("DeepMind adapter preflight reconstruction changed")
    inventory = validate_source_inventory(DEEPMIND_OBJECTS)
    if report.get("inventory") != inventory:
        raise ValueError("DeepMind adapter preflight inventory changed")
    if report.get("next_gate") != plan["next_gate"]:
        raise ValueError("DeepMind adapter next gate changed")
    if sha256_value(NEXT_AUTHORIZATION_STATEMENT) != sha256_value(
        plan["next_gate"]["exact_authorization_statement"]
    ):
        raise ValueError("DeepMind exact next authorization changed")
    verification = {
        "checks": {
            "adapter_contract_bound": True,
            "dmath_owner_review_still_pending": True,
            "exact_source_inventory_verified": True,
            "generated_records": 0,
            "network_bytes": 0,
            "next_gate_is_separately_authorized": True,
            "official_test_or_evaluation_generation_authorized": False,
            "sampled_train_validation_seed_overlap": 0,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_local_no_download_adapter_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if not verification_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind verification must remain under evidence")
    _write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified DeepMind adapter preflight "
        f"generated_records=0 network_bytes=0 "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
