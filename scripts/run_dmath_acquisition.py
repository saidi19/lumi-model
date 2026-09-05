#!/usr/bin/env python3
"""Fetch exactly the three owner-authorized pinned DMath objects."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lumi.posttraining_dmath_acquisition import (
    EXACT_OBJECTS,
    REPORT_SCHEMA_VERSION,
    download_exact_object,
    file_sha256,
    json_file_bytes,
    load_dmath_acquisition_plan,
    sha256_value,
    summarize_train_payload,
    validate_dmath_acquisition_report,
    write_bytes_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_dmath_acquisition_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    if not private_root.is_relative_to(root.parent / "lumi-artifacts"):
        raise ValueError("DMath private root must remain outside Git under lumi-artifacts")
    raw_root = private_root / "raw"
    if raw_root.exists() and any(raw_root.rglob("*")):
        raise FileExistsError("refusing nonempty DMath raw root")

    inventory = []
    payload_by_path = {}
    network_bytes = 0
    for specification in EXACT_OBJECTS:
        remaining = plan["limits"]["maximum_network_bytes"] - network_bytes
        payload, observed = download_exact_object(
            specification,
            maximum_network_bytes=remaining,
        )
        network_bytes += len(payload)
        payload_by_path[specification["path"]] = payload
        inventory.append(observed)
    if network_bytes != plan["limits"]["expected_network_bytes"]:
        raise ValueError("DMath exact network byte inventory changed")
    if len(inventory) != plan["limits"]["expected_objects"]:
        raise ValueError("DMath exact object count changed")

    for specification in EXACT_OBJECTS:
        write_bytes_atomic(raw_root / specification["path"], payload_by_path[specification["path"]])
    train_summary = summarize_train_payload(payload_by_path["dataset/train.json"])

    runtime_seconds = round(time.monotonic() - started, 3)
    if runtime_seconds > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("DMath acquisition exceeded the authorized runtime")
    report = {
        "authorization": dict(plan["authorization"]),
        "inventory": inventory,
        "network_bytes": network_bytes,
        "observed_at": plan["observed_at"],
        "owner_authorization_sha256": sha256_value(plan["owner_authorization_statement"]),
        "retry_authorization_sha256": sha256_value(plan["retry_authorization_statement"]),
        "plan_file_sha256": file_sha256(plan_path),
        "private_artifacts": {
            "object_count": len(inventory),
            "raw_root": plan["output"]["private_root"] + "/raw",
            "raw_text_present": True,
        },
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_sealed": False,
            "production_dataset_promotions": 0,
            "production_eligible": False,
            "production_sft_authorized": False,
        },
        "raw_training_text_retained_in_report": False,
        "runtime_seconds": runtime_seconds,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "declared_license_id": plan["source"]["declared_license_id"],
            "repository_url": plan["source"]["repository_url"],
            "revision": plan["source"]["revision"],
            "source_id": plan["source"]["source_id"],
            "split": "train",
        },
        "status": "exact_dmath_source_acquired_private_nonproduction",
        "test_split_accessed": False,
        "train_summary": train_summary,
    }
    report["report_sha256"] = sha256_value(report)
    validate_dmath_acquisition_report(report)
    report_path = (root / plan["output"]["report_path"]).resolve()
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("DMath acquisition report must remain under evidence/")
    write_bytes_atomic(report_path, json_file_bytes(report))
    print(
        "PASS: exact DMath source acquired; "
        f"objects={len(inventory)} bytes={network_bytes} rows={train_summary['rows']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
