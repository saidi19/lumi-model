#!/usr/bin/env python3
"""Independently re-hash the Lumi closure-canary acquisition receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_PLAN_SHA256 = (
    "31568114288e46c1c8613e20d9167af952050e5ed2a2556a64722bcec9da488a"
)
EXPECTED_OBJECTS = 9
EXPECTED_OBJECT_BYTES = 840_904_222
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-closure-canary-acquisition-verification.v0.1"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if file_sha256(args.plan) != EXPECTED_PLAN_SHA256:
        raise ValueError("closure-canary plan changed during independent verification")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_digest = report.pop("report_sha256")
    if sha256_value(report) != report_digest:
        raise ValueError("closure-canary report internal hash changed")
    report["report_sha256"] = report_digest
    objects = report.get("objects")
    if not isinstance(objects, list) or len(objects) != EXPECTED_OBJECTS:
        raise ValueError("closure-canary report object count changed")
    if sum(int(item["bytes"]) for item in objects) != EXPECTED_OBJECT_BYTES:
        raise ValueError("closure-canary report object bytes changed")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    directory_by_repository = {
        "nvidia/Nemotron-Post-Training-Dataset-v1": (
            "nvidia_Nemotron-Post-Training-Dataset-v1"
        ),
        "CohereLabs/aya_collection_language_split": (
            "CohereLabs_aya_collection_language_split"
        ),
        "aplycaebous/BanglaTLit": "aplycaebous_BanglaTLit",
        "subhajitmahata84/banglabridge-instructions": (
            "subhajitmahata84_banglabridge-instructions"
        ),
    }
    report_by_identity = {
        (item["repository"], item["revision"], item["path"]): item
        for item in objects
    }
    checks = []
    for expected in plan["objects"]:
        identity = (
            expected["repository"],
            expected["revision"],
            expected["path"],
        )
        actual = report_by_identity.get(identity)
        if actual is None:
            raise ValueError("closure-canary report object identity changed")
        path = (
            args.private_root
            / "raw"
            / directory_by_repository[expected["repository"]]
            / expected["path"]
        )
        if path.stat().st_size != expected["bytes"]:
            raise ValueError("closure-canary independent size verification failed")
        digest = file_sha256(path)
        if digest != actual["sha256"]:
            raise ValueError("closure-canary independent SHA-256 verification failed")
        checks.append(
            {
                "bytes": path.stat().st_size,
                "identity_sha256": sha256_value(identity),
                "sha256": digest,
            }
        )
    if report.get("status") != "acquisition_verified_curation_blocked":
        raise ValueError("closure-canary fail-closed status changed")
    blockers = report.get("blockers")
    if blockers != ["missing_local_parquet_decoder_no_dependency_download_authorized"]:
        raise ValueError("closure-canary blocker set changed")
    curation = report.get("curation")
    if curation != {
        "candidate_records_written": 0,
        "dataset_sealed": False,
        "owner_review_packets_written": 0,
        "sft_started": False,
    }:
        raise ValueError("closure-canary curation preservation changed")
    if not all(report.get("prohibitions_preserved", {}).values()):
        raise ValueError("closure-canary prohibition preservation changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_acquisition_complete_curation_blocked",
        "checks": {
            "all_object_hashes_replayed": True,
            "all_object_sizes_replayed": True,
            "candidate_records_written": 0,
            "exact_object_bytes": EXPECTED_OBJECT_BYTES,
            "exact_objects": EXPECTED_OBJECTS,
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "prohibitions_preserved": True,
            "report_file_sha256": file_sha256(args.report),
            "report_internal_sha256": report_digest,
        },
        "object_checks_sha256": sha256_value(checks),
    }
    verification["verification_sha256"] = sha256_value(verification)
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
