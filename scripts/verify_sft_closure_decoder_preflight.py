#!/usr/bin/env python3
"""Independently verify the metadata-only closure decoder preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_METADATA_SHA256 = (
    "952cdc36d7d21120c401fed07931c74e1b85732b43ff7f56861c687a4c531867"
)
EXPECTED_METADATA_BYTES = 40_590
EXPECTED_WHEEL = "pyarrow-25.0.0-cp313-cp313-macosx_12_0_arm64.whl"
EXPECTED_WHEEL_BYTES = 35_844_549
EXPECTED_WHEEL_SHA256 = (
    "8831a3ba52fa7cdb78d368d968b1dcd06171e6dff5461e16d90de91d371e47bc"
)
SCHEMA_VERSION = "lumi.sft-closure-parquet-decoder-preflight-verification.v0.1"


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
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.metadata.stat().st_size != EXPECTED_METADATA_BYTES:
        raise ValueError("PyArrow metadata size changed")
    if file_sha256(args.metadata) != EXPECTED_METADATA_SHA256:
        raise ValueError("PyArrow metadata hash changed")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["info"].get("requires_dist") not in (None, []):
        raise ValueError("PyArrow dependencies changed")
    wheels = [
        item for item in metadata["urls"] if item.get("filename") == EXPECTED_WHEEL
    ]
    if len(wheels) != 1:
        raise ValueError("exact PyArrow wheel selection changed")
    wheel = wheels[0]
    if (
        wheel.get("size") != EXPECTED_WHEEL_BYTES
        or wheel.get("digests", {}).get("sha256") != EXPECTED_WHEEL_SHA256
    ):
        raise ValueError("exact PyArrow wheel binding changed")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_hash = plan.pop("plan_sha256")
    if sha256_value(plan) != plan_hash:
        raise ValueError("overlay plan internal hash changed")
    plan["plan_sha256"] = plan_hash
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_hash = report.pop("report_sha256")
    if sha256_value(report) != report_hash:
        raise ValueError("preflight report internal hash changed")
    report["report_sha256"] = report_hash
    wheel_matches = list(args.artifact_root.rglob(EXPECTED_WHEEL))
    overlay = Path(plan["runtime"]["overlay_root"])
    if wheel_matches or overlay.exists():
        raise ValueError("wheel download or overlay installation occurred")
    total_private_bytes = sum(
        path.stat().st_size for path in args.artifact_root.rglob("*") if path.is_file()
    )
    if total_private_bytes > 16 * 1024 * 1024:
        raise ValueError("metadata preflight private write limit exceeded")
    verification: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_exact_overlay_plan_ready_authorization_required",
        "checks": {
            "declared_runtime_dependencies": [],
            "metadata_file_sha256": EXPECTED_METADATA_SHA256,
            "metadata_objects_fetched": 1,
            "overlay_plan_file_sha256": file_sha256(args.plan),
            "overlay_plan_internal_sha256": plan_hash,
            "preflight_report_file_sha256": file_sha256(args.report),
            "preflight_report_internal_sha256": report_hash,
            "private_bytes": total_private_bytes,
            "wheel_downloaded": False,
            "wheel_installed": False,
            "wheel_sha256": EXPECTED_WHEEL_SHA256,
        },
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
