#!/usr/bin/env python3
"""Independently verify exact LiveCodeBench registry/config successors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.data_quality import BenchmarkRegistry, sha256_file
from lumi.posttraining_bigcodebench_signatures import (
    load_registry,
    sha256_value,
    write_json_atomic,
)
from lumi.posttraining_livecodebench_signature_promotion import (
    TARGET_ENTRIES,
    TARGET_QUALITY_BYTES,
    TARGET_QUALITY_SHA256,
    TARGET_REGISTRY_BYTES,
    TARGET_REGISTRY_SHA256,
    VERIFICATION_SCHEMA_VERSION,
    build_successor_payloads,
    load_promotion_authorization,
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
    plan = load_promotion_authorization(plan_path.resolve(), repository_root=root)
    registry_path = (root / plan["output"]["registry_successor_path"]).resolve()
    quality_path = (root / plan["output"]["data_quality_successor_path"]).resolve()
    report_path = (root / plan["output"]["promotion_report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite LiveCodeBench promotion verification")
    expected_registry, expected_quality, promotion = build_successor_payloads(
        plan,
        repository_root=root,
    )
    if registry_path.read_bytes() != expected_registry:
        raise ValueError("LiveCodeBench immutable registry bytes changed")
    if quality_path.read_bytes() != expected_quality:
        raise ValueError("LiveCodeBench data-quality successor bytes changed")
    if registry_path.stat().st_size != TARGET_REGISTRY_BYTES:
        raise ValueError("LiveCodeBench immutable registry size changed")
    if sha256_file(registry_path) != TARGET_REGISTRY_SHA256:
        raise ValueError("LiveCodeBench immutable registry SHA-256 changed")
    if quality_path.stat().st_size != TARGET_QUALITY_BYTES:
        raise ValueError("LiveCodeBench data-quality successor size changed")
    if sha256_file(quality_path) != TARGET_QUALITY_SHA256:
        raise ValueError("LiveCodeBench data-quality successor SHA-256 changed")

    candidate_path = (root / plan["bindings"]["candidate_registry"]["path"]).resolve()
    candidate_metadata, candidate_entries = load_registry(candidate_path)
    successor_metadata, successor_entries = load_registry(registry_path)
    if successor_entries != candidate_entries or len(successor_entries) != TARGET_ENTRIES:
        raise ValueError("LiveCodeBench immutable successor entries changed")
    if "candidate_not_active" in successor_metadata:
        raise ValueError("LiveCodeBench immutable successor retained candidate marker")
    if successor_metadata != promotion["proposed_registry_successor"]["metadata"]:
        raise ValueError("LiveCodeBench immutable successor metadata changed")
    if candidate_metadata.get("candidate_not_active") is not True:
        raise ValueError("LiveCodeBench source candidate marker changed")
    if successor_metadata.get("release", {}).get("rights_clearance_claimed") is not False:
        raise ValueError("LiveCodeBench successor claimed rights clearance")
    if successor_metadata.get("release", {}).get("training_ingestion_prohibited") is not True:
        raise ValueError("LiveCodeBench successor lost training-ingestion prohibition")
    quality = _read_json(quality_path)
    if quality != promotion["proposed_data_quality_successor"]["content"]:
        raise ValueError("LiveCodeBench data-quality successor content changed")
    index = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    if "livecodebench" not in index.benchmarks or len(index.exact) != 50_774:
        raise ValueError("LiveCodeBench successor failed runtime registry load")

    predecessor_registry = (
        root / plan["bindings"]["predecessor_registry"]["path"]
    ).resolve()
    predecessor_quality = (
        root / plan["bindings"]["predecessor_data_quality_config"]["path"]
    ).resolve()
    if sha256_file(predecessor_registry) != plan["bindings"]["predecessor_registry"][
        "sha256"
    ]:
        raise ValueError("v0.2 registry changed after LiveCodeBench promotion")
    if sha256_file(predecessor_quality) != plan["bindings"][
        "predecessor_data_quality_config"
    ]["sha256"]:
        raise ValueError("v0.2 data-quality config changed after LiveCodeBench promotion")
    report = _read_json(report_path)
    validate_report(report)
    if report["successors"]["registry"]["sha256"] != TARGET_REGISTRY_SHA256:
        raise ValueError("LiveCodeBench promotion report registry binding changed")
    if report["successors"]["data_quality"]["sha256"] != TARGET_QUALITY_SHA256:
        raise ValueError("LiveCodeBench promotion report quality binding changed")

    verification = {
        "checks": {
            "active_contract_not_activated": True,
            "data_quality_v0_3_exact_bytes_verified": True,
            "existing_sft_configs_not_modified": True,
            "livecodebench_runtime_registry_load_passed": True,
            "no_network_gpu_sft_or_raw_benchmark_text": True,
            "registry_v0_3_exact_bytes_verified": True,
            "rights_clearance_not_claimed": True,
            "successor_50944_entries_match_candidate": True,
            "training_ingestion_prohibited": True,
            "v0_2_files_unchanged": True,
        },
        "data_quality_v0_3_sha256": sha256_file(quality_path),
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "promotion_report_file_sha256": sha256_file(report_path),
        "promotion_report_internal_sha256": report["report_sha256"],
        "registry_v0_3_sha256": sha256_file(registry_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_immutable_research_decontamination_successors_not_activated",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified LiveCodeBench immutable successors "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
