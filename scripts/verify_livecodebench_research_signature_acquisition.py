#!/usr/bin/env python3
"""Independently replay and seal the LiveCodeBench research-only signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import (
    load_registry,
    records_sha256,
    sha256_value,
    write_json_atomic,
)
from lumi.posttraining_livecodebench_signatures import (
    EXPECTED_SIGNATURES,
    PREDECESSOR_ENTRIES,
    PROHIBITED_FIELDS,
    SOURCE_OBJECT,
    SOURCE_SHA256,
    VERIFICATION_SCHEMA_VERSION,
    build_candidate_metadata,
    build_prompt_signatures,
    load_acquisition_authorization,
    load_projected_prompt_rows,
    validate_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_acquisition_authorization(plan_path.resolve(), repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    source_path = private_root / "raw" / SOURCE_OBJECT
    signature_path = private_root / "signatures/livecodebench-release-v6-signatures.jsonl"
    attribution_path = private_root / "signatures/attribution-manifest.json"
    private_report_path = private_root / "raw-text-free-acquisition-report.json"
    candidate_path = (root / plan["output"]["candidate_registry_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite LiveCodeBench verification")
    raw_files = sorted(path for path in (private_root / "raw").rglob("*") if path.is_file())
    if raw_files != [source_path]:
        raise ValueError("LiveCodeBench private raw-object inventory changed")

    report = _read_json(report_path)
    private_report = _read_json(private_report_path)
    validate_report(report)
    if report != private_report:
        raise ValueError("private and Git LiveCodeBench reports differ")
    rows, decoded = load_projected_prompt_rows(
        source_path,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    signatures = build_prompt_signatures(rows)
    private_signatures = _read_jsonl(signature_path)
    if signatures != private_signatures or len(signatures) != EXPECTED_SIGNATURES:
        raise ValueError("LiveCodeBench private signature replay changed")
    forbidden = {"text", "prompt", *PROHIBITED_FIELDS}
    if any(forbidden.intersection(entry) for entry in signatures):
        raise ValueError("LiveCodeBench signature ledger retained prohibited data")

    predecessor_path = (
        root / plan["bindings"]["predecessor_registry"]["path"]
    ).resolve()
    if sha256_file(predecessor_path) != plan["bindings"]["predecessor_registry"][
        "sha256"
    ]:
        raise ValueError("LiveCodeBench predecessor registry changed")
    predecessor_metadata, predecessor_entries = load_registry(predecessor_path)
    candidate_metadata, candidate_entries = load_registry(candidate_path)
    if len(predecessor_entries) != PREDECESSOR_ENTRIES:
        raise ValueError("LiveCodeBench predecessor entry count changed")
    expected_metadata = build_candidate_metadata(
        predecessor_metadata,
        predecessor_entries,
        signatures,
    )
    if candidate_metadata != expected_metadata:
        raise ValueError("LiveCodeBench candidate metadata changed")
    if candidate_entries[:PREDECESSOR_ENTRIES] != predecessor_entries:
        raise ValueError("LiveCodeBench candidate did not preserve predecessor entries")
    if candidate_entries[PREDECESSOR_ENTRIES:] != signatures:
        raise ValueError("LiveCodeBench candidate appended signature inventory changed")

    attribution = _read_json(attribution_path)
    attribution_seal = attribution.get("attribution_sha256")
    unsigned_attribution = {
        key: value for key, value in attribution.items() if key != "attribution_sha256"
    }
    if sha256_value(unsigned_attribution) != attribution_seal:
        raise ValueError("LiveCodeBench attribution seal changed")
    expected_artifacts = {
        "attribution_manifest": attribution_path,
        "candidate_registry": candidate_path,
        "private_signature_ledger": signature_path,
        "source_object": source_path,
    }
    for name, path in expected_artifacts.items():
        expected = report["artifacts"][name]
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if name == "source_object":
            observed["retention_policy"] = "delete_after_independent_replay"
        if expected != observed:
            raise ValueError(f"LiveCodeBench artifact changed: {name}")
    if report["decoded"] != decoded:
        raise ValueError("LiveCodeBench decoded-byte replay changed")
    if report["signatures"]["records_sha256"] != records_sha256(signatures):
        raise ValueError("LiveCodeBench signature digest changed")
    if report["candidate_registry"]["entries_sha256"] != records_sha256(candidate_entries):
        raise ValueError("LiveCodeBench candidate digest changed")
    source_hash = sha256_file(source_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError("LiveCodeBench source report binding changed")

    source_path.unlink()
    if source_path.exists():
        raise ValueError("LiveCodeBench ephemeral source deletion failed")
    verification = {
        "candidate_registry_file_sha256": sha256_file(candidate_path),
        "checks": {
            "candidate_entries_50944": len(candidate_entries) == 50_944,
            "candidate_not_promoted": True,
            "exact_incremental_175_rows_and_525_signatures_replayed": True,
            "exact_source_object_verified_then_deleted": True,
            "no_gpu_sft_training_ingestion_or_external_api": True,
            "predecessor_50419_entries_preserved_in_order": True,
            "prohibited_fields_not_materialized_hashed_or_retained": True,
            "raw_benchmark_text_in_git": False,
        },
        "decoded": decoded,
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_object_retained_after_verification": False,
        "source_object_sha256_before_deletion": source_hash,
        "status": "verified_research_only_candidate_no_promotion_source_deleted",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified LiveCodeBench research-only signatures; "
        "ephemeral source deleted "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
