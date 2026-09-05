#!/usr/bin/env python3
"""Independently verify the BigCodeBench prompt-signature candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import (
    MAXIMUM_SIGNATURES,
    PREDECESSOR_ENTRIES,
    PROHIBITED_FIELDS,
    SOURCE_OBJECT,
    SOURCE_SHA256,
    VERIFICATION_SCHEMA_VERSION,
    build_candidate_metadata,
    build_prompt_signatures,
    load_acquisition_plan,
    load_prompt_rows,
    load_registry,
    records_sha256,
    sha256_value,
    validate_report,
    write_json_atomic,
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
    plan = load_acquisition_plan(plan_path.resolve(), repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    source_path = private_root / "raw" / SOURCE_OBJECT
    signature_path = private_root / "signatures" / "bigcodebench-v0.1.4-signatures.jsonl"
    attribution_path = private_root / "signatures" / "attribution-manifest.json"
    private_report_path = private_root / "raw-text-free-acquisition-report.json"
    candidate_path = (root / plan["output"]["candidate_registry_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    verification_path = (root / plan["output"]["verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite BigCodeBench verification")
    raw_files = sorted(path for path in (private_root / "raw").rglob("*") if path.is_file())
    if raw_files != [source_path]:
        raise ValueError("BigCodeBench private raw-object inventory changed")

    report = _read_json(report_path)
    private_report = _read_json(private_report_path)
    validate_report(report)
    if report != private_report:
        raise ValueError("private and Git BigCodeBench reports differ")
    rows, decoded = load_prompt_rows(
        source_path,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    signatures = build_prompt_signatures(rows)
    private_signatures = _read_jsonl(signature_path)
    if signatures != private_signatures or len(signatures) != MAXIMUM_SIGNATURES:
        raise ValueError("BigCodeBench private signature replay changed")
    forbidden = {"text", "prompt", *PROHIBITED_FIELDS}
    if any(forbidden.intersection(entry) for entry in signatures):
        raise ValueError("BigCodeBench signature ledger retained prohibited data")

    active_path = (root / plan["bindings"]["active_benchmark_registry"]["path"]).resolve()
    if sha256_file(active_path) != plan["bindings"]["active_benchmark_registry"]["sha256"]:
        raise ValueError("active benchmark registry changed")
    active_metadata, active_entries = load_registry(active_path)
    candidate_metadata, candidate_entries = load_registry(candidate_path)
    if len(active_entries) != PREDECESSOR_ENTRIES:
        raise ValueError("active benchmark registry count changed")
    expected_metadata = build_candidate_metadata(active_metadata, active_entries, signatures)
    if candidate_metadata != expected_metadata:
        raise ValueError("BigCodeBench candidate metadata changed")
    if candidate_entries[:PREDECESSOR_ENTRIES] != active_entries:
        raise ValueError("BigCodeBench candidate did not preserve predecessor entries")
    if candidate_entries[PREDECESSOR_ENTRIES:] != signatures:
        raise ValueError("BigCodeBench candidate appended signature inventory changed")
    if len(candidate_entries) != PREDECESSOR_ENTRIES + MAXIMUM_SIGNATURES:
        raise ValueError("BigCodeBench candidate entry count changed")

    attribution = _read_json(attribution_path)
    attribution_seal = attribution.get("attribution_sha256")
    unsigned_attribution = {
        key: value for key, value in attribution.items() if key != "attribution_sha256"
    }
    if sha256_value(unsigned_attribution) != attribution_seal:
        raise ValueError("BigCodeBench attribution seal changed")
    expected_artifacts = {
        "attribution_manifest": attribution_path,
        "candidate_registry": candidate_path,
        "private_signature_ledger": signature_path,
        "source_object": source_path,
    }
    if set(report["artifacts"]) != set(expected_artifacts):
        raise ValueError("BigCodeBench report artifact inventory changed")
    for name, path in expected_artifacts.items():
        expected = report["artifacts"][name]
        if expected != {"bytes": path.stat().st_size, "sha256": sha256_file(path)}:
            raise ValueError(f"BigCodeBench artifact changed: {name}")
    if report["decoded"] != decoded:
        raise ValueError("BigCodeBench decoded-byte replay changed")
    if report["signatures"]["records_sha256"] != records_sha256(signatures):
        raise ValueError("BigCodeBench signature digest changed")
    if report["candidate_registry"]["entries_sha256"] != records_sha256(candidate_entries):
        raise ValueError("BigCodeBench candidate digest changed")
    if report["source"]["object_sha256"] != SOURCE_SHA256:
        raise ValueError("BigCodeBench source report binding changed")

    verification = {
        "candidate_registry_file_sha256": sha256_file(candidate_path),
        "checks": {
            "active_46999_entries_preserved_in_order": True,
            "active_registry_unchanged": True,
            "candidate_entries_50419": True,
            "candidate_not_promoted": True,
            "decoded_bytes_within_32_mib": True,
            "exact_source_object_verified": True,
            "no_gpu_or_sft": True,
            "no_livecodebench_access": True,
            "prohibited_field_values_not_read_hashed_or_retained": True,
            "raw_benchmark_text_in_git": False,
            "signature_replay_3420_exact": True,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "report_file_sha256": sha256_file(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_object_sha256": sha256_file(source_path),
        "status": "verified_prompt_only_candidate_no_promotion",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified BigCodeBench prompt-signature candidate "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
