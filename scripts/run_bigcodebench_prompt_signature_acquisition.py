#!/usr/bin/env python3
"""Acquire one pinned BigCodeBench object and build prompt-only signatures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import (
    MAXIMUM_SIGNATURES,
    PREDECESSOR_ENTRIES,
    PROMPT_FIELDS,
    REPORT_SCHEMA_VERSION,
    SOURCE_OBJECT,
    SOURCE_REVISION,
    SOURCE_SHA256,
    build_candidate_metadata,
    build_prompt_signatures,
    download_exact_object,
    load_acquisition_plan,
    load_prompt_rows,
    load_registry,
    records_sha256,
    sha256_value,
    validate_report,
    write_candidate_registry_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


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
    if any(
        path.exists()
        for path in (
            source_path,
            signature_path,
            attribution_path,
            private_report_path,
            candidate_path,
            report_path,
        )
    ):
        raise FileExistsError("refusing to overwrite a BigCodeBench acquisition artifact")

    active_path = (root / plan["bindings"]["active_benchmark_registry"]["path"]).resolve()
    active_sha256_before = sha256_file(active_path)
    network_bytes = download_exact_object(
        source_path,
        maximum_network_bytes=plan["limits"]["maximum_network_bytes"],
    )
    rows, decoded = load_prompt_rows(
        source_path,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    signatures = build_prompt_signatures(rows)
    predecessor_metadata, predecessor_entries = load_registry(active_path)
    if len(predecessor_entries) != PREDECESSOR_ENTRIES:
        raise ValueError("active benchmark registry entry count changed")
    predecessor_identities = {
        (entry["benchmark"], entry["example_id"]) for entry in predecessor_entries
    }
    new_identities = {(entry["benchmark"], entry["example_id"]) for entry in signatures}
    if predecessor_identities.intersection(new_identities):
        raise ValueError("BigCodeBench prompt signatures collide with active registry")

    write_jsonl_atomic(signature_path, signatures)
    field_counts = Counter(
        str(entry["example_id"]).rsplit(":", maxsplit=1)[-1] for entry in signatures
    )
    attribution = {
        "dataset_id": "bigcode/bigcodebench",
        "declared_license_id": "apache-2.0",
        "field_counts": dict(sorted(field_counts.items())),
        "object_sha256": SOURCE_SHA256,
        "prompt_fields": list(PROMPT_FIELDS),
        "raw_text_retained": False,
        "revision": SOURCE_REVISION,
        "signatures": len(signatures),
        "source_object": SOURCE_OBJECT,
    }
    attribution["attribution_sha256"] = sha256_value(attribution)
    write_json_atomic(attribution_path, attribution)

    candidate_metadata = build_candidate_metadata(
        predecessor_metadata,
        predecessor_entries,
        signatures,
    )
    candidate_entries = [*predecessor_entries, *signatures]
    write_candidate_registry_atomic(
        candidate_path,
        metadata=candidate_metadata,
        entries=candidate_entries,
    )
    active_sha256_after = sha256_file(active_path)
    if active_sha256_after != active_sha256_before:
        raise ValueError("active benchmark registry changed during acquisition")

    report = {
        "active_registry": {
            "entries": len(predecessor_entries),
            "file_sha256_after": active_sha256_after,
            "file_sha256_before": active_sha256_before,
            "mutated": False,
            "predecessor_entries_sha256": records_sha256(predecessor_entries),
        },
        "artifacts": {
            "attribution_manifest": {
                "bytes": attribution_path.stat().st_size,
                "sha256": sha256_file(attribution_path),
            },
            "candidate_registry": {
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
            },
            "private_signature_ledger": {
                "bytes": signature_path.stat().st_size,
                "sha256": sha256_file(signature_path),
            },
            "source_object": {
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            },
        },
        "authorization": plan["authorization"],
        "boundaries": {
            "active_registry_mutated": False,
            "candidate_promoted": False,
            "gpu_or_sft_used": False,
            "livecodebench_accessed": False,
            "prohibited_field_values_hashed": False,
            "prohibited_field_values_read": False,
            "raw_benchmark_text_in_git": False,
        },
        "candidate_registry": {
            "appended_entries": len(signatures),
            "candidate_not_active": True,
            "entries": len(candidate_entries),
            "entries_sha256": records_sha256(candidate_entries),
            "path": plan["output"]["candidate_registry_path"],
        },
        "decoded": decoded,
        "network_payload_bytes": network_bytes,
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "remaining_blockers": [
            "independent_verification_pending",
            "candidate_registry_promotion_not_authorized",
            "opencodeinstruct_candidate_registry_promotion_not_authorized",
            "dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "signatures": {
            "entries": len(signatures),
            "field_counts": dict(sorted(field_counts.items())),
            "records_sha256": records_sha256(signatures),
        },
        "source": {
            "allowed_columns_read": ["task_id", *PROMPT_FIELDS],
            "dataset_id": "bigcode/bigcodebench",
            "object": SOURCE_OBJECT,
            "object_sha256": SOURCE_SHA256,
            "rows": len(rows),
            "revision": SOURCE_REVISION,
        },
        "status": "candidate_built_pending_independent_verification_no_promotion",
    }
    if report["signatures"]["entries"] != MAXIMUM_SIGNATURES:
        raise ValueError("BigCodeBench signature count changed")
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    write_json_atomic(private_report_path, report)
    write_json_atomic(report_path, report)
    print(
        "PASS: BigCodeBench prompt-only signature candidate built "
        f"rows={len(rows)} signatures={len(signatures)} "
        f"candidate_entries={len(candidate_entries)} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
