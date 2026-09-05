#!/usr/bin/env python3
"""Acquire one pinned LiveCodeBench object and build research-only signatures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import (
    load_registry,
    records_sha256,
    sha256_value,
    write_candidate_registry_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from lumi.posttraining_livecodebench_signatures import (
    EXPECTED_SIGNATURES,
    PREDECESSOR_ENTRIES,
    PROMPT_VARIANTS,
    REPORT_SCHEMA_VERSION,
    SOURCE_OBJECT,
    SOURCE_REVISION,
    SOURCE_SHA256,
    build_candidate_metadata,
    build_prompt_signatures,
    download_exact_object,
    load_acquisition_authorization,
    load_projected_prompt_rows,
    validate_report,
)


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
    if any(
        path.exists()
        for path in (
            signature_path,
            attribution_path,
            private_report_path,
            candidate_path,
            report_path,
        )
    ):
        raise FileExistsError("refusing to overwrite a LiveCodeBench acquisition artifact")

    predecessor_path = (
        root / plan["bindings"]["predecessor_registry"]["path"]
    ).resolve()
    predecessor_sha256_before = sha256_file(predecessor_path)
    source_reused = source_path.is_file()
    if source_reused:
        if source_path.stat().st_size != 134_303_240 or sha256_file(source_path) != (
            SOURCE_SHA256
        ):
            raise ValueError("failed LiveCodeBench attempt left a changed source object")
        network_bytes = source_path.stat().st_size
    else:
        network_bytes = download_exact_object(
            source_path,
            maximum_network_bytes=plan["limits"]["maximum_network_bytes"],
        )
    rows, decoded = load_projected_prompt_rows(
        source_path,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    signatures = build_prompt_signatures(rows)
    predecessor_metadata, predecessor_entries = load_registry(predecessor_path)
    if len(predecessor_entries) != PREDECESSOR_ENTRIES:
        raise ValueError("LiveCodeBench predecessor registry count changed")
    predecessor_identities = {
        (entry["benchmark"], entry["example_id"]) for entry in predecessor_entries
    }
    new_identities = {(entry["benchmark"], entry["example_id"]) for entry in signatures}
    if predecessor_identities.intersection(new_identities):
        raise ValueError("LiveCodeBench signatures collide with predecessor registry")

    write_jsonl_atomic(signature_path, signatures)
    variant_counts = Counter(
        str(entry["example_id"]).rsplit(":", maxsplit=1)[-1] for entry in signatures
    )
    attribution = {
        "dataset_id": "livecodebench/code_generation_lite",
        "declared_license_id": "cc_non_spdx_ambiguous",
        "object_sha256": SOURCE_SHA256,
        "prompt_variants": list(PROMPT_VARIANTS),
        "raw_text_retained": False,
        "research_only": True,
        "revision": SOURCE_REVISION,
        "signatures": len(signatures),
        "source_object": SOURCE_OBJECT,
        "training_ingestion_prohibited": True,
        "variant_counts": dict(sorted(variant_counts.items())),
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
    predecessor_sha256_after = sha256_file(predecessor_path)
    if predecessor_sha256_after != predecessor_sha256_before:
        raise ValueError("LiveCodeBench predecessor changed during acquisition")

    report = {
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
                "retention_policy": "delete_after_independent_replay",
            },
        },
        "authorization": plan["authorization"],
        "boundaries": {
            "candidate_promoted": False,
            "gpu_or_sft_used": False,
            "prohibited_field_values_hashed": False,
            "prohibited_field_values_materialized": False,
            "prohibited_field_values_retained_outside_exact_source_object": False,
            "raw_benchmark_text_in_git": False,
            "training_ingestion_performed": False,
        },
        "candidate_registry": {
            "appended_entries": len(signatures),
            "candidate_not_active": True,
            "entries": len(candidate_entries),
            "entries_sha256": records_sha256(candidate_entries),
            "path": plan["output"]["candidate_registry_path"],
            "research_only": True,
        },
        "decoded": decoded,
        "network_payload_bytes": network_bytes,
        "source_reused_after_fail_closed_row_gate": source_reused,
        "observed_at": "2026-08-07",
        "predecessor_registry": {
            "entries": len(predecessor_entries),
            "file_sha256_after": predecessor_sha256_after,
            "file_sha256_before": predecessor_sha256_before,
            "mutated": False,
        },
        "production_eligible": False,
        "remaining_blockers": [
            "independent_verification_and_ephemeral_source_deletion_pending",
            "candidate_registry_promotion_not_authorized",
            "opencodeinstruct_reaudit_against_livecodebench_not_run",
            "dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "signatures": {
            "entries": len(signatures),
            "records_sha256": records_sha256(signatures),
            "variant_counts": dict(sorted(variant_counts.items())),
        },
        "source": {
            "allowed_columns_projected": list(decoded["projected_columns"]),
            "dataset_id": "livecodebench/code_generation_lite",
            "declared_license_id": "cc_non_spdx_ambiguous",
            "object": SOURCE_OBJECT,
            "object_sha256": SOURCE_SHA256,
            "rows": len(rows),
            "revision": SOURCE_REVISION,
            "split": "release_v6",
        },
        "status": "research_only_candidate_built_pending_independent_verification",
    }
    if report["signatures"]["entries"] != EXPECTED_SIGNATURES:
        raise ValueError("LiveCodeBench signature count changed")
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    write_json_atomic(private_report_path, report)
    write_json_atomic(report_path, report)
    print(
        "PASS: LiveCodeBench research-only prompt-signature candidate built "
        f"rows={len(rows)} signatures={len(signatures)} "
        f"candidate_entries={len(candidate_entries)} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
