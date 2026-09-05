#!/usr/bin/env python3
"""Materialize the exact immutable LiveCodeBench registry/config successors."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_bigcodebench_signatures import sha256_value, write_json_atomic
from lumi.posttraining_livecodebench_signature_promotion import (
    REPORT_SCHEMA_VERSION,
    TARGET_ENTRIES,
    build_successor_payloads,
    load_promotion_authorization,
    validate_report,
    write_bytes_atomic,
)


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
    if any(path.exists() for path in (registry_path, quality_path, report_path)):
        raise FileExistsError("refusing to overwrite a LiveCodeBench promotion output")
    predecessor_registry = (
        root / plan["bindings"]["predecessor_registry"]["path"]
    ).resolve()
    predecessor_quality = (
        root / plan["bindings"]["predecessor_data_quality_config"]["path"]
    ).resolve()
    predecessor_registry_before = sha256_file(predecessor_registry)
    predecessor_quality_before = sha256_file(predecessor_quality)
    registry_bytes, quality_bytes, promotion = build_successor_payloads(
        plan,
        repository_root=root,
    )
    write_bytes_atomic(registry_path, registry_bytes)
    write_bytes_atomic(quality_path, quality_bytes)
    if sha256_file(predecessor_registry) != predecessor_registry_before:
        raise ValueError("v0.2 benchmark registry changed during promotion")
    if sha256_file(predecessor_quality) != predecessor_quality_before:
        raise ValueError("v0.2 data-quality config changed during promotion")

    report = {
        "authorization": plan["authorization"],
        "boundaries": {
            "active_contract_activated": False,
            "dataset_sealed": False,
            "external_distribution_performed": False,
            "gpu_or_sft_used": False,
            "network_used": False,
            "raw_benchmark_text_accessed": False,
            "rights_clearance_claimed": False,
            "training_ingestion_performed": False,
            "v0_2_config_mutated": False,
            "v0_2_registry_mutated": False,
        },
        "observed_at": "2026-08-07",
        "predecessors": {
            "data_quality_v0_2_sha256": predecessor_quality_before,
            "registry_v0_2_sha256": predecessor_registry_before,
        },
        "production_eligible": False,
        "promotion_plan_internal_sha256": promotion["plan_sha256"],
        "remaining_blockers": [
            "successor_not_activated_in_sft_contracts",
            "livecodebench_rights_remain_unresolved",
            "opencodeinstruct_candidate_reaudit_against_livecodebench_not_run",
            "dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "immutable_research_decontamination_successors_materialized_not_activated",
        "successors": {
            "data_quality": {
                "bytes": quality_path.stat().st_size,
                "path": plan["output"]["data_quality_successor_path"],
                "sha256": sha256_file(quality_path),
            },
            "registry": {
                "bytes": registry_path.stat().st_size,
                "entries": TARGET_ENTRIES,
                "path": plan["output"]["registry_successor_path"],
                "sha256": sha256_file(registry_path),
            },
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    write_json_atomic(report_path, report)
    print(
        "PASS: immutable LiveCodeBench research/decontamination successors materialized "
        f"registry={report['successors']['registry']['sha256']} "
        f"quality={report['successors']['data_quality']['sha256']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
