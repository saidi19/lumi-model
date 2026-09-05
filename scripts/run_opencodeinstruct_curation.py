#!/usr/bin/env python3
"""Run the authorized CPU-only OpenCodeInstruct single-shard curation."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_opencodeinstruct_curation import (
    REPORT_SCHEMA_VERSION,
    REQUIRED_CODING_BENCHMARKS,
    SHARD_PATH,
    apply_conversation_near_dedup,
    artifact_inventory,
    build_opencode_owner_review_html,
    deterministic_owner_sample,
    file_sha256,
    load_opencodeinstruct_plan,
    load_runtime_components,
    load_source_envelopes,
    measure_and_select,
    run_local_unit_test_verifier,
    sha256_value,
    validate_raw_text_free_report,
    verify_raw_objects,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_opencodeinstruct_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    artifact_root = (root.parent / "lumi-artifacts").resolve()
    if not private_root.is_relative_to(artifact_root):
        raise ValueError("OpenCodeInstruct private root escaped lumi-artifacts")
    raw_root = private_root / "raw"
    curated_root = private_root / "curated"
    review_root = private_root / "review"
    if not raw_root.is_dir():
        raise FileNotFoundError("OpenCodeInstruct authorized raw root is missing")
    for directory in (curated_root, review_root):
        if directory.exists() and any(directory.rglob("*")):
            raise FileExistsError(
                f"refusing nonempty OpenCodeInstruct output directory: {directory}"
            )
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite curation report: {report_path}")

    acquisition = verify_raw_objects(raw_root)
    if acquisition["network_bytes"] > plan["limits"]["maximum_network_bytes"]:
        raise ValueError("OpenCodeInstruct network byte limit exceeded")
    envelopes, source_metrics = load_source_envelopes(
        raw_root / SHARD_PATH,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
    )
    worker_path = root / plan["bindings"]["unit_test_worker"]["path"]
    verifier_ledger, verified_ids, verifier_reasons = run_local_unit_test_verifier(
        envelopes,
        worker_path=worker_path,
    )
    by_id = {str(envelope["record"]["record_id"]): envelope for envelope in envelopes}
    locally_verified = [by_id[record_id] for record_id in sorted(verified_ids)]
    candidate_records = [envelope["record"] for envelope in locally_verified]
    candidate_attributions = {
        str(envelope["record"]["record_id"]): envelope["attribution"]
        for envelope in locally_verified
    }

    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, filter_report = filter_sft_records(
        candidate_records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    near_accepted, near_ledger, near_metrics = apply_conversation_near_dedup(
        structurally_accepted,
        near_duplicate_config=near_config,
    )
    final_records, token_metrics, token_ledger = measure_and_select(
        near_accepted,
        maximum_train_supervised_tokens=plan["limits"][
            "maximum_retained_train_supervised_tokens"
        ],
        target_train_supervised_tokens=plan["limits"][
            "target_train_supervised_tokens"
        ],
        validation_fraction=plan["limits"]["validation_fraction"],
        template=template,
        tokenizer=tokenizer,
    )
    final_ids = {str(record["record_id"]) for record in final_records}
    split_by_id = {str(record["record_id"]): str(record["split"]) for record in final_records}
    final_attributions = [
        {**candidate_attributions[record_id], "split": split_by_id[record_id]}
        for record_id in sorted(final_ids)
    ]
    final_records = sorted(final_records, key=lambda row: (row["split"], row["record_id"]))
    owner_sample = deterministic_owner_sample(
        final_records,
        maximum_records=plan["limits"]["owner_sample_records"],
    )
    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("OpenCodeInstruct owner sample count changed")
    owner_html, owner_sample_sha256 = build_opencode_owner_review_html(owner_sample)

    private_paths = {
        "accepted_records": curated_root / "accepted-records.jsonl",
        "attribution_manifest": curated_root / "attribution-manifest.jsonl",
        "filter_decisions": curated_root / "filter-decisions.json",
        "near_dedup_ledger": curated_root / "near-dedup-ledger.jsonl",
        "owner_review_html": review_root / "owner-review-32.html",
        "owner_sample": review_root / "owner-sample.jsonl",
        "token_ledger": curated_root / "token-ledger.jsonl",
        "token_metrics": curated_root / "token-metrics.json",
        "verifier_ledger": curated_root / "unit-test-verifier-ledger.jsonl",
    }
    write_jsonl_atomic(private_paths["accepted_records"], final_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], final_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["near_dedup_ledger"], near_ledger)
    write_text_atomic(private_paths["owner_review_html"], owner_html)
    write_jsonl_atomic(private_paths["owner_sample"], owner_sample)
    write_jsonl_atomic(private_paths["token_ledger"], token_ledger)
    write_json_atomic(private_paths["token_metrics"], token_metrics)
    write_jsonl_atomic(private_paths["verifier_ledger"], verifier_ledger)

    artifacts = artifact_inventory(
        private_paths,
        raw_text_artifacts={"accepted_records", "owner_review_html", "owner_sample"},
    )
    present_benchmarks = sorted(REQUIRED_CODING_BENCHMARKS & benchmark.benchmarks)
    missing_benchmarks = sorted(REQUIRED_CODING_BENCHMARKS - benchmark.benchmarks)
    benchmark_rejections = {
        reason.split(":", 1)[1]: count
        for reason, count in filter_report["rejection_counts"].items()
        if reason.startswith("benchmark_contamination:")
    }
    split_counts = Counter(str(record["split"]) for record in final_records)
    train_tokens = token_metrics["by_split"]["train"]["supervised_tokens"]
    validation_tokens = token_metrics["by_split"]["validation"]["supervised_tokens"]
    report = {
        "acquisition": {
            **acquisition,
            "exact_authorized_objects_only": True,
            "source_objects": 2,
        },
        "artifacts": artifacts,
        "authorization": dict(plan["authorization"]),
        "benchmark_decontamination": {
            "active_registry_sha256": benchmark.sha256,
            "matches_rejected": dict(sorted(benchmark_rejections.items())),
            "missing_required_benchmarks": missing_benchmarks,
            "present_required_benchmarks": present_benchmarks,
            "production_complete": not missing_benchmarks,
            "required_benchmarks": sorted(REQUIRED_CODING_BENCHMARKS),
        },
        "curation": {
            "filter": {
                "accepted": filter_report["accepted"],
                "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
                "raw_text_retained": False,
                "rejected": filter_report["rejected"],
                "rejection_counts": filter_report["rejection_counts"],
                "report_sha256": filter_report["report_sha256"],
                "source_registry_check_deferred_until_promotion": True,
            },
            "near_duplicate": near_metrics,
            "retained_records": len(final_records),
            "split_counts": dict(sorted(split_counts.items())),
            "token_measurement": token_metrics,
            "unit_test_verifier": {
                "eligible_before_local_verification": len(envelopes),
                "failed": len(envelopes) - len(verified_ids),
                "passed": len(verified_ids),
                "published_status_required": "pass",
                "reason_counts": verifier_reasons,
                "sandbox": "python_isolated_ast_and_runtime_fail_closed_v0.1",
            },
        },
        "mix_projection": {
            "candidate_train_supervised_tokens": train_tokens,
            "candidate_validation_supervised_tokens": validation_tokens,
            "planning_target_train_supervised_tokens": plan["limits"][
                "target_train_supervised_tokens"
            ],
            "projection_is_not_promotion": True,
        },
        "observed_at": plan["observed_at"],
        "owner_review": {
            "complete": False,
            "record_count": len(owner_sample),
            "sample_sha256": owner_sample_sha256,
            "status": "pending_owner_review",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_sealed": False,
            "production_eligible": False,
            "production_sft_authorized": False,
        },
        "raw_training_text_present": False,
        "remaining_blockers": [
            "deterministic_32_row_owner_quality_review_not_completed",
            *(
                ["livecodebench_and_bigcodebench_hash_registry_not_available_locally"]
                if missing_benchmarks
                else []
            ),
            "opencodeinstruct_source_registry_promotion_not_authorized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "runtime_seconds": round(time.monotonic() - started, 3),
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": plan["source"]["dataset_id"],
            "decoded_bytes": source_metrics["decoded_bytes"],
            "license_id": plan["source"]["declared_license_id"],
            "object_path": SHARD_PATH,
            "object_sha256": plan["source"]["objects"][1]["sha256"],
            "revision": plan["source"]["revision"],
            "row_gate": source_metrics,
            "source_id": plan["source"]["source_id"],
            "source_rows": source_metrics["rows"],
        },
        "status": "curated_nonproduction_owner_review_pending",
    }
    if report["runtime_seconds"] > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OpenCodeInstruct curation exceeded authorized runtime")
    report["report_sha256"] = sha256_value(report)
    validate_raw_text_free_report(report)
    write_json_atomic(private_root / "raw-text-free-curation-report.json", report)
    write_json_atomic(report_path, report)
    print(
        "PASS: OpenCodeInstruct CPU curation "
        f"rows={source_metrics['rows']} local_verified={len(verified_ids)} "
        f"retained={len(final_records)} train_supervised_tokens={train_tokens} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
