#!/usr/bin/env python3
"""Run the exact CPU-only DMath curation authorized by the owner."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_dmath_curation import (
    REPORT_SCHEMA_VERSION,
    SOURCE_OBJECT_PATH,
    apply_prompt_near_dedup,
    artifact_inventory,
    assign_validation_split,
    build_candidate_records,
    build_dmath_owner_review_html,
    count_by_split,
    deterministic_owner_sample,
    file_sha256,
    load_dmath_curation_plan,
    load_runtime_components,
    load_source_rows,
    run_isolated_verifier,
    sha256_value,
    validate_raw_text_free_report,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from lumi.posttraining_oasst1_full_run import measure_token_fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_dmath_curation_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    artifact_root = (root.parent / "lumi-artifacts").resolve()
    if not private_root.is_relative_to(artifact_root):
        raise ValueError("DMath curation private root escaped lumi-artifacts")
    raw_root = private_root / "raw"
    expected_raw_files = {
        "LICENSE",
        "README.md",
        SOURCE_OBJECT_PATH,
    }
    observed_raw_files = {
        path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*") if path.is_file()
    }
    if observed_raw_files != expected_raw_files:
        raise ValueError("DMath private raw inventory changed or test split appeared")
    for directory in (private_root / "curated", private_root / "review"):
        if directory.exists() and any(directory.rglob("*")):
            raise FileExistsError(f"refusing nonempty DMath curation directory: {directory}")
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite DMath curation report: {report_path}")

    rows = load_source_rows(raw_root / SOURCE_OBJECT_PATH)
    worker_path = root / plan["bindings"]["verifier_worker"]["path"]
    verifier_ledger, verified_source_ids, verifier_reason_counts = run_isolated_verifier(
        rows,
        worker_path=worker_path,
    )
    records, attributions = build_candidate_records(
        rows,
        verified_source_ids=verified_source_ids,
    )
    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    near_accepted, near_ledger, near_metrics = apply_prompt_near_dedup(
        structurally_accepted,
        near_duplicate_config=near_config,
    )
    attribution_by_id = {row["record_id"]: row for row in attributions}
    split_records = assign_validation_split(
        near_accepted,
        attribution_by_id=attribution_by_id,
        validation_fraction=plan["limits"]["validation_fraction"],
    )
    final_records, token_metrics = measure_token_fit(
        split_records,
        template=template,
        tokenizer=tokenizer,
    )
    if len(final_records) > plan["limits"]["maximum_retained_records"]:
        raise ValueError("DMath retained record limit exceeded")
    final_ids = {str(record["record_id"]) for record in final_records}
    final_attributions = []
    split_by_id = {str(record["record_id"]): str(record["split"]) for record in final_records}
    for attribution in attributions:
        record_id = str(attribution["record_id"])
        if record_id in final_ids:
            final_attributions.append({**attribution, "split": split_by_id[record_id]})
    final_attributions.sort(key=lambda row: row["record_id"])
    final_records = sorted(final_records, key=lambda row: (row["split"], row["record_id"]))
    final_attribution_by_id = {row["record_id"]: row for row in final_attributions}
    owner_sample = deterministic_owner_sample(
        final_records,
        attribution_by_id=final_attribution_by_id,
        maximum_records=plan["limits"]["owner_sample_records"],
    )
    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("DMath owner-review sample count changed")
    owner_html, owner_sample_sha256 = build_dmath_owner_review_html(owner_sample)

    private_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_dedup_ledger": private_root / "curated" / "near-dedup-ledger.jsonl",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
        "verifier_ledger": private_root / "curated" / "verifier-ledger.jsonl",
    }
    write_jsonl_atomic(private_paths["accepted_records"], final_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], final_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["near_dedup_ledger"], near_ledger)
    write_text_atomic(private_paths["owner_review_html"], owner_html)
    write_jsonl_atomic(private_paths["owner_sample"], owner_sample)
    write_json_atomic(private_paths["token_metrics"], token_metrics)
    write_jsonl_atomic(private_paths["verifier_ledger"], verifier_ledger)

    artifacts = artifact_inventory(
        private_paths,
        raw_text_artifacts={"accepted_records", "owner_review_html", "owner_sample"},
    )
    category_counts = Counter(str(row["upstream_category"]) for row in final_attributions)
    train_supervised_tokens = int(
        token_metrics["by_split"].get("train", {}).get("supervised_tokens", 0)
    )
    planning_train_tokens = 1_002_200
    report = {
        "acquisition_report_internal_sha256": (
            "73756b72b0b0734ab50bf177c6c04b58c7e67dd68d0629a198669da184d0d7a9"
        ),
        "artifacts": artifacts,
        "authorization": dict(plan["authorization"]),
        "curation": {
            "category_counts": dict(sorted(category_counts.items())),
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
            "split_counts": count_by_split(final_records),
            "token_measurement": token_metrics,
            "verifier": {
                "abstract_expression_used_as_training_target": False,
                "failed": len(rows) - len(verified_source_ids),
                "passed": len(verified_source_ids),
                "python_solution_used_as_training_target": False,
                "reason_counts": verifier_reason_counts,
                "sandbox": "python_isolated_no_site_fail_closed_v0.1",
            },
        },
        "mix_projection": {
            "candidate_train_supervised_tokens": train_supervised_tokens,
            "planning_train_supervised_tokens_before_candidate": planning_train_tokens,
            "projected_10m_floor_gap_after_review_and_promotion": max(
                0,
                10_000_000 - planning_train_tokens - train_supervised_tokens,
            ),
            "projected_planning_train_supervised_tokens": (
                planning_train_tokens + train_supervised_tokens
            ),
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
            "dmath_source_registry_promotion_not_authorized",
            "sft_mix_floor_not_met",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "runtime_seconds": round(time.monotonic() - started, 3),
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": plan["source"]["dataset_id"],
            "license_id": plan["source"]["declared_license_id"],
            "object_path": plan["source"]["object_path"],
            "object_sha256": plan["source"]["object_sha256"],
            "revision": plan["source"]["revision"],
            "source_id": plan["source"]["source_id"],
            "source_rows": len(rows),
            "test_split_accessed": False,
        },
        "status": "curated_nonproduction_owner_review_pending",
    }
    if report["runtime_seconds"] > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("DMath curation exceeded the authorized runtime")
    report["report_sha256"] = sha256_value(report)
    validate_raw_text_free_report(report)
    write_json_atomic(private_root / "raw-text-free-curation-report.json", report)
    write_json_atomic(report_path, report)
    print(
        "PASS: DMath CPU curation "
        f"source_rows={len(rows)} verifier_passed={len(verified_source_ids)} "
        f"accepted={len(final_records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
