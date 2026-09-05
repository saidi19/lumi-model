#!/usr/bin/env python3
"""Run the authorized CPU-only OASST2 ready-tree curation."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst2_curation import (
    REPORT_SCHEMA_VERSION,
    SOURCE_DATASET_ID,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    TREE_OBJECT,
    apply_conversation_near_dedup,
    artifact_inventory,
    build_oasst2_owner_review_html,
    deterministic_owner_sample,
    file_sha256,
    load_oasst2_plan,
    load_runtime_components,
    load_tree_candidates,
    measure_and_select,
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
    plan = load_oasst2_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    artifact_root = (root.parent / "lumi-artifacts").resolve()
    if not private_root.is_relative_to(artifact_root):
        raise ValueError("OASST2 private root escaped lumi-artifacts")
    raw_root = private_root / "raw"
    curated_root = private_root / "curated"
    review_root = private_root / "review"
    if not raw_root.is_dir():
        raise FileNotFoundError("OASST2 authorized raw root is missing")
    for directory in (curated_root, review_root):
        if directory.exists() and any(directory.rglob("*")):
            raise FileExistsError(f"refusing nonempty OASST2 output directory: {directory}")
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 report: {report_path}")

    acquisition = verify_raw_objects(raw_root)
    if acquisition["network_bytes"] > plan["limits"]["maximum_network_bytes"]:
        raise ValueError("OASST2 network byte limit exceeded")
    tree_selection = load_tree_candidates(
        raw_root / TREE_OBJECT,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
        maximum_messages=plan["limits"]["maximum_input_messages"],
        maximum_trees=plan["limits"]["maximum_input_trees"],
    )
    if tree_selection.metrics["input_trees"] != plan["limits"]["maximum_input_trees"]:
        raise ValueError("OASST2 exact tree count changed")
    if tree_selection.metrics["input_messages"] != plan["limits"]["maximum_input_messages"]:
        raise ValueError("OASST2 exact message count changed")

    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan, repository_root=root
    )
    structurally_accepted, filter_report = filter_sft_records(
        tree_selection.records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    near_accepted, near_ledger, near_metrics = apply_conversation_near_dedup(
        structurally_accepted,
        metadata=tree_selection.metadata,
        near_duplicate_config=near_config,
    )
    final_records, token_metrics, token_ledger = measure_and_select(
        near_accepted,
        maximum_train_supervised_tokens=plan["limits"]["maximum_retained_train_supervised_tokens"],
        metadata=tree_selection.metadata,
        target_task_train_supervised_tokens=plan["policy"]["target_task_train_supervised_tokens"],
        target_train_supervised_tokens=plan["limits"]["target_retained_train_supervised_tokens"],
        template=template,
        tokenizer=tokenizer,
        validation_fraction=plan["limits"]["validation_fraction"],
    )
    final_records.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    final_ids = {str(record["record_id"]) for record in final_records}
    attribution_by_id = {str(row["record_id"]): row for row in tree_selection.attributions}
    final_attributions = [attribution_by_id[record_id] for record_id in sorted(final_ids)]
    selection_ledger = [
        {
            "depth": tree_selection.metadata[record_id]["depth"],
            "endpoint_quality": tree_selection.metadata[record_id]["endpoint_quality"],
            "maximum_assistant_rank": tree_selection.metadata[record_id]["maximum_assistant_rank"],
            "message_tree_id": tree_selection.metadata[record_id]["message_tree_id"],
            "record_id": record_id,
            "split": attribution_by_id[record_id]["split"],
            "task_category": attribution_by_id[record_id]["task_category"],
            "terminal": tree_selection.metadata[record_id]["terminal"],
        }
        for record_id in sorted(final_ids)
    ]
    train_trees = {row["message_tree_id"] for row in selection_ledger if row["split"] == "train"}
    validation_trees = {
        row["message_tree_id"] for row in selection_ledger if row["split"] == "validation"
    }
    if train_trees & validation_trees:
        raise ValueError("OASST2 tree-disjoint split failed")

    owner_sample = deterministic_owner_sample(
        final_records, maximum_records=plan["limits"]["owner_sample_records"]
    )
    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("OASST2 owner sample count changed")
    owner_html, owner_sample_sha256 = build_oasst2_owner_review_html(owner_sample)

    private_paths = {
        "accepted_records": curated_root / "accepted-records.jsonl",
        "attribution_manifest": curated_root / "attribution-manifest.jsonl",
        "filter_decisions": curated_root / "filter-decisions.json",
        "near_dedup_ledger": curated_root / "near-dedup-ledger.jsonl",
        "owner_review_html": review_root / "owner-review-32.html",
        "owner_sample": review_root / "owner-sample.jsonl",
        "selection_ledger": curated_root / "selection-ledger.jsonl",
        "token_ledger": curated_root / "token-ledger.jsonl",
        "token_metrics": curated_root / "token-metrics.json",
    }
    write_jsonl_atomic(private_paths["accepted_records"], final_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], final_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["near_dedup_ledger"], near_ledger)
    write_text_atomic(private_paths["owner_review_html"], owner_html)
    write_jsonl_atomic(private_paths["owner_sample"], owner_sample)
    write_jsonl_atomic(private_paths["selection_ledger"], selection_ledger)
    write_jsonl_atomic(private_paths["token_ledger"], token_ledger)
    write_json_atomic(private_paths["token_metrics"], token_metrics)

    artifacts = artifact_inventory(
        private_paths,
        raw_text_artifacts={"accepted_records", "owner_review_html", "owner_sample"},
    )
    split_counts = Counter(str(record["split"]) for record in final_records)
    rank_counts = Counter(str(row["maximum_assistant_rank"]) for row in selection_ledger)
    benchmark_rejections = {
        reason.split(":", 1)[1]: count
        for reason, count in filter_report["rejection_counts"].items()
        if reason.startswith("benchmark_contamination:")
    }
    train_tokens = token_metrics["train_supervised_tokens"]
    planning_target = plan["limits"]["target_retained_train_supervised_tokens"]
    planning_target_reached = train_tokens >= planning_target
    report = {
        "acquisition": {
            **acquisition,
            "decoded_bytes": tree_selection.metrics["decoded_bytes"],
            "input_messages": tree_selection.metrics["input_messages"],
            "input_trees": tree_selection.metrics["input_trees"],
        },
        "artifacts": artifacts,
        "authorization": dict(plan["authorization"]),
        "benchmark_decontamination": {
            "active_registry_sha256": benchmark.sha256,
            "matches_rejected": dict(sorted(benchmark_rejections.items())),
            "raw_benchmark_text_retained": False,
        },
        "curation": {
            "filter": {
                "accepted": filter_report["accepted"],
                "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
                "rejected": filter_report["rejected"],
                "rejection_counts": filter_report["rejection_counts"],
                "report_sha256": filter_report["report_sha256"],
                "source_registry_check_deferred_until_promotion": True,
            },
            "near_duplicate": near_metrics,
            "retained_by_maximum_assistant_rank": dict(sorted(rank_counts.items())),
            "retained_records": len(final_records),
            "split_counts": dict(sorted(split_counts.items())),
            "token_measurement": token_metrics,
            "tree_disjoint_split": {
                "overlap": 0,
                "train_trees": len(train_trees),
                "validation_trees": len(validation_trees),
            },
            "tree_selection": tree_selection.metrics,
        },
        "mix_projection": {
            "candidate_train_supervised_tokens": train_tokens,
            "candidate_validation_supervised_tokens": token_metrics["validation_supervised_tokens"],
            "planning_target_reached": planning_target_reached,
            "planning_target_train_supervised_tokens": planning_target,
            "projection_is_not_promotion": True,
            "task_target_shortfalls": token_metrics["task_target_shortfalls"],
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
            *([] if planning_target_reached else ["oasst2_planning_token_target_not_reached"]),
            "oasst2_source_registry_promotion_not_authorized",
            "first_party_multilingual_safety_tool_lane_not_materialized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "runtime_seconds": round(time.monotonic() - started, 3),
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "license_id": SOURCE_LICENSE,
            "object_path": TREE_OBJECT,
            "object_sha256": file_sha256(raw_root / TREE_OBJECT),
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": (
            "curated_nonproduction_owner_review_pending"
            if planning_target_reached
            else "curated_nonproduction_short_of_planning_target_owner_review_pending"
        ),
    }
    if report["runtime_seconds"] > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OASST2 curation exceeded authorized runtime")
    report["report_sha256"] = sha256_value(report)
    validate_raw_text_free_report(report)
    write_json_atomic(private_root / "raw-text-free-curation-report.json", report)
    write_json_atomic(report_path, report)
    print(
        "PASS: OASST2 CPU curation "
        f"trees={tree_selection.metrics['input_trees']} "
        f"messages={tree_selection.metrics['input_messages']} "
        f"retained={len(final_records)} train_supervised_tokens={train_tokens} "
        f"target_reached={planning_target_reached} owner_sample={len(owner_sample)} "
        f"production_eligible=False report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
