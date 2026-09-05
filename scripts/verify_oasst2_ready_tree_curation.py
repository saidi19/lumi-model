#!/usr/bin/env python3
"""Independently replay and verify the private OASST2 ready-tree candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst2_curation import (
    ATTRIBUTION_SCHEMA_VERSION,
    SOURCE_ID,
    SOURCE_REVISION,
    TREE_OBJECT,
    VERIFICATION_SCHEMA_VERSION,
    apply_conversation_near_dedup,
    build_oasst2_owner_review_html,
    deterministic_owner_sample,
    file_sha256,
    load_oasst2_plan,
    load_runtime_components,
    load_tree_candidates,
    measure_and_select,
    sha256_value,
    tree_split,
    validate_raw_text_free_report,
    verify_raw_objects,
    write_json_atomic,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required: {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_oasst2_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (root / plan["output"]["raw_text_free_verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 verification: {verification_path}")

    acquisition = verify_raw_objects(private_root / "raw")
    report = _load_json(report_path)
    validate_raw_text_free_report(report)
    private_report = _load_json(private_root / "raw-text-free-curation-report.json")
    if private_report != report:
        raise ValueError("private and Git OASST2 reports differ")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("OASST2 report plan binding failed")
    if report["acquisition"]["network_bytes"] != acquisition["network_bytes"]:
        raise ValueError("OASST2 acquisition bytes changed")

    paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_dedup_ledger": private_root / "curated" / "near-dedup-ledger.jsonl",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "selection_ledger": private_root / "curated" / "selection-ledger.jsonl",
        "token_ledger": private_root / "curated" / "token-ledger.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    inventory = {item["name"]: item for item in report["artifacts"]["objects"]}
    if set(inventory) != set(paths):
        raise ValueError("OASST2 artifact inventory names changed")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"OASST2 private artifact missing: {name}")
        item = inventory[name]
        if item["bytes"] != path.stat().st_size or item["sha256"] != file_sha256(path):
            raise ValueError(f"OASST2 artifact changed: {name}")

    records = _load_jsonl(paths["accepted_records"])
    attributions = _load_jsonl(paths["attribution_manifest"])
    selection_ledger = _load_jsonl(paths["selection_ledger"])
    token_ledger = _load_jsonl(paths["token_ledger"])
    token_metrics = _load_json(paths["token_metrics"])
    filter_decisions = _load_json(paths["filter_decisions"])
    owner_sample = _load_jsonl(paths["owner_sample"])

    expected_record_fields = {
        "language",
        "messages",
        "provenance",
        "quality",
        "record_id",
        "safety",
        "schema_version",
        "source",
        "split",
        "task_category",
    }
    record_ids = [str(row.get("record_id")) for row in records]
    if len(record_ids) != len(set(record_ids)) or "None" in record_ids:
        raise ValueError("OASST2 record IDs are invalid or duplicated")
    for record in records:
        if set(record) != expected_record_fields:
            raise ValueError("OASST2 accepted record schema changed")
        roles = [message.get("role") for message in record.get("messages", [])]
        if not roles or roles[0] != "user" or roles[-1] != "assistant":
            raise ValueError("OASST2 message boundary changed")
        if any(left == right for left, right in zip(roles, roles[1:], strict=False)):
            raise ValueError("OASST2 role alternation changed")
        if record.get("language") != "en":
            raise ValueError("OASST2 language changed")
        if record.get("task_category") not in {
            "conversation",
            "factual_qa",
            "writing_summarization",
        }:
            raise ValueError("OASST2 retained task changed")
        source = record.get("source", {})
        if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
            raise ValueError("OASST2 source binding changed")
        if record.get("split") not in {"train", "validation"}:
            raise ValueError("OASST2 split changed")

    attribution_by_id = {str(row.get("record_id")): row for row in attributions}
    if set(attribution_by_id) != set(record_ids) or len(attributions) != len(records):
        raise ValueError("OASST2 attribution coverage changed")
    forbidden = {"messages", "raw_text", "text", "user_id"}
    for row in attributions:
        if row.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("OASST2 attribution schema changed")
        if forbidden.intersection(row):
            raise ValueError("OASST2 attribution leaked raw or user text")
        if row.get("source_object_sha256") != report["source"]["object_sha256"]:
            raise ValueError("OASST2 attribution object identity changed")
        if row.get("split") != tree_split(str(row.get("message_tree_id"))):
            raise ValueError("OASST2 tree split changed")
    if {str(row.get("record_id")) for row in selection_ledger} != set(record_ids):
        raise ValueError("OASST2 selection ledger coverage changed")
    if {str(row.get("record_id")) for row in token_ledger} != set(record_ids):
        raise ValueError("OASST2 token ledger coverage changed")
    train_trees = {row["message_tree_id"] for row in selection_ledger if row["split"] == "train"}
    validation_trees = {
        row["message_tree_id"] for row in selection_ledger if row["split"] == "validation"
    }
    if train_trees & validation_trees:
        raise ValueError("OASST2 selected trees overlap across splits")

    tree_selection = load_tree_candidates(
        private_root / "raw" / TREE_OBJECT,
        maximum_decoded_bytes=plan["limits"]["maximum_decoded_bytes"],
        maximum_messages=plan["limits"]["maximum_input_messages"],
        maximum_trees=plan["limits"]["maximum_input_trees"],
    )
    if tree_selection.metrics != report["curation"]["tree_selection"]:
        raise ValueError("OASST2 tree selection did not reproduce")
    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan, repository_root=root
    )
    structurally_accepted, replay_filter = filter_sft_records(
        tree_selection.records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if replay_filter != filter_decisions:
        raise ValueError("OASST2 global filter replay changed")
    near_accepted, _near_ledger, near_metrics = apply_conversation_near_dedup(
        structurally_accepted,
        metadata=tree_selection.metadata,
        near_duplicate_config=near_config,
    )
    if near_metrics != report["curation"]["near_duplicate"]:
        raise ValueError("OASST2 near dedup replay changed")
    replay_records, replay_tokens, replay_token_ledger = measure_and_select(
        near_accepted,
        maximum_train_supervised_tokens=plan["limits"]["maximum_retained_train_supervised_tokens"],
        metadata=tree_selection.metadata,
        target_task_train_supervised_tokens=plan["policy"]["target_task_train_supervised_tokens"],
        target_train_supervised_tokens=plan["limits"]["target_retained_train_supervised_tokens"],
        template=template,
        tokenizer=tokenizer,
        validation_fraction=plan["limits"]["validation_fraction"],
    )
    replay_records.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    if replay_records != records:
        raise ValueError("OASST2 selected records did not reproduce")
    if replay_tokens != token_metrics or replay_token_ledger != token_ledger:
        raise ValueError("OASST2 frozen token measurement did not reproduce")

    refiltered, final_filter = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if len(refiltered) != len(records) or final_filter["rejected"] != 0:
        raise ValueError("OASST2 final candidate failed global filters")
    rededuped, _final_near_ledger, final_near = apply_conversation_near_dedup(
        records,
        metadata=tree_selection.metadata,
        near_duplicate_config=near_config,
    )
    if len(rededuped) != len(records) or final_near["rejected"] != 0:
        raise ValueError("OASST2 final candidate contains near duplicates")

    expected_owner_sample = deterministic_owner_sample(
        records, maximum_records=plan["limits"]["owner_sample_records"]
    )
    if owner_sample != expected_owner_sample:
        raise ValueError("OASST2 owner sample did not reproduce")
    owner_html, sample_sha256 = build_oasst2_owner_review_html(owner_sample)
    if paths["owner_review_html"].read_text(encoding="utf-8") != owner_html:
        raise ValueError("OASST2 owner-review HTML changed")
    if report["owner_review"]["sample_sha256"] != sample_sha256:
        raise ValueError("OASST2 owner sample SHA-256 changed")

    verification = {
        "checks": {
            "apache_2_0_attribution_verified": True,
            "assistant_only_frozen_token_measurement_reproduced": True,
            "benchmark_decontamination_reproduced": True,
            "exact_object_inventory_verified": True,
            "exact_tree_and_message_boundaries_verified": True,
            "four_k_fit_reproduced": True,
            "global_filters_reproduced": True,
            "near_deduplication_reproduced": True,
            "owner_review_packet_pending_and_reproduced": True,
            "rank_aware_path_selection_reproduced": True,
            "raw_text_free_report_verified": True,
            "tree_disjoint_split_reproduced": True,
        },
        "curated": {
            "accepted_records": len(records),
            "owner_sample_records": len(owner_sample),
            "planning_target_reached": token_metrics["planning_target_reached"],
            "train_supervised_tokens": token_metrics["train_supervised_tokens"],
            "validation_supervised_tokens": token_metrics["validation_supervised_tokens"],
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "run_plan_file_sha256": file_sha256(plan_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_owner_review_pending",
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: OASST2 independent verification "
        f"records={len(records)} "
        f"train_supervised_tokens={token_metrics['train_supervised_tokens']} "
        f"target_reached={token_metrics['planning_target_reached']} "
        f"production_eligible=False verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
