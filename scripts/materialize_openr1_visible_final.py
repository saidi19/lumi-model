#!/usr/bin/env python3
"""Materialize the local non-production OpenR1 visible-final repair."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from lumi.data_quality import MinHashNearDuplicateIndex
from lumi.posttraining_openr1_full_run import (
    artifact_inventory,
    build_token_metrics,
    encode_record,
    file_sha256,
    load_runtime_components,
    reconcile_openr1_pii_decision,
    report_sha256,
    sha256_value,
    source_registry_presplit_decision,
    validate_report_is_raw_text_free,
)
from lumi.posttraining_openr1_style_repair import (
    EXPECTED_LIMITS,
    REPORT_SCHEMA_VERSION,
    VisibleFinalRepairRejection,
    build_visible_final_review_html,
    load_style_repair_plan,
    repair_record,
    resolve_binding,
    select_visible_final_review_sample,
    style_metrics,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_style_repair_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_artifact_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    if private_root.exists() or report_path.exists() or verification_path.exists():
        raise FileExistsError("refusing to overwrite OpenR1 visible-final artifacts")
    private_root.mkdir(parents=True)

    bindings = plan["bindings"]
    source_records = _read_jsonl(resolve_binding(root, bindings["input_candidate"]))
    source_attributions = _read_jsonl(
        resolve_binding(root, bindings["attribution_manifest"])
    )
    source_sample = _read_jsonl(resolve_binding(root, bindings["input_owner_sample"]))
    if len(source_records) != EXPECTED_LIMITS["maximum_records"]:
        raise ValueError("OpenR1 style-repair source record count changed")
    if len(source_attributions) != len(source_records):
        raise ValueError("OpenR1 style-repair attribution count changed")
    attribution_by_id = {
        str(row["record_id"]): row for row in source_attributions
    }
    if {str(row["record_id"]) for row in source_records} != set(attribution_by_id):
        raise ValueError("OpenR1 style-repair attribution coverage changed")

    repaired_records: list[dict[str, Any]] = []
    hidden_reasoning: list[dict[str, Any]] = []
    repaired_attributions: list[dict[str, Any]] = []
    repaired_by_parent: dict[str, dict[str, Any]] = {}
    repair_rejections: Counter[str] = Counter()
    for source_record in source_records:
        parent_id = str(source_record["record_id"])
        try:
            repaired, hidden, attribution = repair_record(
                source_record,
                attribution_by_id[parent_id],
            )
        except VisibleFinalRepairRejection as error:
            repair_rejections[str(error)] += 1
            continue
        repaired_records.append(repaired)
        hidden_reasoning.append(hidden)
        repaired_attributions.append(attribution)
        repaired_by_parent[parent_id] = repaired
    if len(repaired_records) + sum(repair_rejections.values()) != len(source_records):
        raise ValueError("OpenR1 style-repair accounting changed")
    if len(repaired_records) < 2_500:
        raise ValueError("OpenR1 style-repair retained too few records")

    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    normalized: set[str] = set()
    token_by_id: dict[str, dict[str, int]] = {}
    pii_reclassified = 0
    near_index = MinHashNearDuplicateIndex(
        {
            "num_perm": 128,
            "package_version": "2.0.0",
            "scheme": "affine32",
            "seed": 1,
            "shingle_tokens": 5,
            "threshold": 0.9,
        }
    )
    for record in repaired_records:
        presplit = dict(record)
        presplit["split"] = "train"
        decision = source_registry_presplit_decision(
            presplit,
            policy=policy,
            source_registry=source_registry,
            benchmark=benchmark,
        )
        accepted, classification = reconcile_openr1_pii_decision(presplit, decision)
        if not accepted:
            raise ValueError(
                f"OpenR1 visible final failed filters: {decision.reasons}"
            )
        if classification is not None:
            pii_reclassified += 1
        if decision.normalized_sha256 in normalized:
            raise ValueError("OpenR1 visible final has normalized duplicates")
        normalized.add(decision.normalized_sha256)
        measured = encode_record(record, template=template, tokenizer=tokenizer)
        if measured is None:
            raise ValueError("OpenR1 visible final exceeded 4K")
        token_by_id[str(record["record_id"])] = measured
        signature = near_index.signature(str(record["messages"][0]["content"]))
        if near_index.duplicate_of(signature) is not None:
            raise ValueError("OpenR1 visible final has prompt near-duplicates")
        near_index.add(str(record["record_id"]), signature)
    near_index.close()

    repaired_records.sort(key=lambda row: (row["split"], row["record_id"]))
    repaired_attributions.sort(key=lambda row: row["record_id"])
    hidden_reasoning.sort(key=lambda row: row["visible_record_id"])
    metrics = style_metrics(repaired_records)
    if any(
        metrics[field]
        for field in (
            "legacy_tex_delimiter_records",
            "unbalanced_display_math_records",
            "unbalanced_latex_brace_records",
            "visible_think_tag_records",
        )
    ):
        raise ValueError("OpenR1 visible-final style gate did not close")
    token_metrics = build_token_metrics(
        repaired_records,
        token_by_id,
        overlength_rejected=0,
    )
    sample_records, retained_source_sample_records = select_visible_final_review_sample(
        source_sample,
        repaired_by_parent,
        repaired_records,
    )
    review_html, sample_sha256 = build_visible_final_review_html(sample_records)
    review_payload = [
        {
            "language_mode": "english",
            "messages": [dict(message) for message in row["messages"]],
            "record_id": row["record_id"],
            "skill_id": "reasoning_math_science_visible_final",
            "split": row["split"],
        }
        for row in sample_records
    ]
    if sha256_value(review_payload) != sample_sha256:
        raise ValueError("OpenR1 repaired owner sample hash changed")

    private_paths = {
        "accepted_records": private_root / "curated" / "visible-final-records.jsonl",
        "attribution_manifest": private_root
        / "curated"
        / "visible-final-attribution.jsonl",
        "hidden_reasoning": private_root / "private" / "hidden-reasoning.jsonl",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "rejection_metrics": private_root / "curated" / "rejection-metrics.json",
        "style_metrics": private_root / "curated" / "style-metrics.json",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    write_jsonl_atomic(private_paths["accepted_records"], repaired_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], repaired_attributions)
    write_jsonl_atomic(private_paths["hidden_reasoning"], hidden_reasoning)
    write_jsonl_atomic(private_paths["owner_sample"], review_payload)
    write_text_atomic(private_paths["owner_review_html"], review_html)
    write_json_atomic(
        private_paths["rejection_metrics"],
        {
            "reasons": dict(sorted(repair_rejections.items())),
            "rejected_records": sum(repair_rejections.values()),
            "retained_records": len(repaired_records),
            "source_records": len(source_records),
        },
    )
    write_json_atomic(private_paths["style_metrics"], metrics)
    write_json_atomic(private_paths["token_metrics"], token_metrics)
    artifacts = artifact_inventory(
        private_paths,
        raw_text_names=frozenset(
            {
                "accepted_records",
                "attribution_manifest",
                "hidden_reasoning",
                "owner_review_html",
                "owner_sample",
            }
        ),
    )
    elapsed = time.monotonic() - started
    if elapsed > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OpenR1 visible-final repair exceeded runtime")
    report = {
        "artifacts": artifacts,
        "authorization": plan["authorization"],
        "candidate": {
            "hidden_reasoning_records": len(hidden_reasoning),
            "records": len(repaired_records),
            "supervised_tokens": token_metrics["supervised_tokens"],
            "train_records": sum(row["split"] == "train" for row in repaired_records),
            "validation_records": sum(
                row["split"] == "validation" for row in repaired_records
            ),
        },
        "checks": {
            "assistant_only_loss_verified": True,
            "benchmark_decontamination_reproduced": True,
            "global_pii_filter_preserved": True,
            "hidden_reasoning_excluded_from_sft_target": True,
            "malformed_visible_answers_quarantined": True,
            "no_reference_solution_access": True,
            "prompt_near_deduplication_reproduced": True,
            "visible_answer_4k_fit": True,
            "visible_latex_structure_valid": True,
            "visible_think_tags_removed": True,
        },
        "input": {
            "candidate_file_sha256": file_sha256(
                resolve_binding(root, bindings["input_candidate"])
            ),
            "curation_report_file_sha256": file_sha256(
                resolve_binding(root, bindings["curation_report"])
            ),
            "curation_verification_file_sha256": file_sha256(
                resolve_binding(root, bindings["curation_verification"])
            ),
            "records": len(source_records),
        },
        "observed_at": "2026-08-06",
        "owner_review": {
            "complete": False,
            "prior_raw_trace_decisions_carried_forward": False,
            "record_count": len(review_payload),
            "retained_source_sample_records": retained_source_sample_records,
            "sample_sha256": sample_sha256,
            "status": "corrected_visible_final_review_pending",
        },
        "pii_reclassified_records": pii_reclassified,
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "remaining_blockers": [
            "corrected_visible_final_owner_review_pending",
            "product_markdown_math_renderer_not_yet_verified",
            "upstream_problem_source_rights_map_unresolved",
            "candidate_source_registry_promotion_not_authorized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "runtime_seconds": round(elapsed, 3),
        "style_rejections": {
            "reasons": dict(sorted(repair_rejections.items())),
            "records": sum(repair_rejections.values()),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "repaired_nonproduction_visible_final_review_pending",
        "style_metrics": metrics,
        "token_metrics": token_metrics,
    }
    report["report_sha256"] = report_sha256(report)
    validate_report_is_raw_text_free(report)
    write_json_atomic(private_root / "raw-text-free-report.json", report)
    write_json_atomic(report_path, report)
    print(
        "PASS: OpenR1 visible-final repair "
        f"records={len(repaired_records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"hidden_reasoning_target=False owner_sample={len(review_payload)} "
        f"production_eligible=False report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
