#!/usr/bin/env python3
"""Independently verify the local OpenR1 visible-final repair."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lumi.data_quality import MinHashNearDuplicateIndex
from lumi.posttraining_openr1_full_run import (
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
    VERIFICATION_SCHEMA_VERSION,
    VisibleFinalRepairRejection,
    build_visible_final_review_html,
    load_style_repair_plan,
    repair_record,
    resolve_binding,
    select_visible_final_review_sample,
    style_metrics,
    write_json_atomic,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_style_repair_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_artifact_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    output_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    private_output = private_root / "raw-text-free-verification.json"
    if output_path.exists() or private_output.exists():
        raise FileExistsError("refusing to overwrite OpenR1 visible-final verification")

    report = _load_json(report_path)
    validate_report_is_raw_text_free(report)
    if report.get("report_sha256") != report_sha256(report):
        raise ValueError("OpenR1 visible-final report SHA-256 failed")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("OpenR1 visible-final plan binding failed")
    if (private_root / "raw-text-free-report.json").read_bytes() != report_path.read_bytes():
        raise ValueError("private and repository visible-final reports differ")

    artifact_paths = {
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
    inventory = {str(item["name"]): item for item in report["artifacts"]["objects"]}
    if set(inventory) != set(artifact_paths):
        raise ValueError("OpenR1 visible-final artifact inventory changed")
    for name, path in artifact_paths.items():
        item = inventory[name]
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"OpenR1 visible-final artifact hash failed: {name}")

    bindings = plan["bindings"]
    source_records = _read_jsonl(resolve_binding(root, bindings["input_candidate"]))
    source_attributions = _read_jsonl(
        resolve_binding(root, bindings["attribution_manifest"])
    )
    source_sample = _read_jsonl(resolve_binding(root, bindings["input_owner_sample"]))
    attribution_by_id = {
        str(row["record_id"]): row for row in source_attributions
    }
    expected_records: list[dict[str, Any]] = []
    expected_hidden: list[dict[str, Any]] = []
    expected_attributions: list[dict[str, Any]] = []
    expected_by_parent: dict[str, dict[str, Any]] = {}
    expected_rejections: Counter[str] = Counter()
    for source_record in source_records:
        parent_id = str(source_record["record_id"])
        try:
            repaired, hidden, attribution = repair_record(
                source_record,
                attribution_by_id[parent_id],
            )
        except VisibleFinalRepairRejection as error:
            expected_rejections[str(error)] += 1
            continue
        expected_records.append(repaired)
        expected_hidden.append(hidden)
        expected_attributions.append(attribution)
        expected_by_parent[parent_id] = repaired
    expected_records.sort(key=lambda row: (row["split"], row["record_id"]))
    expected_hidden.sort(key=lambda row: row["visible_record_id"])
    expected_attributions.sort(key=lambda row: row["record_id"])
    observed_records = _read_jsonl(artifact_paths["accepted_records"])
    observed_hidden = _read_jsonl(artifact_paths["hidden_reasoning"])
    observed_attributions = _read_jsonl(artifact_paths["attribution_manifest"])
    if observed_records != expected_records:
        raise ValueError("OpenR1 visible-final records did not reproduce")
    if observed_hidden != expected_hidden:
        raise ValueError("OpenR1 hidden reasoning separation did not reproduce")
    if observed_attributions != expected_attributions:
        raise ValueError("OpenR1 visible-final attribution did not reproduce")
    rejection_metrics = _load_json(artifact_paths["rejection_metrics"])
    expected_rejection_metrics = {
        "reasons": dict(sorted(expected_rejections.items())),
        "rejected_records": sum(expected_rejections.values()),
        "retained_records": len(expected_records),
        "source_records": len(source_records),
    }
    if rejection_metrics != expected_rejection_metrics:
        raise ValueError("OpenR1 visible-final rejection metrics did not reproduce")
    if report.get("style_rejections") != {
        "reasons": expected_rejection_metrics["reasons"],
        "records": expected_rejection_metrics["rejected_records"],
    }:
        raise ValueError("OpenR1 visible-final report rejection binding failed")

    style = style_metrics(observed_records)
    if style != _load_json(artifact_paths["style_metrics"]):
        raise ValueError("OpenR1 visible-final style metrics did not reproduce")
    if any(
        style[field]
        for field in (
            "legacy_tex_delimiter_records",
            "unbalanced_display_math_records",
            "unbalanced_latex_brace_records",
            "visible_think_tag_records",
        )
    ):
        raise ValueError("OpenR1 visible-final style gate failed")

    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    token_by_id: dict[str, dict[str, int]] = {}
    normalized: set[str] = set()
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
    for record in observed_records:
        presplit = dict(record)
        presplit["split"] = "train"
        decision = source_registry_presplit_decision(
            presplit,
            policy=policy,
            source_registry=source_registry,
            benchmark=benchmark,
        )
        accepted, _ = reconcile_openr1_pii_decision(presplit, decision)
        if not accepted:
            raise ValueError("OpenR1 visible-final record failed independent filters")
        if decision.normalized_sha256 in normalized:
            raise ValueError("OpenR1 visible-final normalized duplicate reproduced")
        normalized.add(decision.normalized_sha256)
        measured = encode_record(record, template=template, tokenizer=tokenizer)
        if measured is None:
            raise ValueError("OpenR1 visible-final 4K overflow reproduced")
        token_by_id[str(record["record_id"])] = measured
        signature = near_index.signature(str(record["messages"][0]["content"]))
        if near_index.duplicate_of(signature) is not None:
            raise ValueError("OpenR1 visible-final prompt near-duplicate reproduced")
        near_index.add(str(record["record_id"]), signature)
    near_index.close()
    tokens = build_token_metrics(observed_records, token_by_id, overlength_rejected=0)
    if tokens != _load_json(artifact_paths["token_metrics"]):
        raise ValueError("OpenR1 visible-final token metrics did not reproduce")

    expected_sample_records, retained_source_sample_records = (
        select_visible_final_review_sample(
            source_sample,
            expected_by_parent,
            expected_records,
        )
    )
    _, expected_sample_sha256 = build_visible_final_review_html(expected_sample_records)
    observed_sample = _read_jsonl(artifact_paths["owner_sample"])
    if sha256_value(observed_sample) != expected_sample_sha256:
        raise ValueError("OpenR1 visible-final owner sample did not reproduce")
    if expected_sample_sha256 != report["owner_review"]["sample_sha256"]:
        raise ValueError("OpenR1 visible-final owner sample report binding failed")
    if (
        report["owner_review"]["retained_source_sample_records"]
        != retained_source_sample_records
    ):
        raise ValueError("OpenR1 visible-final sample replacement count changed")

    verification = {
        "checks": {
            "assistant_only_token_measurement_reproduced": True,
            "benchmark_and_pii_filters_reproduced": True,
            "hidden_reasoning_separation_reproduced": True,
            "malformed_visible_answer_quarantine_reproduced": True,
            "no_reference_solution_access": True,
            "owner_sample_pending_and_reproduced": True,
            "prompt_near_deduplication_reproduced": True,
            "raw_text_free_report_verified": True,
            "visible_final_transform_reproduced": True,
            "visible_latex_structure_reproduced": True,
            "visible_think_tags_absent": True,
        },
        "curated": {
            "hidden_reasoning_records": len(observed_hidden),
            "owner_sample_records": len(observed_sample),
            "records": len(observed_records),
            "style_rejected_records": sum(expected_rejections.values()),
            "supervised_tokens": tokens["supervised_tokens"],
        },
        "observed_at": "2026-08-06",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "run_plan_file_sha256": file_sha256(plan_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_visible_final_review_pending",
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_report_is_raw_text_free(verification)
    write_json_atomic(private_output, verification)
    write_json_atomic(output_path, verification)
    print(
        "PASS: OpenR1 visible-final independent verification "
        f"records={len(observed_records)} supervised_tokens={tokens['supervised_tokens']} "
        f"production_eligible=False verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
