#!/usr/bin/env python3
"""Independently verify the private DMath candidate and raw-text-free receipt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_dmath_curation import (
    SOURCE_OBJECT_PATH,
    SOURCE_ROWS,
    VERIFICATION_SCHEMA_VERSION,
    apply_prompt_near_dedup,
    file_sha256,
    load_dmath_curation_plan,
    load_runtime_components,
    sha256_value,
    validate_raw_text_free_report,
    write_json_atomic,
)
from lumi.posttraining_oasst1_full_run import measure_token_fit


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
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
    plan = load_dmath_curation_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (root / plan["output"]["raw_text_free_verification_path"]).resolve()
    if verification_path.exists():
        raise FileExistsError(
            f"refusing to overwrite DMath verification receipt: {verification_path}"
        )
    report = _load_json(report_path)
    validate_raw_text_free_report(report)
    private_report = _load_json(private_root / "raw-text-free-curation-report.json")
    if private_report != report:
        raise ValueError("private and Git DMath curation receipts differ")

    artifact_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_dedup_ledger": private_root / "curated" / "near-dedup-ledger.jsonl",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
        "verifier_ledger": private_root / "curated" / "verifier-ledger.jsonl",
    }
    reported_artifacts = {item["name"]: item for item in report["artifacts"]["objects"]}
    if set(reported_artifacts) != set(artifact_paths):
        raise ValueError("DMath artifact inventory names changed")
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"DMath private artifact is missing: {name}")
        item = reported_artifacts[name]
        if item["bytes"] != path.stat().st_size or item["sha256"] != file_sha256(path):
            raise ValueError(f"DMath private artifact changed: {name}")

    records = _load_jsonl(artifact_paths["accepted_records"])
    attributions = _load_jsonl(artifact_paths["attribution_manifest"])
    near_ledger = _load_jsonl(artifact_paths["near_dedup_ledger"])
    owner_sample = _load_jsonl(artifact_paths["owner_sample"])
    verifier_ledger = _load_jsonl(artifact_paths["verifier_ledger"])
    token_metrics = _load_json(artifact_paths["token_metrics"])
    filter_decisions = _load_json(artifact_paths["filter_decisions"])

    record_ids = [str(record.get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)) or "None" in record_ids:
        raise ValueError("DMath accepted record IDs are invalid or duplicated")
    if len(records) != report["curation"]["retained_records"]:
        raise ValueError("DMath retained record count changed")
    expected_top_level_fields = {
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
    for record in records:
        if set(record) != expected_top_level_fields:
            raise ValueError("DMath accepted record schema changed")
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError("DMath message sequence changed")
        source = record.get("source")
        if not isinstance(source, Mapping) or source.get("source_id") != "dmath":
            raise ValueError("DMath source binding changed")
        if source.get("document_id") in {None, ""}:
            raise ValueError("DMath source document ID is missing")
        if record.get("split") not in {"train", "validation"}:
            raise ValueError("DMath internal split changed")

    attribution_ids = [str(row.get("record_id")) for row in attributions]
    if attribution_ids != sorted(record_ids):
        raise ValueError("DMath attribution coverage changed")
    forbidden_attribution_fields = {
        "answer_en",
        "question_en",
        "solution_abst_en",
        "solution_code_en",
    }
    for row in attributions:
        if forbidden_attribution_fields.intersection(row):
            raise ValueError("DMath attribution leaked raw source fields")
        if row.get("excluded_training_target_fields") != [
            "solution_abst_en",
            "solution_code_en",
        ]:
            raise ValueError("DMath hidden-solution exclusion changed")

    if len(verifier_ledger) != SOURCE_ROWS:
        raise ValueError("DMath verifier ledger coverage changed")
    passed_ids = {
        str(row["record_id"])
        for row in verifier_ledger
        if row.get("verified") is True and row.get("reason") == "verified"
    }
    if not set(record_ids).issubset(passed_ids):
        raise ValueError("DMath candidate contains a verifier-failed record")
    verifier_reasons = Counter(str(row.get("reason")) for row in verifier_ledger)
    if dict(sorted(verifier_reasons.items())) != report["curation"]["verifier"]["reason_counts"]:
        raise ValueError("DMath verifier reason counts changed")

    if filter_decisions.get("report_sha256") != report["curation"]["filter"]["report_sha256"]:
        raise ValueError("DMath structural filter receipt changed")
    if len(near_ledger) != filter_decisions.get("accepted"):
        raise ValueError("DMath near-dedup ledger coverage changed")
    accepted_near_ids = {
        str(row["record_id"]) for row in near_ledger if row.get("accepted") is True
    }
    if set(record_ids) != accepted_near_ids:
        raise ValueError("DMath near-dedup accepted set changed")

    policy, benchmark, near_config, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    refiltered, final_filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=None,
    )
    if len(refiltered) != len(records) or final_filter_report["rejected"] != 0:
        raise ValueError("DMath accepted candidate no longer passes global filters")
    rededuped, _dedup_ledger, dedup_metrics = apply_prompt_near_dedup(
        records,
        near_duplicate_config=near_config,
    )
    if len(rededuped) != len(records) or dedup_metrics["rejected"] != 0:
        raise ValueError("DMath accepted candidate contains a prompt near-duplicate")
    remeasured, remeasured_metrics = measure_token_fit(
        records,
        template=template,
        tokenizer=tokenizer,
    )
    if len(remeasured) != len(records) or remeasured_metrics != token_metrics:
        raise ValueError("DMath frozen-tokenizer measurement changed")
    if token_metrics != report["curation"]["token_measurement"]:
        raise ValueError("DMath report token metrics changed")

    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("DMath owner sample count changed")
    if sha256_value(owner_sample) != report["owner_review"]["sample_sha256"]:
        raise ValueError("DMath owner sample SHA-256 changed")
    if not {str(row["record_id"]) for row in owner_sample}.issubset(set(record_ids)):
        raise ValueError("DMath owner sample escaped the candidate")
    owner_html = artifact_paths["owner_review_html"].read_text(encoding="utf-8")
    if "<script src=" in owner_html or "<link rel=" in owner_html:
        raise ValueError("DMath owner review gained an external dependency")
    if "lumi.dmath-owner-decisions.v0.1" not in owner_html:
        raise ValueError("DMath owner decision schema is missing from review page")

    raw_root = private_root / "raw"
    raw_files = {
        path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*") if path.is_file()
    }
    if raw_files != {"LICENSE", "README.md", SOURCE_OBJECT_PATH}:
        raise ValueError("DMath raw inventory changed or official test split appeared")

    verification = {
        "checks": {
            "all_candidate_records_verifier_passed": True,
            "all_global_filters_replayed": True,
            "artifact_hashes_verified": True,
            "assistant_only_loss_verified": True,
            "benchmark_decontamination_replayed": True,
            "cross_split_exact_and_prompt_near_duplicates": 0,
            "frozen_tokenizer_measurement_replayed": True,
            "hidden_solution_fields_used_as_training_targets": False,
            "official_test_split_accessed": False,
            "owner_sample_records": len(owner_sample),
            "source_registry_check_deferred_until_promotion": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "retained_records": len(records),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_owner_review_pending",
        "supervised_tokens": token_metrics["supervised_tokens"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    write_json_atomic(verification_path, verification)
    print(
        "PASS: independently verified DMath curation "
        f"records={len(records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
