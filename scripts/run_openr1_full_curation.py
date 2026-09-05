#!/usr/bin/env python3
"""Run the exact owner-authorized CPU-only OpenR1 acquisition and curation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lumi.data_quality import SQLiteMinHashNearDuplicateIndex
from lumi.posttraining_openr1_full_run import (
    DATA_OBJECTS,
    EXACT_OBJECTS,
    EXPECTED_POLICY,
    REPORT_SCHEMA_VERSION,
    artifact_inventory,
    build_token_metrics,
    deterministic_owner_sample,
    deterministic_split,
    download_exact_object,
    encode_record,
    exact_object_url,
    expected_data_columns,
    file_sha256,
    load_full_run_plan,
    load_runtime_components,
    reconcile_openr1_pii_decision,
    report_sha256,
    select_verified_generation,
    sha256_value,
    source_registry_presplit_decision,
    validate_report_is_raw_text_free,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from lumi.posttraining_owner_review import build_owner_review_html


def _artifact_path(root: Path, object_path: str) -> Path:
    destination = (root / "raw" / object_path).resolve()
    if not destination.is_relative_to((root / "raw").resolve()):
        raise ValueError("OpenR1 object path escaped the private artifact root")
    return destination


def _write_spool_row(handle: Any, value: dict[str, Any]) -> tuple[int, int]:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    offset = handle.tell()
    handle.write(payload)
    return offset, len(payload)


def _read_spool_row(handle: Any, *, offset: int, length: int) -> dict[str, Any]:
    handle.seek(offset)
    payload = handle.read(length)
    if len(payload) != length:
        raise ValueError("OpenR1 eligible spool truncated")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("OpenR1 eligible spool row is invalid")
    return value


def _assert_private_inventory_is_resumable(
    private_root: Path,
) -> set[str]:
    existing = {
        path.relative_to(private_root).as_posix()
        for path in private_root.rglob("*")
        if path.is_file()
    } if private_root.exists() else set()
    raw_allowlist = {f"raw/{item['path']}" for item in EXACT_OBJECTS}
    unexpected = existing - raw_allowlist
    if unexpected:
        raise FileExistsError(
            f"refusing unexpected OpenR1 private-run inventory: {sorted(unexpected)}"
        )
    return existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_full_run_plan(plan_path, repository_root=root)
    plan_sha256 = file_sha256(plan_path)
    private_root = (root / plan["output"]["private_artifact_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    verification_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    existing_files = _assert_private_inventory_is_resumable(private_root)
    if report_path.exists() or verification_path.exists():
        raise FileExistsError("refusing to overwrite OpenR1 report or verification")
    private_root.mkdir(parents=True, exist_ok=True)

    limits = plan["limits"]
    remaining_network = int(limits["maximum_network_bytes"])
    downloads: list[dict[str, Any]] = []
    for ordinal, specification in enumerate(EXACT_OBJECTS, start=1):
        object_path = str(specification["path"])
        destination = _artifact_path(private_root, object_path)
        relative = f"raw/{object_path}"
        if relative in existing_files:
            downloaded = {
                "bytes": destination.stat().st_size,
                "object_path": object_path,
                "redirect_host": "verified_local_resume",
                "requested_url": exact_object_url(object_path),
                "sha256": file_sha256(destination),
            }
        else:
            downloaded = download_exact_object(
                destination=destination,
                maximum_bytes=remaining_network,
                object_path=object_path,
            )
        if downloaded["bytes"] > remaining_network:
            raise ValueError("OpenR1 source objects exceed network byte limit")
        remaining_network -= int(downloaded["bytes"])
        downloads.append(downloaded)
        print(
            f"ACQUIRED {ordinal}/{len(EXACT_OBJECTS)} {object_path} "
            f"bytes={downloaded['bytes']} sha256={downloaded['sha256']}",
            flush=True,
        )
    if len(downloads) != limits["maximum_objects"]:
        raise ValueError("OpenR1 exact object count changed")
    data_physical_bytes = sum(
        int(item["bytes"])
        for item in downloads
        if item["object_path"] in set(DATA_OBJECTS)
    )
    if data_physical_bytes != limits["expected_data_download_bytes"]:
        raise ValueError("OpenR1 exact parquet byte inventory changed")

    object_sha256_by_path = {
        str(item["object_path"]): str(item["sha256"]) for item in downloads
    }
    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    spool_path = private_root / "work" / "eligible-records.jsonl"
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    if spool_path.exists():
        raise FileExistsError(f"refusing stale OpenR1 spool: {spool_path}")

    selection_rejections: Counter[str] = Counter()
    filter_rejections: Counter[str] = Counter()
    pii_reclassifications: Counter[str] = Counter()
    upstream_source_counts: Counter[str] = Counter()
    upstream_selected_counts: Counter[str] = Counter()
    generation_index_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_normalized: set[str] = set()
    decision_digest = hashlib.sha256()
    spool_index: list[dict[str, Any]] = []
    total_rows = 0
    decoded_bytes = 0
    selected_verified = 0
    structurally_accepted = 0
    overlength_rejected = 0
    parquet_inventory: list[dict[str, Any]] = []

    with spool_path.open("xb") as spool:
        global_row_index = 0
        for shard_ordinal, object_path in enumerate(DATA_OBJECTS, start=1):
            path = _artifact_path(private_root, object_path)
            parquet = pq.ParquetFile(path)
            if frozenset(parquet.schema_arrow.names) != expected_data_columns():
                raise ValueError(f"OpenR1 parquet schema changed: {object_path}")
            shard_rows = int(parquet.metadata.num_rows)
            shard_decoded = 0
            object_row_index = 0
            for batch in parquet.iter_batches(batch_size=32):
                shard_decoded += int(batch.nbytes)
                rows = batch.to_pylist()
                for row in rows:
                    if not isinstance(row, dict):
                        selection_rejections["invalid_parquet_row"] += 1
                        object_row_index += 1
                        global_row_index += 1
                        continue
                    upstream_source = row.get("source")
                    if isinstance(upstream_source, str) and upstream_source.strip():
                        upstream_source_counts[upstream_source] += 1
                    record, attribution, selection_reason = select_verified_generation(
                        row,
                        global_row_index=global_row_index,
                        object_path=object_path,
                        object_row_index=object_row_index,
                        object_sha256=object_sha256_by_path[object_path],
                    )
                    object_row_index += 1
                    global_row_index += 1
                    if selection_reason is not None:
                        selection_rejections[selection_reason] += 1
                        continue
                    assert record is not None and attribution is not None
                    selected_verified += 1
                    generation_index_counts[
                        str(attribution["generation_index"])
                    ] += 1
                    decision = source_registry_presplit_decision(
                        record,
                        policy=policy,
                        source_registry=source_registry,
                        benchmark=benchmark,
                    )
                    reconciled, reclassification = reconcile_openr1_pii_decision(
                        record,
                        decision,
                    )
                    reasons = list(decision.reasons)
                    if reconciled and reasons == ["pii_pattern"]:
                        reasons = []
                        assert reclassification is not None
                        pii_reclassifications[reclassification] += 1
                    record_id = str(record["record_id"])
                    if record_id in seen_ids:
                        reasons.append("duplicate_record_id")
                    if decision.normalized_sha256 in seen_normalized:
                        reasons.append("duplicate_normalized_conversation")
                    seen_ids.add(record_id)
                    seen_normalized.add(decision.normalized_sha256)
                    reasons = list(dict.fromkeys(reasons))
                    decision_row = {
                        "accepted": not reasons,
                        "normalized_sha256": decision.normalized_sha256,
                        "reasons": reasons,
                        "record_id": record_id,
                        "response_sha256": decision.response_sha256,
                    }
                    decision_digest.update(
                        json.dumps(
                            decision_row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8") + b"\n"
                    )
                    if reasons:
                        filter_rejections.update(reasons)
                        continue
                    structurally_accepted += 1
                    token_metrics = encode_record(
                        record,
                        template=template,
                        tokenizer=tokenizer,
                    )
                    if token_metrics is None:
                        overlength_rejected += 1
                        continue
                    upstream_selected_counts[
                        str(attribution["upstream_problem_source"])
                    ] += 1
                    priority = hashlib.sha256(
                        f"retain:{record_id}".encode()
                    ).hexdigest()
                    payload = {
                        "attribution": attribution,
                        "normalized_sha256": decision.normalized_sha256,
                        "record": record,
                        "token_metrics": token_metrics,
                    }
                    offset, length = _write_spool_row(spool, payload)
                    spool_index.append(
                        {
                            "length": length,
                            "offset": offset,
                            "priority": priority,
                            "record_id": record_id,
                            "supervised_tokens": token_metrics["supervised_tokens"],
                        }
                    )
                del rows
            if object_row_index != shard_rows:
                raise ValueError(f"OpenR1 shard row count changed: {object_path}")
            total_rows += shard_rows
            decoded_bytes += shard_decoded
            parquet_inventory.append(
                {
                    "columns": sorted(parquet.schema_arrow.names),
                    "object_path": object_path,
                    "physical_bytes": path.stat().st_size,
                    "pyarrow_decoded_bytes": shard_decoded,
                    "rows": shard_rows,
                    "sha256": object_sha256_by_path[object_path],
                }
            )
            print(
                f"CURATED shard={shard_ordinal}/{len(DATA_OBJECTS)} rows={shard_rows} "
                f"total_rows={total_rows} eligible_4k={len(spool_index)}",
                flush=True,
            )
        spool.flush()
        os.fsync(spool.fileno())

    if total_rows != limits["maximum_source_rows"]:
        raise ValueError("OpenR1 full-source row count changed")
    if decoded_bytes > limits["maximum_decoded_bytes"]:
        raise ValueError("OpenR1 decoded byte limit exceeded")
    if len(parquet_inventory) != limits["maximum_data_objects"]:
        raise ValueError("OpenR1 data-object count changed")

    near_index_path = private_root / "curated" / "prompt-near-duplicate-index.sqlite"
    near_index = SQLiteMinHashNearDuplicateIndex(
        EXPECTED_POLICY["near_duplicate"],
        path=near_index_path,
        table_prefix="openr1_prompt",
    )
    retained_records: list[dict[str, Any]] = []
    retained_attributions: list[dict[str, Any]] = []
    token_by_id: dict[str, dict[str, int]] = {}
    normalized_by_id: dict[str, str] = {}
    near_duplicate_rejected = 0
    token_cap_skipped = 0
    retained_supervised_tokens = 0
    with spool_path.open("rb") as spool:
        for ordinal, item in enumerate(
            sorted(spool_index, key=lambda row: row["priority"]),
            start=1,
        ):
            value = _read_spool_row(
                spool,
                offset=int(item["offset"]),
                length=int(item["length"]),
            )
            record = value["record"]
            record_id = str(record["record_id"])
            signature = near_index.signature(str(record["messages"][0]["content"]))
            if near_index.duplicate_of(signature) is not None:
                near_duplicate_rejected += 1
                continue
            supervised_tokens = int(item["supervised_tokens"])
            if (
                retained_supervised_tokens + supervised_tokens
                > limits["maximum_retained_supervised_tokens"]
            ):
                token_cap_skipped += 1
                continue
            near_index.add(record_id, signature)
            retained_supervised_tokens += supervised_tokens
            retained_records.append(record)
            retained_attributions.append(value["attribution"])
            token_by_id[record_id] = dict(value["token_metrics"])
            normalized_by_id[record_id] = str(value["normalized_sha256"])
            if ordinal % 500 == 0:
                near_index.flush()
    near_index.flush()
    near_index.close()
    if retained_supervised_tokens > limits["maximum_retained_supervised_tokens"]:
        raise ValueError("OpenR1 retained supervised-token cap exceeded")

    validation_count = deterministic_split(
        retained_records,
        float(EXPECTED_POLICY["validation_fraction"]),
    )
    split_by_id = {
        str(record["record_id"]): str(record["split"]) for record in retained_records
    }
    for attribution in retained_attributions:
        attribution["derived_split"] = split_by_id[str(attribution["record_id"])]
        attribution["rights_status"] = "unresolved_quarantined"
    retained_records.sort(key=lambda row: (row["split"], row["record_id"]))
    retained_attributions.sort(key=lambda row: row["record_id"])
    token_metrics = build_token_metrics(
        retained_records,
        token_by_id,
        overlength_rejected=overlength_rejected,
    )
    if token_metrics["supervised_tokens"] != retained_supervised_tokens:
        raise ValueError("OpenR1 retained token ledger did not reconcile")
    owner_sample = deterministic_owner_sample(
        retained_records,
        maximum_records=limits["owner_sample_records"],
    )
    owner_sample_sha256 = sha256_value(owner_sample)
    owner_html = build_owner_review_html(
        owner_sample,
        sample_sha256=owner_sample_sha256,
    )

    filter_report = {
        "accepted_dataset_sha256": sha256_value(
            sorted(normalized_by_id.values())
        ),
        "decision_ledger_sha256": decision_digest.hexdigest(),
        "eligible_4k_records": len(spool_index),
        "filter_rejection_counts": dict(sorted(filter_rejections.items())),
        "near_duplicate_prompt_rejected": near_duplicate_rejected,
        "overlength_rejected": overlength_rejected,
        "pii_reclassification_counts": dict(sorted(pii_reclassifications.items())),
        "raw_text_retained": False,
        "retained_records": len(retained_records),
        "retained_supervised_tokens": retained_supervised_tokens,
        "selected_verified_records": selected_verified,
        "selection_rejection_counts": dict(sorted(selection_rejections.items())),
        "source_rows": total_rows,
        "structurally_accepted_records": structurally_accepted,
        "token_cap_skipped": token_cap_skipped,
    }
    filter_report["report_sha256"] = sha256_value(filter_report)
    source_inventory = {
        "dataset_id": plan["source"]["dataset_id"],
        "declared_dataset_license": plan["source"]["license_id"],
        "generation_index_counts": dict(sorted(generation_index_counts.items())),
        "records_by_upstream_problem_source": dict(
            sorted(upstream_source_counts.items())
        ),
        "rights_resolution": "unresolved_quarantined_before_promotion",
        "selected_by_upstream_problem_source": dict(
            sorted(upstream_selected_counts.items())
        ),
        "source_revision": plan["source"]["revision"],
    }

    private_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_duplicate_index": near_index_path,
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "source_inventory": private_root / "curated" / "upstream-source-inventory.json",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    write_jsonl_atomic(private_paths["accepted_records"], retained_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], retained_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["owner_sample"], owner_sample)
    write_text_atomic(private_paths["owner_review_html"], owner_html)
    write_json_atomic(private_paths["source_inventory"], source_inventory)
    write_json_atomic(private_paths["token_metrics"], token_metrics)

    inventory = artifact_inventory(
        private_paths,
        raw_text_names=frozenset(
            {
                "accepted_records",
                "attribution_manifest",
                "owner_review_html",
                "owner_sample",
                "source_inventory",
            }
        ),
    )
    acquisition = {
        "data_physical_bytes": data_physical_bytes,
        "expected_metadata_decoded_bytes": limits["expected_decoded_data_bytes"],
        "network_bytes": sum(int(item["bytes"]) for item in downloads),
        "object_count": len(downloads),
        "objects": downloads,
        "parquet": parquet_inventory,
        "pyarrow_decoded_bytes": decoded_bytes,
        "rows": total_rows,
    }
    elapsed = time.monotonic() - started
    if elapsed > limits["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OpenR1 curation exceeded the authorized runtime")
    report = {
        "acquisition": acquisition,
        "artifacts": inventory,
        "authorization": plan["authorization"],
        "curation": {
            "filter": filter_report,
            "token_measurement": token_metrics,
            "validation_records": validation_count,
        },
        "observed_at": "2026-08-06",
        "owner_review": {
            "complete": False,
            "record_count": len(owner_sample),
            "sample_sha256": owner_sample_sha256,
            "status": "pending_owner_review",
        },
        "plan_file_sha256": plan_sha256,
        "production": {
            "candidate_source_registry_changed": False,
            "dataset_promotion_authorized": False,
            "dataset_seal_authorized": False,
            "production_eligible": False,
            "sft_training_authorized": False,
        },
        "raw_training_text_present": False,
        "remaining_blockers": [
            "upstream_problem_source_rights_map_unresolved",
            "deterministic_owner_quality_sample_not_reviewed",
            "candidate_source_registry_promotion_not_authorized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "runtime_seconds": round(elapsed, 3),
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": plan["source"]["dataset_id"],
            "declared_license_id": plan["source"]["license_id"],
            "revision": plan["source"]["revision"],
            "source_id": plan["source"]["source_id"],
            "upstream_problem_source_inventory_sha256": file_sha256(
                private_paths["source_inventory"]
            ),
        },
        "status": "curated_nonproduction_rights_and_owner_review_pending",
    }
    report["report_sha256"] = report_sha256(report)
    validate_report_is_raw_text_free(report)
    write_json_atomic(private_root / "raw-text-free-report.json", report)
    write_json_atomic(report_path, report)
    spool_path.unlink()
    print(
        "PASS: OpenR1 full CPU curation "
        f"rows={total_rows} verified={selected_verified} "
        f"retained={len(retained_records)} supervised_tokens={retained_supervised_tokens} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"report={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
