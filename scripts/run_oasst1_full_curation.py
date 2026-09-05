#!/usr/bin/env python3
"""Run the exact owner-authorized CPU-only OASST1 acquisition and curation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst1_curation import EXPECTED_OBJECTS, select_oasst1_pairs
from lumi.posttraining_oasst1_full_run import (
    artifact_inventory,
    build_raw_text_free_report,
    build_sft_records,
    deterministic_owner_sample,
    download_exact_object,
    exact_object_url,
    file_sha256,
    inspect_parquet_object,
    load_full_run_plan,
    load_runtime_components,
    measure_token_fit,
    sha256_value,
    write_json_atomic,
    write_jsonl_atomic,
)


def _artifact_path(root: Path, object_path: str) -> Path:
    destination = (root / "raw" / object_path).resolve()
    if not destination.is_relative_to((root / "raw").resolve()):
        raise ValueError("OASST1 object path escaped the private artifact root")
    return destination


def _public_acquisition_manifest(
    downloads: list[dict[str, Any]],
    parquet: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata_logical_bytes": sum(
            item["metadata_logical_bytes"] for item in parquet.values()
        ),
        "network_bytes": sum(item["bytes"] for item in downloads),
        "object_count": len(downloads),
        "objects": downloads,
        "parquet": parquet,
        "pyarrow_decoded_bytes": sum(
            item["pyarrow_decoded_bytes"] for item in parquet.values()
        ),
        "rows": sum(item["rows"] for item in parquet.values()),
    }


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
    existing_files = {
        path.relative_to(private_root).as_posix()
        for path in private_root.rglob("*")
        if path.is_file()
    } if private_root.exists() else set()
    allowed_existing = {f"raw/{item['path']}" for item in EXPECTED_OBJECTS}
    if existing_files and existing_files != allowed_existing:
        raise FileExistsError(
            f"refusing unexpected OASST1 private-run inventory: {sorted(existing_files)}"
        )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite OASST1 report: {report_path}")
    private_root.mkdir(parents=True, exist_ok=True)

    limits = plan["limits"]
    downloads: list[dict[str, Any]] = []
    remaining_network = limits["maximum_network_bytes"]
    for specification in EXPECTED_OBJECTS:
        destination = _artifact_path(private_root, specification["path"])
        if existing_files:
            downloaded = {
                "bytes": destination.stat().st_size,
                "object_path": specification["path"],
                "redirect_host": "verified_local_resume",
                "requested_url": exact_object_url(specification["path"]),
                "sha256": file_sha256(destination),
            }
            if downloaded["bytes"] > remaining_network:
                raise ValueError("existing OASST1 objects exceed network byte limit")
        else:
            downloaded = download_exact_object(
                destination=destination,
                maximum_bytes=remaining_network,
                object_path=specification["path"],
            )
        remaining_network -= downloaded["bytes"]
        downloads.append(downloaded)
    if len(downloads) != limits["maximum_objects"]:
        raise ValueError("OASST1 exact object count changed")

    selections = {}
    parquet_metrics: dict[str, dict[str, Any]] = {}
    object_sha256_by_path = {item["object_path"]: item["sha256"] for item in downloads}
    total_rows = 0
    total_decoded = 0
    all_candidates = []
    for specification in EXPECTED_OBJECTS:
        if specification["kind"] != "data":
            continue
        wrappers, metrics = inspect_parquet_object(
            _artifact_path(private_root, specification["path"]),
            expected_decoded_bytes=specification["decoded_bytes"],
            expected_rows=specification["rows"],
        )
        split = specification["split"]
        selection = select_oasst1_pairs(
            wrappers,
            object_path=specification["path"],
            revision=plan["source"]["revision"],
            split=split,
        )
        selections[split] = selection
        all_candidates.extend(selection.candidates)
        parquet_metrics[split] = metrics
        total_rows += metrics["rows"]
        total_decoded += metrics["pyarrow_decoded_bytes"]
        del wrappers
    if total_rows != limits["maximum_source_rows"]:
        raise ValueError("OASST1 total row count changed")
    if total_decoded > limits["maximum_decoded_bytes"]:
        raise ValueError("OASST1 decoded byte limit exceeded")

    records, attributions = build_sft_records(
        all_candidates,
        object_sha256_by_path=object_sha256_by_path,
    )
    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=source_registry,
    )
    final_records, token_metrics = measure_token_fit(
        structurally_accepted,
        template=template,
        tokenizer=tokenizer,
    )
    final_ids = {record["record_id"] for record in final_records}
    final_attributions = [row for row in attributions if row["record_id"] in final_ids]
    final_attributions.sort(key=lambda row: row["record_id"])
    final_records.sort(key=lambda row: (row["split"], row["record_id"]))
    owner_sample = deterministic_owner_sample(
        final_records,
        maximum_records=limits["owner_sample_records"],
    )

    private_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    write_jsonl_atomic(private_paths["accepted_records"], final_records)
    write_jsonl_atomic(private_paths["attribution_manifest"], final_attributions)
    write_json_atomic(private_paths["filter_decisions"], filter_report)
    write_jsonl_atomic(private_paths["owner_sample"], owner_sample)
    write_json_atomic(private_paths["token_metrics"], token_metrics)
    inventory = artifact_inventory(private_paths)
    acquisition = _public_acquisition_manifest(downloads, parquet_metrics)
    report = build_raw_text_free_report(
        acquisition=acquisition,
        artifacts=inventory,
        filter_report=filter_report,
        owner_sample=owner_sample,
        plan_file_sha256=plan_sha256,
        selections=selections,
        token_metrics=token_metrics,
    )
    elapsed = time.monotonic() - started
    if elapsed > limits["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OASST1 curation exceeded the authorized runtime")
    report["runtime_seconds"] = round(elapsed, 3)
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    write_json_atomic(private_root / "raw-text-free-report.json", report)
    write_json_atomic(report_path, report)
    print(
        "PASS: OASST1 full CPU curation "
        f"rows={acquisition['rows']} selected={len(records)} "
        f"accepted={len(final_records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
