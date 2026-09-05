#!/usr/bin/env python3
"""Independently verify the bounded OASST1 curation without exposing raw text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst1_curation import EXPECTED_OBJECTS
from lumi.posttraining_oasst1_full_run import (
    EXPECTED_DATA_COLUMNS,
    canonical_json_bytes,
    file_sha256,
    load_full_run_plan,
    load_runtime_components,
    measure_token_fit,
    sha256_value,
)

VERIFICATION_SCHEMA_VERSION = "lumi.sft-oasst1-full-curation-verification.v0.1"
FORBIDDEN_REPORT_KEYS = frozenset(
    {"messages", "prompt", "raw_text", "response", "text", "user_id"}
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            rows.append(value)
    return rows


def _assert_raw_text_free(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = set(value) & FORBIDDEN_REPORT_KEYS
        if forbidden:
            raise ValueError(f"raw-text field in report at {path}: {sorted(forbidden)}")
        for key, child in value.items():
            _assert_raw_text_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_raw_text_free(child, f"{path}[{index}]")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification report: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    report_path = args.report if args.report.is_absolute() else root / args.report
    output = args.output if args.output.is_absolute() else root / args.output
    if not output.resolve().is_relative_to((root / "evidence").resolve()):
        raise ValueError("OASST1 verification report must stay under evidence/")
    plan = load_full_run_plan(plan_path, repository_root=root)
    report = _load_json(report_path)
    private_root = (root / plan["output"]["private_artifact_root"]).resolve()

    claimed_report_sha = report.get("report_sha256")
    if hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ).hexdigest() != claimed_report_sha:
        raise ValueError("OASST1 report internal SHA-256 failed")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("OASST1 report plan binding failed")
    if (private_root / "raw-text-free-report.json").read_bytes() != report_path.read_bytes():
        raise ValueError("private and repository OASST1 reports differ")
    _assert_raw_text_free(report)

    acquisition = report["acquisition"]
    if acquisition.get("object_count") != 4 or acquisition.get("rows") != 88_838:
        raise ValueError("OASST1 acquisition object or row boundary changed")
    if acquisition.get("network_bytes", 0) > plan["limits"]["maximum_network_bytes"]:
        raise ValueError("OASST1 acquisition exceeded network cap")
    if acquisition.get("pyarrow_decoded_bytes", 0) > plan["limits"][
        "maximum_decoded_bytes"
    ]:
        raise ValueError("OASST1 acquisition exceeded decoded cap")
    source_inventory = {item["object_path"]: item for item in acquisition["objects"]}
    if set(source_inventory) != {item["path"] for item in EXPECTED_OBJECTS}:
        raise ValueError("OASST1 acquired object allowlist changed")
    parquet_physical_bytes = 0
    for specification in EXPECTED_OBJECTS:
        path = private_root / "raw" / specification["path"]
        inventory = source_inventory[specification["path"]]
        if path.stat().st_size != inventory["bytes"] or file_sha256(path) != inventory["sha256"]:
            raise ValueError(f"OASST1 source object hash failed: {specification['path']}")
        if specification["kind"] == "data":
            parquet_physical_bytes += path.stat().st_size
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != specification["rows"]:
                raise ValueError(f"OASST1 source row count failed: {specification['path']}")
            if frozenset(parquet.schema_arrow.names) != EXPECTED_DATA_COLUMNS:
                raise ValueError(f"OASST1 source schema failed: {specification['path']}")
    if parquet_physical_bytes != 41_596_430:
        raise ValueError("OASST1 parquet physical byte inventory changed")
    license_bytes = (private_root / "raw" / "LICENSE").read_bytes()
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise ValueError("OASST1 Apache-2.0 license evidence failed")

    artifact_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    report_artifacts = {item["name"]: item for item in report["artifacts"]["objects"]}
    if set(report_artifacts) != set(artifact_paths):
        raise ValueError("OASST1 curated artifact inventory changed")
    for name, path in artifact_paths.items():
        inventory = report_artifacts[name]
        if path.stat().st_size != inventory["bytes"] or file_sha256(path) != inventory["sha256"]:
            raise ValueError(f"OASST1 curated artifact hash failed: {name}")

    records = _read_jsonl(artifact_paths["accepted_records"])
    attributions = _read_jsonl(artifact_paths["attribution_manifest"])
    owner_sample = _read_jsonl(artifact_paths["owner_sample"])
    expected_count = report["curation"]["token_measurement"]["accepted_records"]
    if len(records) != expected_count or len(attributions) != expected_count:
        raise ValueError("OASST1 accepted or attribution count changed")
    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("OASST1 owner sample count changed")
    record_ids = {row["record_id"] for row in records}
    if record_ids != {row["record_id"] for row in attributions}:
        raise ValueError("OASST1 attribution coverage is incomplete")
    if not {row["record_id"] for row in owner_sample}.issubset(record_ids):
        raise ValueError("OASST1 owner sample escaped the accepted dataset")
    source_hashes = {item["sha256"] for item in acquisition["objects"]}
    for attribution in attributions:
        if set(attribution) & {"messages", "raw_text", "text", "user_id"}:
            raise ValueError("OASST1 attribution retained raw or user text")
        if attribution.get("source_object_sha256") not in source_hashes:
            raise ValueError("OASST1 attribution object hash changed")

    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    refiltered, refilter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=source_registry,
    )
    if len(refiltered) != len(records) or refilter_report["rejected"] != 0:
        raise ValueError("OASST1 accepted dataset failed independent filters")
    remeasured, token_metrics = measure_token_fit(
        records,
        template=template,
        tokenizer=tokenizer,
    )
    expected_token_metrics = dict(report["curation"]["token_measurement"])
    if expected_token_metrics.pop("overlength_rejected") != 1:
        raise ValueError("OASST1 original overlength rejection count changed")
    if token_metrics.pop("overlength_rejected") != 0:
        raise ValueError("OASST1 final accepted data still contains overlength rows")
    if len(remeasured) != len(records) or token_metrics != expected_token_metrics:
        raise ValueError("OASST1 token measurement did not reproduce")

    verification = {
        "checks": {
            "apache_2_0_attribution": True,
            "assistant_only_token_measurement_reproduced": True,
            "benchmark_decontamination_reproduced": True,
            "decoded_byte_cap_passed": True,
            "exact_four_object_inventory": True,
            "exact_and_normalized_deduplication_reproduced": True,
            "frozen_4096_fit_reproduced": True,
            "network_byte_cap_passed": True,
            "owner_sample_pending_and_bounded": True,
            "pii_and_secret_scan_reproduced": True,
            "raw_text_free_report_verified": True,
            "schema_and_source_registry_reproduced": True,
        },
        "curated": {
            "accepted_records": len(records),
            "owner_sample_records": len(owner_sample),
            "supervised_tokens": expected_token_metrics["supervised_tokens"],
        },
        "observed_at": "2026-08-06",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": claimed_report_sha,
        "run_plan_file_sha256": file_sha256(plan_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_owner_review_pending",
    }
    verification["verification_sha256"] = sha256_value(verification)
    _write_json_atomic(output, verification)
    print(
        "PASS: OASST1 independent verification "
        f"records={len(records)} supervised_tokens={expected_token_metrics['supervised_tokens']} "
        f"production_eligible=False verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

