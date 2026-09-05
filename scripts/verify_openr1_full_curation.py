#!/usr/bin/env python3
"""Independently verify the bounded non-production OpenR1 curation."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lumi.data_quality import MinHashNearDuplicateIndex
from lumi.posttraining_openr1_full_run import (
    DATA_OBJECTS,
    EXACT_OBJECTS,
    EXPECTED_POLICY,
    VERIFICATION_SCHEMA_VERSION,
    build_token_metrics,
    encode_record,
    expected_data_columns,
    file_sha256,
    load_full_run_plan,
    load_runtime_components,
    reconcile_openr1_pii_decision,
    report_sha256,
    sha256_value,
    source_registry_presplit_decision,
    validate_report_is_raw_text_free,
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
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _artifact_path(root: Path, object_path: str) -> Path:
    destination = (root / "raw" / object_path).resolve()
    if not destination.is_relative_to((root / "raw").resolve()):
        raise ValueError("OpenR1 source object escaped private root")
    return destination


def _verify_source_rows(
    *,
    private_root: Path,
    records_by_id: Mapping[str, Mapping[str, Any]],
    attributions: list[dict[str, Any]],
) -> int:
    by_object: dict[str, dict[int, dict[str, Any]]] = {}
    for attribution in attributions:
        object_path = str(attribution["source_object_path"])
        object_row = int(attribution["source_object_row_index"])
        by_object.setdefault(object_path, {})[object_row] = attribution
    if set(by_object) - set(DATA_OBJECTS):
        raise ValueError("OpenR1 attribution references an unlisted shard")

    verified = 0
    columns = [
        "problem",
        "solution",
        "source",
        "uuid",
        "generations",
        "is_reasoning_complete",
        "correctness_math_verify",
    ]
    for object_path, targets in sorted(by_object.items()):
        parquet = pq.ParquetFile(_artifact_path(private_root, object_path))
        row_offset = 0
        remaining = set(targets)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            rows = batch.to_pylist()
            for local_index, row in enumerate(rows):
                object_row = row_offset + local_index
                if object_row not in remaining:
                    continue
                attribution = targets[object_row]
                record = records_by_id[str(attribution["record_id"])]
                generation_index = int(attribution["generation_index"])
                generations = row["generations"]
                complete = row["is_reasoning_complete"]
                verifier = row["correctness_math_verify"]
                if not (
                    isinstance(generations, list)
                    and isinstance(complete, list)
                    and isinstance(verifier, list)
                    and len(generations) == len(complete) == len(verifier)
                    and 0 <= generation_index < len(generations)
                ):
                    raise ValueError("OpenR1 final record has invalid verifier vectors")
                if complete[generation_index] is not True or verifier[generation_index] is not True:
                    raise ValueError("OpenR1 final record is not verifier-passed and complete")
                trace = generations[generation_index]
                if trace == row["solution"]:
                    raise ValueError("OpenR1 final record used a reference solution")
                messages = record["messages"]
                if messages[0]["content"] != row["problem"] or messages[1]["content"] != trace:
                    raise ValueError("OpenR1 final record text does not bind to source row")
                if attribution["uuid"] != row["uuid"]:
                    raise ValueError("OpenR1 final record UUID changed")
                if attribution["upstream_problem_source"] != row["source"]:
                    raise ValueError("OpenR1 final record upstream source changed")
                remaining.remove(object_row)
                verified += 1
            row_offset += len(rows)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"OpenR1 source rows missing from shard: {object_path}")
    return verified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_full_run_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_artifact_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    output_path = (
        root / plan["output"]["raw_text_free_verification_path"]
    ).resolve()
    private_output = private_root / "raw-text-free-verification.json"
    if output_path.exists() or private_output.exists():
        raise FileExistsError("refusing to overwrite OpenR1 verification")

    report = _load_json(report_path)
    validate_report_is_raw_text_free(report)
    if report.get("report_sha256") != report_sha256(report):
        raise ValueError("OpenR1 report internal SHA-256 failed")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("OpenR1 report plan binding failed")
    private_report = private_root / "raw-text-free-report.json"
    if private_report.read_bytes() != report_path.read_bytes():
        raise ValueError("private and repository OpenR1 reports differ")

    acquisition = report["acquisition"]
    if acquisition.get("object_count") != 11 or acquisition.get("rows") != 93_733:
        raise ValueError("OpenR1 acquisition object or row boundary changed")
    if acquisition.get("network_bytes", 0) > plan["limits"]["maximum_network_bytes"]:
        raise ValueError("OpenR1 acquisition exceeded network cap")
    if acquisition.get("pyarrow_decoded_bytes", 0) > plan["limits"][
        "maximum_decoded_bytes"
    ]:
        raise ValueError("OpenR1 acquisition exceeded decoded cap")
    source_inventory = {
        str(item["object_path"]): item for item in acquisition["objects"]
    }
    if set(source_inventory) != {str(item["path"]) for item in EXACT_OBJECTS}:
        raise ValueError("OpenR1 acquired object allowlist changed")
    physical_data_bytes = 0
    source_hashes: dict[str, str] = {}
    total_rows = 0
    for specification in EXACT_OBJECTS:
        object_path = str(specification["path"])
        path = _artifact_path(private_root, object_path)
        inventory = source_inventory[object_path]
        if path.stat().st_size != inventory["bytes"] or file_sha256(path) != inventory[
            "sha256"
        ]:
            raise ValueError(f"OpenR1 source object hash failed: {object_path}")
        source_hashes[object_path] = str(inventory["sha256"])
        if specification["kind"] == "data":
            physical_data_bytes += path.stat().st_size
            parquet = pq.ParquetFile(path)
            total_rows += int(parquet.metadata.num_rows)
            if frozenset(parquet.schema_arrow.names) != expected_data_columns():
                raise ValueError(f"OpenR1 source schema failed: {object_path}")
    if physical_data_bytes != plan["limits"]["expected_data_download_bytes"]:
        raise ValueError("OpenR1 parquet physical byte inventory changed")
    if total_rows != plan["limits"]["maximum_source_rows"]:
        raise ValueError("OpenR1 source row inventory changed")

    artifact_paths = {
        "accepted_records": private_root / "curated" / "accepted-records.jsonl",
        "attribution_manifest": private_root / "curated" / "attribution-manifest.jsonl",
        "filter_decisions": private_root / "curated" / "filter-decisions.json",
        "near_duplicate_index": private_root
        / "curated"
        / "prompt-near-duplicate-index.sqlite",
        "owner_review_html": private_root / "review" / "owner-review-32.html",
        "owner_sample": private_root / "review" / "owner-sample.jsonl",
        "source_inventory": private_root / "curated" / "upstream-source-inventory.json",
        "token_metrics": private_root / "curated" / "token-metrics.json",
    }
    report_artifacts = {
        str(item["name"]): item for item in report["artifacts"]["objects"]
    }
    if set(report_artifacts) != set(artifact_paths):
        raise ValueError("OpenR1 curated artifact inventory changed")
    for name, path in artifact_paths.items():
        inventory = report_artifacts[name]
        if path.stat().st_size != inventory["bytes"] or file_sha256(path) != inventory[
            "sha256"
        ]:
            raise ValueError(f"OpenR1 curated artifact hash failed: {name}")

    records = _read_jsonl(artifact_paths["accepted_records"])
    attributions = _read_jsonl(artifact_paths["attribution_manifest"])
    owner_sample = _read_jsonl(artifact_paths["owner_sample"])
    token_metrics = _load_json(artifact_paths["token_metrics"])
    if not records or len(records) != len(attributions):
        raise ValueError("OpenR1 final candidate or attribution count changed")
    if len(owner_sample) != plan["limits"]["owner_sample_records"]:
        raise ValueError("OpenR1 owner sample count changed")
    record_ids = [str(row["record_id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("OpenR1 final candidate contains duplicate record IDs")
    records_by_id = {str(row["record_id"]): row for row in records}
    if set(record_ids) != {str(row["record_id"]) for row in attributions}:
        raise ValueError("OpenR1 attribution coverage is incomplete")
    if not {str(row["record_id"]) for row in owner_sample}.issubset(record_ids):
        raise ValueError("OpenR1 owner sample escaped final candidate")
    if sha256_value(owner_sample) != report["owner_review"]["sample_sha256"]:
        raise ValueError("OpenR1 owner sample SHA-256 changed")
    for attribution in attributions:
        object_path = str(attribution["source_object_path"])
        if attribution.get("source_object_sha256") != source_hashes.get(object_path):
            raise ValueError("OpenR1 attribution source-object hash changed")
        if attribution.get("rights_status") != "unresolved_quarantined":
            raise ValueError("OpenR1 unresolved rights case escaped quarantine")

    expected_validation = max(1, round(len(records) * 0.05))
    observed_validation = sum(row["split"] == "validation" for row in records)
    if observed_validation != expected_validation:
        raise ValueError("OpenR1 deterministic validation split changed")
    if {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("OpenR1 final candidate must contain both splits")

    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    normalized: set[str] = set()
    normalized_by_id: dict[str, str] = {}
    token_by_id: dict[str, dict[str, int]] = {}
    pii_reclassified = 0
    with tempfile.TemporaryDirectory(prefix="lumi-openr1-verify-") as temporary:
        near_index = MinHashNearDuplicateIndex(EXPECTED_POLICY["near_duplicate"])
        for record in records:
            presplit = dict(record)
            presplit["split"] = "train"
            decision = source_registry_presplit_decision(
                presplit,
                policy=policy,
                source_registry=source_registry,
                benchmark=benchmark,
            )
            accepted, classification = reconcile_openr1_pii_decision(
                presplit,
                decision,
            )
            if not accepted:
                raise ValueError(
                    f"OpenR1 final record failed independent filters: {decision.reasons}"
                )
            if classification is not None:
                pii_reclassified += 1
            if decision.normalized_sha256 in normalized:
                raise ValueError("OpenR1 final candidate has normalized duplicates")
            normalized.add(decision.normalized_sha256)
            normalized_by_id[str(record["record_id"])] = decision.normalized_sha256
            measured = encode_record(record, template=template, tokenizer=tokenizer)
            if measured is None:
                raise ValueError("OpenR1 final candidate contains a 4K overflow")
            token_by_id[str(record["record_id"])] = measured
            signature = near_index.signature(str(record["messages"][0]["content"]))
            if near_index.duplicate_of(signature) is not None:
                raise ValueError("OpenR1 final candidate has prompt near-duplicates")
            near_index.add(str(record["record_id"]), signature)
        near_index.close()
        if not Path(temporary).is_dir():
            raise AssertionError("verification temporary directory disappeared")

    reproduced_tokens = build_token_metrics(
        records,
        token_by_id,
        overlength_rejected=int(token_metrics["overlength_rejected"]),
    )
    if reproduced_tokens != token_metrics:
        raise ValueError("OpenR1 token metrics did not reproduce")
    if reproduced_tokens["supervised_tokens"] > plan["limits"][
        "maximum_retained_supervised_tokens"
    ]:
        raise ValueError("OpenR1 retained supervised-token cap failed")
    if sha256_value(sorted(normalized_by_id.values())) != report["curation"][
        "filter"
    ]["accepted_dataset_sha256"]:
        raise ValueError("OpenR1 accepted dataset SHA-256 did not reproduce")

    source_rows_verified = _verify_source_rows(
        private_root=private_root,
        records_by_id=records_by_id,
        attributions=attributions,
    )
    if source_rows_verified != len(records):
        raise ValueError("OpenR1 source-row verification did not cover final candidate")

    verification = {
        "checks": {
            "assistant_only_token_measurement_reproduced": True,
            "benchmark_decontamination_reproduced": True,
            "exact_11_object_inventory": True,
            "exact_and_normalized_deduplication_reproduced": True,
            "frozen_4096_fit_reproduced": True,
            "global_pii_filter_preserved": True,
            "network_and_decoded_caps_passed": True,
            "no_reference_solution_targets": True,
            "owner_sample_pending_and_bounded": True,
            "prompt_near_deduplication_reproduced": True,
            "raw_text_free_report_verified": True,
            "source_row_and_generation_bindings_verified": True,
            "unresolved_rights_quarantined": True,
            "verifier_pass_and_reasoning_complete_verified": True,
        },
        "curated": {
            "records": len(records),
            "owner_sample_records": len(owner_sample),
            "pii_reclassified_records": pii_reclassified,
            "source_rows_verified": source_rows_verified,
            "supervised_tokens": reproduced_tokens["supervised_tokens"],
            "train_records": len(records) - observed_validation,
            "validation_records": observed_validation,
        },
        "observed_at": "2026-08-06",
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "run_plan_file_sha256": file_sha256(plan_path),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonproduction_rights_and_owner_review_pending",
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_report_is_raw_text_free(verification)
    write_json_atomic(private_output, verification)
    write_json_atomic(output_path, verification)
    print(
        "PASS: OpenR1 independent verification "
        f"records={len(records)} supervised_tokens={reproduced_tokens['supervised_tokens']} "
        f"production_eligible=False verification={verification['verification_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
