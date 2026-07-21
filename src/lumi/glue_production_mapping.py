"""Fail-closed release, authorization, and checkpoints for Glue mapping.

The production source objects are already sealed as immutable zstd chunks in
private S3.  This module binds the next paid phase to those exact chunks and to
the frozen Lumi tokenizer/filter contracts.  It never calls AWS itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from .emr_serverless_production import build_emr_production_plan
from .glue_canary import (
    _artifact,
    _valid_commit,
    _validate_lingua_wheel_pin,
    _write_bytes_atomic,
    _zip_code,
)
from .glue_production import _read_contract
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

SCHEMA_VERSION = 1
RELEASE_FILE = "glue-production-map-release-v0.1.json"
AUTHORIZATION_FILE = "glue-production-map-authorization-v0.1.json"
CODE_ARCHIVE = "lumi-glue-production-map-code-v0.1.zip"
JOB_SCRIPT = "run_glue_production_mapping.py"
PACKAGE_MANIFEST = "manifest.sha256"
MAPPING_OWNER_APPROVAL_TEXT = (
    "I approve one additional replacement 100-DPU AWS Glue mapping run after the "
    "staged-manifest digest handoff fix, under the same total 150B, 72-hour/$3,168 "
    "maximum, with no dedup, sharding, GPU, or model training."
)


def _fingerprint(value: Mapping[str, Any], *excluded: str) -> str:
    material = {key: row for key, row in value.items() if key not in set(excluded)}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _mapping_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "hard_maximum_dpu": 100,
        "maximum_runtime_minutes": 4_320,
        "number_of_workers": 100,
        "soft_stop_minutes": 3_960,
        "target_total_candidate_tokens": 150_000_000_000,
        "wave_maximum_chunks": 500,
        "worker_type": "G.1X",
    }
    if contract.get("mapping") != expected:
        raise ValueError("Glue production mapping envelope changed")
    if contract.get("subsequent_phases") != {
        "dedup_maximum_dpu": 512,
        "mapping_maximum_dpu": 100,
        "sharding_maximum_dpu": 512,
    }:
        raise ValueError("Glue production phase capacity changed")
    return expected


def _validate_stage_evidence(
    stage_release: Mapping[str, Any],
    stage_result: Mapping[str, Any],
    stage_verification: Mapping[str, Any],
    *,
    stage_result_sha256: str,
) -> None:
    from .glue_production import validate_glue_production_stage_release

    validate_glue_production_stage_release(dict(stage_release))
    expected = {
        "gate": "lumi_glue_production_stage_result_v0.1",
        "release_fingerprint": stage_release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": stage_release["source_commit"],
        "source_files": len(stage_release["sources"]),
        "status": "pass",
        "training_started": False,
    }
    for name, value in expected.items():
        if stage_result.get(name) != value:
            raise ValueError(f"Glue production stage result changed {name}")
    rows = stage_result.get("sources")
    if not isinstance(rows, list) or len(rows) != len(stage_release["sources"]):
        raise ValueError("Glue production stage result source inventory changed")
    for source, staged in zip(stage_release["sources"], rows, strict=True):
        identity = {
            "category": source["category"],
            "component": source["component"],
            "source_bytes": int(source["source_bytes"]),
            "source_file": source["source_file"],
            "source_ordinal": int(source["ordinal"]),
            "source_sha256": source["source_sha256"],
            "start_after_line": int(source["start_after_line"]),
        }
        for name, value in identity.items():
            if staged.get(name) != value:
                raise ValueError(f"Glue production staged source changed {name}")
        if staged.get("source_complete") is not True:
            raise ValueError("Glue production staged source is incomplete")
        if int(staged.get("chunk_count", 0)) < 1:
            raise ValueError("Glue production staged source has no chunks")
        _valid_sha256(staged.get("manifest_sha256"), "staged manifest")
        key = staged.get("manifest_s3_key")
        if not isinstance(key, str) or ".." in key or not key.endswith(".json"):
            raise ValueError("Glue production staged manifest key is unsafe")
    aggregate = {
        "source_files": len(rows),
        "staged_chunks": sum(int(row["chunk_count"]) for row in rows),
        "staged_compressed_bytes": sum(
            int(row["staged_compressed_bytes"]) for row in rows
        ),
        "staged_rows": sum(int(row["staged_rows"]) for row in rows),
        "staged_uncompressed_bytes": sum(
            int(row["staged_uncompressed_bytes"]) for row in rows
        ),
    }
    if any(stage_result.get(name) != value for name, value in aggregate.items()):
        raise ValueError("Glue production stage aggregate totals changed")
    verification_expected = {
        "gate": "lumi_glue_production_stage_verification_v0.1",
        "release_fingerprint": stage_release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": stage_release["source_commit"],
        "stage_result_sha256": stage_result_sha256,
        "status": "pass",
        "training_started": False,
    }
    for name, value in verification_expected.items():
        if stage_verification.get(name) != value:
            raise ValueError(f"Glue production stage verification changed {name}")
    remote = stage_verification.get("remote_objects", {})
    if remote != {
        "chunk_objects": aggregate["staged_chunks"],
        "compressed_bytes": aggregate["staged_compressed_bytes"],
        "manifest_objects": aggregate["source_files"],
        "rows": aggregate["staged_rows"],
        "uncompressed_bytes": aggregate["staged_uncompressed_bytes"],
    }:
        raise ValueError("Glue production remote-object verification totals changed")


def _mapping_targets(plan: Mapping[str, Any], total_target: int) -> list[dict[str, Any]]:
    """Allocate the exact remaining 150B ceiling across the frozen categories."""

    accepted_total = sum(int(row["accepted_tokens"]) for row in plan["categories"])
    remaining_total = total_target - accepted_total
    if remaining_total <= 0:
        raise ValueError("Glue production mapping target is already complete")
    weights = [
        max(0, int(row["floor_tokens"]) - int(row["accepted_tokens"]))
        for row in plan["categories"]
    ]
    weight_total = sum(weights)
    if weight_total <= 0:
        raise ValueError("Glue production mapping categories have no remaining weight")
    allocations: list[int] = []
    remainders: list[tuple[Decimal, int]] = []
    for index, weight in enumerate(weights):
        exact = Decimal(remaining_total) * Decimal(weight) / Decimal(weight_total)
        floor = int(exact.to_integral_value(rounding=ROUND_FLOOR))
        allocations.append(floor)
        remainders.append((exact - Decimal(floor), index))
    missing = remaining_total - sum(allocations)
    for _, index in sorted(remainders, key=lambda row: (-row[0], row[1]))[:missing]:
        allocations[index] += 1
    rows = []
    for category, mapping_target in zip(plan["categories"], allocations, strict=True):
        rows.append(
            {
                "accepted_tokens": int(category["accepted_tokens"]),
                "category": str(category["category"]),
                "clean_floor_tokens": int(category["floor_tokens"]),
                "mapping_target_tokens": mapping_target,
                "ordinal": int(category["ordinal"]),
            }
        )
    if sum(row["mapping_target_tokens"] for row in rows) + accepted_total != total_target:
        raise RuntimeError("Glue mapping target allocation does not conserve 150B")
    return rows


def _existing_artifact(plan: Mapping[str, Any], name: str) -> dict[str, Any]:
    pin = plan["artifacts"][name]
    key = plan["storage"]["artifacts"][name]
    return _artifact(
        bucket=str(plan["storage"]["bucket"]),
        key=str(key),
        sha256=str(pin["sha256"]),
        size=int(pin["bytes"]),
        state="existing_verified_production_object",
    )


def _manifest(directory: Path, paths: list[Path]) -> None:
    body = "".join(f"{_sha256_file(path)}  {path.name}\n" for path in paths)
    _write_bytes_atomic(directory / PACKAGE_MANIFEST, body.encode())


def build_initial_mapping_checkpoint(release: Mapping[str, Any]) -> dict[str, Any]:
    categories = [
        {
            "category": row["category"],
            "closed": False,
            "cutoff": None,
            "mapped_documents": 0,
            "mapped_tokens": 0,
            "next_chunk_index": 0,
            "target_tokens": int(row["mapping_target_tokens"]),
        }
        for row in release["categories"]
    ]
    checkpoint = {
        "categories": categories,
        "checkpoint_sequence": 0,
        "gate": "lumi_glue_production_map_checkpoint_v0.1",
        "mapped_parquet_bytes": 0,
        "outputs": [],
        "previous_checkpoint_sha256": None,
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "status": "mapping",
        "total_mapped_documents": 0,
        "total_mapped_tokens": 0,
    }
    checkpoint["checkpoint_sha256"] = _fingerprint(
        checkpoint, "checkpoint_sha256"
    )
    return validate_mapping_checkpoint(checkpoint, release)


def validate_mapping_checkpoint(
    checkpoint: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production map checkpoint schema")
    if checkpoint.get("gate") != "lumi_glue_production_map_checkpoint_v0.1":
        raise ValueError("unexpected Glue production map checkpoint gate")
    if checkpoint.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue production map checkpoint targets another release")
    if checkpoint.get("checkpoint_sha256") != _fingerprint(
        checkpoint, "checkpoint_sha256"
    ):
        raise ValueError("Glue production map checkpoint fingerprint changed")
    sequence = _non_negative_int(
        checkpoint.get("checkpoint_sequence"), "map checkpoint sequence"
    )
    previous = checkpoint.get("previous_checkpoint_sha256")
    if sequence == 0 and previous is not None:
        raise ValueError("initial Glue map checkpoint cannot have a parent")
    if sequence > 0:
        _valid_sha256(previous, "previous Glue map checkpoint")
    outputs = checkpoint.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != sequence:
        raise ValueError("Glue production map output ledger changed")
    expected = {row["category"]: row for row in release["categories"]}
    available_chunks = {
        row["category"]: int(row["staged_chunks"])
        for row in release["category_inventory"]
    }
    categories = checkpoint.get("categories")
    if not isinstance(categories, list) or len(categories) != len(expected):
        raise ValueError("Glue production map categories changed")
    documents = 0
    tokens = 0
    all_closed = True
    for row in categories:
        if not isinstance(row, Mapping) or row.get("category") not in expected:
            raise ValueError("Glue production map category is invalid")
        name = str(row["category"])
        target = _non_negative_int(row.get("target_tokens"), "map category target")
        if target != int(expected[name]["mapping_target_tokens"]):
            raise ValueError("Glue production map category target changed")
        mapped_tokens = _non_negative_int(row.get("mapped_tokens"), "mapped tokens")
        mapped_documents = _non_negative_int(
            row.get("mapped_documents"), "mapped documents"
        )
        next_chunk = _non_negative_int(
            row.get("next_chunk_index"), "map next chunk index"
        )
        if next_chunk > available_chunks[name]:
            raise ValueError("Glue production map cursor exceeds staged chunks")
        closed = row.get("closed")
        if not isinstance(closed, bool):
            raise ValueError("Glue production map category closure is invalid")
        if mapped_tokens > target:
            raise ValueError("Glue production map exceeds its 150B allocation")
        if closed:
            shortfall = target - mapped_tokens
            if shortfall > int(release["limits"]["maximum_category_shortfall_tokens"]):
                raise ValueError("Glue production map category closed too early")
            cutoff = row.get("cutoff")
            if cutoff is not None and not isinstance(cutoff, Mapping):
                raise ValueError("Glue production map cutoff is invalid")
        else:
            all_closed = False
            if row.get("cutoff") is not None:
                raise ValueError("open Glue production map category has a cutoff")
            if next_chunk == available_chunks[name]:
                raise ValueError("Glue production map exhausted a category before closure")
        tokens += mapped_tokens
        documents += mapped_documents
    if tokens != checkpoint.get("total_mapped_tokens"):
        raise ValueError("Glue production map token conservation failed")
    if documents != checkpoint.get("total_mapped_documents"):
        raise ValueError("Glue production map document conservation failed")
    if tokens + int(release["resume"]["accepted_tokens"]) > int(
        release["limits"]["target_total_candidate_tokens"]
    ):
        raise ValueError("Glue production map exceeds the 150B hard stop")
    parquet_bytes = _non_negative_int(
        checkpoint.get("mapped_parquet_bytes"), "mapped Parquet bytes"
    )
    if parquet_bytes > int(release["limits"]["maximum_mapped_parquet_bytes"]):
        raise ValueError("Glue production map exceeds its S3 storage envelope")
    status = checkpoint.get("status")
    if status not in {"mapping", "paused", "ready_for_global_dedup"}:
        raise ValueError("Glue production map checkpoint status is invalid")
    if (status == "ready_for_global_dedup") != all_closed:
        raise ValueError("Glue production map completion differs from category closures")
    seen: set[str] = set()
    output_tokens = 0
    output_documents = 0
    output_bytes = 0
    for index, output in enumerate(outputs, start=1):
        if not isinstance(output, Mapping) or output.get("wave_sequence") != index:
            raise ValueError("Glue production map output sequence changed")
        digest = _valid_sha256(output.get("manifest_sha256"), "map wave manifest")
        if digest in seen:
            raise ValueError("Glue production map repeats a wave manifest")
        seen.add(digest)
        output_tokens += _non_negative_int(output.get("candidate_tokens"), "wave tokens")
        output_documents += _non_negative_int(
            output.get("candidate_documents"), "wave documents"
        )
        output_bytes += _non_negative_int(output.get("parquet_bytes"), "wave bytes")
    if (output_tokens, output_documents, output_bytes) != (
        tokens,
        documents,
        parquet_bytes,
    ):
        raise ValueError("Glue production map output ledger does not conserve totals")
    return dict(checkpoint)


def plan_mapping_wave(
    checkpoint: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    current = validate_mapping_checkpoint(checkpoint, release)
    if current["status"] != "mapping":
        raise ValueError("Glue production map checkpoint is not accepting work")
    limit = int(release["job"]["wave_maximum_chunks"])
    inventory = {
        row["category"]: int(row["staged_chunks"])
        for row in release["category_inventory"]
    }
    rows = {
        row["category"]: row for row in current["categories"] if not row["closed"]
    }
    allocations = {name: 0 for name in rows}
    order = [row["category"] for row in release["categories"] if row["category"] in rows]
    while sum(allocations.values()) < limit:
        progressed = False
        for name in order:
            if sum(allocations.values()) >= limit:
                break
            cursor = int(rows[name]["next_chunk_index"]) + allocations[name]
            if cursor >= inventory[name]:
                continue
            allocations[name] += 1
            progressed = True
        if not progressed:
            break
    work = []
    for name in order:
        count = allocations[name]
        if not count:
            continue
        start = int(rows[name]["next_chunk_index"])
        work.append(
            {
                "category": name,
                "end_chunk_index": start + count,
                "remaining_target_tokens": int(rows[name]["target_tokens"])
                - int(rows[name]["mapped_tokens"]),
                "start_chunk_index": start,
            }
        )
    if not work:
        raise RuntimeError("Glue production map has no staged chunks left to schedule")
    wave = {
        "categories": work,
        "chunk_count": sum(
            row["end_chunk_index"] - row["start_chunk_index"] for row in work
        ),
        "gate": "lumi_glue_production_map_wave_plan_v0.1",
        "parent_checkpoint_sha256": current["checkpoint_sha256"],
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "wave_sequence": int(current["checkpoint_sequence"]) + 1,
    }
    wave["wave_sha256"] = _fingerprint(wave, "wave_sha256")
    return wave


def advance_mapping_checkpoint(
    checkpoint: Mapping[str, Any],
    release: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    current = validate_mapping_checkpoint(checkpoint, release)
    material = dict(manifest)
    observed = material.pop("manifest_sha256", None)
    if _valid_sha256(observed, "Glue map wave manifest") != hashlib.sha256(
        _canonical_json_bytes(material)
    ).hexdigest():
        raise ValueError("Glue production map wave manifest fingerprint changed")
    expected = {
        "gate": "lumi_glue_production_map_wave_result_v0.1",
        "parent_checkpoint_sha256": current["checkpoint_sha256"],
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "wave_sequence": int(current["checkpoint_sequence"]) + 1,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"Glue production map wave changed {name}")
    planned = plan_mapping_wave(current, release)
    if manifest.get("wave_plan_sha256") != planned["wave_sha256"]:
        raise ValueError("Glue production map wave used another input plan")
    results = manifest.get("categories")
    if not isinstance(results, list) or [row.get("category") for row in results] != [
        row["category"] for row in planned["categories"]
    ]:
        raise ValueError("Glue production map wave categories changed")
    by_name = {row["category"]: dict(row) for row in current["categories"]}
    wave_tokens = 0
    wave_documents = 0
    wave_bytes = 0
    for work, result in zip(planned["categories"], results, strict=True):
        name = work["category"]
        for field in ("start_chunk_index", "end_chunk_index"):
            if result.get(field) != work[field]:
                raise ValueError(f"Glue production map wave changed {field}")
        available_tokens = _non_negative_int(
            result.get("available_tokens"), "wave available tokens"
        )
        selected_tokens = _non_negative_int(
            result.get("candidate_tokens"), "wave candidate tokens"
        )
        available_documents = _non_negative_int(
            result.get("available_documents"), "wave available documents"
        )
        selected_documents = _non_negative_int(
            result.get("candidate_documents"), "wave candidate documents"
        )
        if selected_tokens > available_tokens or selected_documents > available_documents:
            raise ValueError("Glue production map selected more than its wave input")
        closed = result.get("closed")
        if not isinstance(closed, bool):
            raise ValueError("Glue production map wave closure is invalid")
        remaining = int(work["remaining_target_tokens"])
        if selected_tokens > remaining:
            raise ValueError("Glue production map wave exceeded the 150B hard stop")
        if not closed and (
            selected_tokens != available_tokens
            or selected_documents != available_documents
        ):
            raise ValueError("open Glue production map wave discarded candidates")
        if closed and remaining - selected_tokens > int(
            release["limits"]["maximum_category_shortfall_tokens"]
        ):
            raise ValueError("Glue production map wave closed too far below target")
        parquet_bytes = _non_negative_int(result.get("parquet_bytes"), "wave Parquet bytes")
        uri = result.get("parquet_s3_uri")
        if selected_tokens:
            prefix = (
                f"s3://{release['storage']['bucket']}/"
                f"{release['output']['mapped_prefix']}/{release['release_fingerprint']}/"
            )
            if not isinstance(uri, str) or not uri.startswith(prefix):
                raise ValueError("Glue production map wave left its bounded namespace")
            _valid_sha256(result.get("objects_sha256"), "wave object inventory")
        elif uri is not None or parquet_bytes:
            raise ValueError("empty Glue production map wave published Parquet")
        row = by_name[name]
        row["mapped_tokens"] += selected_tokens
        row["mapped_documents"] += selected_documents
        row["next_chunk_index"] = int(work["end_chunk_index"])
        row["closed"] = closed
        row["cutoff"] = result.get("cutoff") if closed else None
        wave_tokens += selected_tokens
        wave_documents += selected_documents
        wave_bytes += parquet_bytes
    if manifest.get("candidate_tokens") != wave_tokens:
        raise ValueError("Glue production map wave token aggregate changed")
    if manifest.get("candidate_documents") != wave_documents:
        raise ValueError("Glue production map wave document aggregate changed")
    if manifest.get("parquet_bytes") != wave_bytes:
        raise ValueError("Glue production map wave byte aggregate changed")
    output = {
        "candidate_documents": wave_documents,
        "candidate_tokens": wave_tokens,
        "manifest_s3_key": manifest.get("manifest_s3_key"),
        "manifest_sha256": observed,
        "parquet_bytes": wave_bytes,
        "wave_sequence": expected["wave_sequence"],
    }
    if not isinstance(output["manifest_s3_key"], str):
        raise ValueError("Glue production map wave manifest key is missing")
    categories = [by_name[row["category"]] for row in release["categories"]]
    all_closed = all(row["closed"] for row in categories)
    result = {
        **current,
        "categories": categories,
        "checkpoint_sequence": expected["wave_sequence"],
        "mapped_parquet_bytes": int(current["mapped_parquet_bytes"]) + wave_bytes,
        "outputs": [*current["outputs"], output],
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "status": "ready_for_global_dedup" if all_closed else "mapping",
        "total_mapped_documents": int(current["total_mapped_documents"])
        + wave_documents,
        "total_mapped_tokens": int(current["total_mapped_tokens"]) + wave_tokens,
    }
    result.pop("checkpoint_sha256", None)
    result["checkpoint_sha256"] = _fingerprint(result, "checkpoint_sha256")
    return validate_mapping_checkpoint(result, release)


def build_glue_production_mapping_package(
    *,
    contract_path: Path,
    repository_root: Path,
    source_commit: str,
    lingua_wheel_pin_path: Path,
    stage_release_path: Path,
    stage_result_path: Path,
    stage_verification_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_contract(contract_path, root)
    mapping = _mapping_contract(contract)
    emr_pin = contract["inputs"]["emr_resume_contract"]
    plan = build_emr_production_plan(
        root / emr_pin["path"], repository_root=root, source_commit=commit
    )
    stage_release = _read_json(stage_release_path)
    stage_result = _read_json(stage_result_path)
    stage_verification = _read_json(stage_verification_path)
    stage_result_sha = _sha256_file(stage_result_path)
    _validate_stage_evidence(
        stage_release,
        stage_result,
        stage_verification,
        stage_result_sha256=stage_result_sha,
    )
    schedule = [dict(row) for row in plan["source_schedule"]]
    resume_ordinal = int(plan["resume"]["source_ordinal"])
    resume_line = int(plan["resume"]["source_next_line"])
    expected_stage_sources = []
    for row in schedule:
        expected_stage_sources.append(
            {
                **row,
                "start_after_line": resume_line
                if int(row["ordinal"]) == resume_ordinal
                else 0,
            }
        )
    if expected_stage_sources != stage_release["sources"]:
        raise ValueError("Glue mapping source semantics differ from sealed staging")
    total_target = int(mapping["target_total_candidate_tokens"])
    categories = _mapping_targets(plan, total_target)
    staged_by_category: dict[str, int] = {row["category"]: 0 for row in categories}
    for row in stage_result["sources"]:
        staged_by_category[str(row["category"])] += int(row["chunk_count"])
    category_inventory = [
        {"category": row["category"], "staged_chunks": staged_by_category[row["category"]]}
        for row in categories
    ]
    if any(row["staged_chunks"] < 1 for row in category_inventory):
        raise ValueError("Glue production mapping category has no staged chunks")

    lingua_pin = _read_json(lingua_wheel_pin_path)
    lingua = _validate_lingua_wheel_pin(lingua_pin)
    output_directory.mkdir(parents=True, exist_ok=True)
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    source_script = root / "scripts" / JOB_SCRIPT
    packaged_script = output_directory / JOB_SCRIPT
    _write_bytes_atomic(packaged_script, source_script.read_bytes())
    implementation[str(source_script.relative_to(root))] = _sha256_file(source_script)
    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    bootstrap = f"{namespace}/control/distributed/glue-production/mapping/bootstrap"
    code_sha = _sha256_file(code_path)
    script_sha = _sha256_file(source_script)
    artifacts = {
        "benchmark_registry": _existing_artifact(plan, "benchmark_registry"),
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/code/{code_sha}.zip",
            sha256=code_sha,
            size=code_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=code_path.name,
        ),
        "frozen_tokenizer": _existing_artifact(plan, "frozen_tokenizer"),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/jobs/{script_sha}.py",
            sha256=script_sha,
            size=source_script.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_script.name,
        ),
        "lingua_cp311_wheel": {
            **lingua_pin["wheel"],
            "state": "existing_verified_codebuild_object",
        },
        "training_exclusion_ledger": _existing_artifact(
            plan, "training_exclusion_ledger"
        ),
    }
    pricing = _read_json(root / contract["inputs"]["glue_pricing"]["path"])
    maximum_compute = (
        Decimal(mapping["hard_maximum_dpu"])
        * Decimal(mapping["maximum_runtime_minutes"])
        / Decimal(60)
        * Decimal(str(pricing["dpu_hour_usd"]))
    ).quantize(Decimal("0.000001"))
    stage_bytes = int(stage_result["staged_compressed_bytes"])
    maximum_mapped = int(contract["limits"]["hard_logical_storage_bytes"]) - stage_bytes
    release: dict[str, Any] = {
        "artifacts": artifacts,
        "authorization": dict(contract["authorization"]),
        "budget": dict(contract["budget"]),
        "categories": categories,
        "category_inventory": category_inventory,
        "data_quality": plan["data_quality"],
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_mapping": "not_authorized",
            "status": "mapping_package_ready_waiting_for_exact_authorization",
        },
        "filters": plan["filters"],
        "gate": "lumi_glue_production_map_release_v0.1",
        "implementation": implementation,
        "job": {
            "additional_python_modules": (
                "tokenizers==0.22.2,datasketch==2.0.0,zstandard==0.25.0,"
                f"{lingua['wheel_uri']}"
            ),
            "execution_class": "STANDARD",
            "glue_version": "5.1",
            "hard_maximum_dpu": int(mapping["hard_maximum_dpu"]),
            "maximum_concurrent_runs": 1,
            "number_of_workers": int(mapping["number_of_workers"]),
            "python_version": "3.11",
            "soft_stop_minutes": int(mapping["soft_stop_minutes"]),
            "timeout_minutes": int(mapping["maximum_runtime_minutes"]),
            "wave_maximum_chunks": int(mapping["wave_maximum_chunks"]),
            "worker_type": mapping["worker_type"],
        },
        "limits": {
            "maximum_category_shortfall_tokens": int(
                contract["limits"]["maximum_category_overshoot_tokens"]
            ),
            "maximum_compute_usd": str(maximum_compute),
            "maximum_mapped_parquet_bytes": maximum_mapped,
            "target_new_candidate_tokens": total_target
            - int(plan["resume"]["accepted_tokens"]),
            "target_total_candidate_tokens": total_target,
        },
        "lingua_compatibility": lingua_pin,
        "output": {
            "checkpoint_prefix": f"{namespace}/control/distributed/glue-production/map-checkpoints",
            "mapped_prefix": f"{namespace}/transient/glue-production/mapped",
            "result_prefix": f"{namespace}/control/distributed/glue-production/map-runs",
            "temp_prefix": f"{namespace}/control/distributed/glue-production/map-temp",
        },
        "region": str(contract["region"]),
        "resume": {
            "accepted_documents": int(plan["resume"]["accepted_documents"]),
            "accepted_tokens": int(plan["resume"]["accepted_tokens"]),
            "source_next_line": resume_line,
            "source_ordinal": resume_ordinal,
        },
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "source_schedule": schedule,
        "stage": {
            "release_fingerprint": stage_release["release_fingerprint"],
            "release_sha256": _sha256_file(stage_release_path),
            "result_sha256": stage_result_sha,
            "staged_chunks": int(stage_result["staged_chunks"]),
            "staged_compressed_bytes": stage_bytes,
            "staged_rows": int(stage_result["staged_rows"]),
            "verification_sha256": _sha256_file(stage_verification_path),
        },
        "staged_sources": stage_result["sources"],
        "storage": {"bucket": bucket, "namespace": namespace},
    }
    release["release_fingerprint"] = _fingerprint(release, "release_fingerprint")
    validate_glue_production_mapping_release(release)
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path, (json.dumps(release, indent=2, sort_keys=True) + "\n").encode()
    )
    initial = build_initial_mapping_checkpoint(release)
    checkpoint_path = output_directory / "initial-map-checkpoint-v0.1.json"
    _write_bytes_atomic(
        checkpoint_path, (json.dumps(initial, indent=2, sort_keys=True) + "\n").encode()
    )
    _manifest(output_directory, [code_path, checkpoint_path, release_path, packaged_script])
    return release


def validate_glue_production_mapping_release(
    release: Mapping[str, Any]
) -> dict[str, Any]:
    if release.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production mapping release schema")
    if release.get("gate") != "lumi_glue_production_map_release_v0.1":
        raise ValueError("unexpected Glue production mapping release gate")
    if release.get("release_fingerprint") != _fingerprint(
        release, "release_fingerprint"
    ):
        raise ValueError("Glue production mapping release fingerprint changed")
    authorization = release.get("authorization", {})
    if authorization.get("production_candidate_acquisition") is not True or any(
        value
        for key, value in authorization.items()
        if key != "production_candidate_acquisition"
    ):
        raise ValueError("Glue production mapping release authorizes paid work")
    job = release.get("job", {})
    if job.get("hard_maximum_dpu") != 100:
        raise ValueError("Glue production mapping DPU target changed")
    if job.get("worker_type") != "G.1X" or job.get("number_of_workers") != 100:
        raise ValueError("Glue production mapping worker shape changed")
    if job.get("timeout_minutes") != 4_320 or job.get("soft_stop_minutes") != 3_960:
        raise ValueError("Glue production mapping runtime envelope changed")
    if job.get("wave_maximum_chunks") != 500:
        raise ValueError("Glue production mapping checkpoint wave changed")
    maximum = (
        Decimal(100) * Decimal(72) * Decimal("0.44")
    ).quantize(Decimal("0.000001"))
    if Decimal(str(release.get("limits", {}).get("maximum_compute_usd"))) != maximum:
        raise ValueError("Glue production mapping cost envelope changed")
    categories = release.get("categories")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("Glue production mapping categories changed")
    accepted = int(release.get("resume", {}).get("accepted_tokens", -1))
    mapped = sum(int(row["mapping_target_tokens"]) for row in categories)
    if accepted + mapped != 150_000_000_000:
        raise ValueError("Glue production mapping no longer targets total 150B")
    if release.get("limits", {}).get("target_total_candidate_tokens") != 150_000_000_000:
        raise ValueError("Glue production mapping 150B hard stop changed")
    sources = release.get("source_schedule")
    staged = release.get("staged_sources")
    if not isinstance(sources, list) or not isinstance(staged, list):
        raise ValueError("Glue production mapping source inventory is missing")
    if len(sources) != 758 or len(staged) != 758:
        raise ValueError("Glue production mapping must bind all staged sources")
    if int(release.get("stage", {}).get("staged_chunks", -1)) != sum(
        int(row["chunk_count"]) for row in staged
    ):
        raise ValueError("Glue production mapping staged chunk count changed")
    if int(release["limits"]["maximum_mapped_parquet_bytes"]) + int(
        release["stage"]["staged_compressed_bytes"]
    ) != 2_000_000_000_000:
        raise ValueError("Glue production mapping storage envelope changed")
    _validate_lingua_wheel_pin(dict(release.get("lingua_compatibility", {})))
    return {
        "maximum_compute_usd": str(maximum),
        "release_fingerprint": release["release_fingerprint"],
        "status": "verified_not_authorized",
        "target_total_candidate_tokens": 150_000_000_000,
    }


def build_glue_production_mapping_authorization(
    release: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_production_mapping_release(release)
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    starting = (
        build_initial_mapping_checkpoint(release)
        if checkpoint is None
        else validate_mapping_checkpoint(checkpoint, release)
    )
    if starting["status"] != "mapping":
        raise ValueError("Glue production mapping authorization requires open work")
    authorization = {
        "approval_text": MAPPING_OWNER_APPROVAL_TEXT,
        "bounds": {
            "authorized_runs": 1,
            "hard_maximum_dpu": 100,
            "maximum_runtime_minutes": 4_320,
            "maximum_usd": release["limits"]["maximum_compute_usd"],
            "starting_checkpoint_sequence": starting["checkpoint_sequence"],
            "starting_checkpoint_sha256": starting["checkpoint_sha256"],
            "target_total_candidate_tokens": 150_000_000_000,
        },
        "expires_at": (issued + timedelta(hours=24)).isoformat(),
        "gate": "lumi_glue_production_map_authorization_v0.1",
        "issued_at": issued.isoformat(),
        "permissions": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_dedup": False,
            "paid_glue_mapping": True,
            "paid_glue_sharding": False,
            "paid_glue_staging": False,
            "paid_gpu_compute": False,
        },
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": release["source_commit"],
    }
    authorization["authorization_sha256"] = _fingerprint(
        authorization, "authorization_sha256"
    )
    validate_glue_production_mapping_authorization(
        authorization, release, checkpoint=starting, now=issued
    )
    return authorization


def validate_glue_production_mapping_authorization(
    authorization: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_production_mapping_release(release)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue production mapping authorization schema")
    if authorization.get("gate") != "lumi_glue_production_map_authorization_v0.1":
        raise ValueError("unexpected Glue production mapping authorization gate")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("Glue production mapping authorization fingerprint changed")
    if authorization.get("approval_text") != MAPPING_OWNER_APPROVAL_TEXT:
        raise ValueError("Glue production mapping owner approval text changed")
    if authorization.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue production mapping authorization targets another release")
    if authorization.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue production mapping authorization source commit changed")
    if authorization.get("permissions") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": False,
        "paid_glue_mapping": True,
        "paid_glue_sharding": False,
        "paid_glue_staging": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue production mapping authorization permissions changed")
    starting = (
        build_initial_mapping_checkpoint(release)
        if checkpoint is None
        else validate_mapping_checkpoint(checkpoint, release)
    )
    if starting["status"] != "mapping":
        raise ValueError("Glue production mapping authorization requires open work")
    if authorization.get("bounds") != {
        "authorized_runs": 1,
        "hard_maximum_dpu": 100,
        "maximum_runtime_minutes": 4_320,
        "maximum_usd": release["limits"]["maximum_compute_usd"],
        "starting_checkpoint_sequence": starting["checkpoint_sequence"],
        "starting_checkpoint_sha256": starting["checkpoint_sha256"],
        "target_total_candidate_tokens": 150_000_000_000,
    }:
        raise ValueError("Glue production mapping authorization bounds changed")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    expires = datetime.fromisoformat(str(authorization["expires_at"])).astimezone(UTC)
    if expires <= issued or expires - issued > timedelta(hours=24):
        raise ValueError("Glue production mapping authorization lifetime changed")
    if current < issued - timedelta(minutes=5) or current >= expires:
        raise ValueError("Glue production mapping authorization is not currently valid")
    return {
        "authorization_sha256": authorization["authorization_sha256"],
        "maximum_compute_usd": authorization["bounds"]["maximum_usd"],
        "status": "authorized_one_100_dpu_mapping_run",
    }


def write_glue_production_mapping_authorization(
    release_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    release = _read_json(release_path)
    checkpoint = None if checkpoint_path is None else _read_json(checkpoint_path)
    authorization = build_glue_production_mapping_authorization(
        release, checkpoint=checkpoint, now=now
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return authorization
