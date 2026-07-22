"""Fail-closed Glue global-dedup packaging for the sealed 150B map.

This module is local-only.  It validates the final Glue map checkpoint and
every immutable wave manifest, renders a content-addressed runtime package,
and can bind a later owner approval to either one small canary or one bounded
production run.  It never calls AWS or starts paid compute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from .emr_serverless_production import build_emr_production_plan
from .glue_canary import _artifact, _valid_commit, _write_bytes_atomic, _zip_code
from .glue_production_mapping import (
    validate_glue_production_mapping_release,
    validate_mapping_checkpoint,
)
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

SCHEMA_VERSION = 1
PACKAGE_MANIFEST = "manifest.sha256"
RELEASE_FILE = "glue-production-global-dedup-release-v0.1.json"
INPUT_INDEX_FILE = "glue-production-global-dedup-input-v0.1.json"
AUTHORIZATION_FILE = "glue-production-global-dedup-authorization-v0.1.json"
MAP_RELEASE_FILE = "sealed-glue-map-release-v0.1.json"
MAP_CHECKPOINT_FILE = "sealed-glue-map-checkpoint-v0.1.json"
CODE_ARCHIVE = "lumi-glue-production-global-dedup-code-v0.1.zip"
JOB_SCRIPT = "run_glue_production_global_dedup.py"
MODES = {"canary", "production"}


def _fingerprint(value: Mapping[str, Any], *excluded: str) -> str:
    skipped = set(excluded)
    material = {key: row for key, row in value.items() if key not in skipped}
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _bounded_key(value: Any, *, namespace: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(f"{namespace}/"):
        raise ValueError(f"{label} left the production namespace")
    if "//" in value or "/../" in f"/{value}/" or value.endswith("/"):
        raise ValueError(f"{label} is not a canonical S3 key")
    return value


def _bounded_uri(
    value: Any,
    *,
    bucket: str,
    prefix: str,
    label: str,
) -> str:
    expected = f"s3://{bucket}/{prefix}/"
    if not isinstance(value, str) or not value.startswith(expected):
        raise ValueError(f"{label} left its bounded S3 prefix")
    if "/../" in f"/{value}/" or value.endswith("/"):
        raise ValueError(f"{label} is not a canonical S3 URI")
    return value


def _read_contract(path: Path, root: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue global-dedup plan schema")
    if contract.get("gate") != "lumi_glue_production_global_dedup_plan_v0.1":
        raise ValueError("unexpected Glue global-dedup plan gate")
    expected_authorization = {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": False,
        "paid_glue_mapping": False,
        "paid_glue_sharding": False,
        "paid_gpu_compute": False,
    }
    if contract.get("authorization") != expected_authorization:
        raise ValueError("Glue global-dedup plan authorizes paid work")
    expected_limits = {
        "candidate_hard_maximum_tokens": 150_000_000_000,
        "candidate_total_floor_tokens": 120_000_000_000,
        "clean_category_floor_tokens": 120_000_000_000,
        "clean_shard_target_tokens": 100_000_000,
        "final_training_tokens": 80_000_000_000,
        "hard_logical_storage_bytes": 2_000_000_000_000,
        "maximum_category_overshoot_tokens": 100_000_000,
    }
    if contract.get("limits") != expected_limits:
        raise ValueError("Glue global-dedup token or storage limits changed")
    expected_runtime = {
        "additional_python_modules": "datasketch==2.0.0",
        "glue_version": "5.1",
        "python_version": "3.11",
        "spark_version": "3.5.6",
    }
    if contract.get("runtime") != expected_runtime:
        raise ValueError("Glue global-dedup runtime changed")
    expected_jobs = {
        "canary": {
            "hard_maximum_dpu": 8,
            "maximum_candidate_documents": 100_000,
            "maximum_compute_usd": "1.760000",
            "maximum_input_uris": 1,
            "maximum_runtime_minutes": 30,
        },
        "production": {
            "hard_maximum_dpu": 100,
            "maximum_compute_usd": "1056.000000",
            "maximum_runtime_minutes": 1_440,
        },
        "worker_type": "G.1X",
    }
    if contract.get("jobs") != expected_jobs:
        raise ValueError("Glue global-dedup job envelope changed")
    if contract.get("region") != "us-east-1":
        raise ValueError("Glue global-dedup region changed")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "emr_resume_contract",
        "glue_150b_contract",
        "glue_pricing",
    }:
        raise ValueError("Glue global-dedup input pins changed")
    for pin in inputs.values():
        if not isinstance(pin, Mapping):
            raise ValueError("Glue global-dedup input pin is invalid")
        pinned = root / str(pin.get("path"))
        if not pinned.is_file() or _sha256_file(pinned) != pin.get("sha256"):
            raise ValueError(f"Glue global-dedup input changed: {pin.get('path')}")
    pricing = _read_json(root / str(inputs["glue_pricing"]["path"]))
    dpu_hour = Decimal(str(pricing["dpu_hour_usd"]))
    for mode in MODES:
        job = expected_jobs[mode]
        maximum = (
            Decimal(job["hard_maximum_dpu"])
            * Decimal(job["maximum_runtime_minutes"])
            / Decimal(60)
            * dpu_hour
        ).quantize(Decimal("0.000001"))
        if maximum != Decimal(job["maximum_compute_usd"]):
            raise ValueError(f"Glue global-dedup {mode} cost envelope changed")
    return contract


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    material = dict(manifest)
    observed = material.pop("manifest_sha256", None)
    digest = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    if _valid_sha256(observed, "Glue map wave manifest") != digest:
        raise ValueError("Glue map wave manifest fingerprint changed")
    return digest


def build_glue_global_dedup_input_index(
    *,
    map_release_path: Path,
    map_checkpoint_path: Path,
    wave_manifest_paths: Sequence[Path],
) -> dict[str, Any]:
    """Flatten and seal all map-wave Parquet inputs for one dedup run."""

    release = _read_json(map_release_path)
    checkpoint = _read_json(map_checkpoint_path)
    validate_glue_production_mapping_release(release)
    validate_mapping_checkpoint(checkpoint, release)
    if checkpoint.get("status") != "ready_for_global_dedup":
        raise ValueError("Glue map checkpoint is not ready for global dedup")
    manifests = [_read_json(path) for path in wave_manifest_paths]
    if len(manifests) != len(checkpoint["outputs"]):
        raise ValueError("Glue global-dedup wave-manifest count changed")
    by_sequence: dict[int, tuple[dict[str, Any], Path]] = {}
    supplied_sequences = []
    for manifest, path in zip(manifests, wave_manifest_paths, strict=True):
        sequence = _positive_int(manifest.get("wave_sequence"), "wave sequence")
        supplied_sequences.append(sequence)
        if sequence in by_sequence:
            raise ValueError("Glue global-dedup repeats a wave sequence")
        by_sequence[sequence] = (manifest, path)
    expected_sequences = list(range(1, int(checkpoint["checkpoint_sequence"]) + 1))
    if sorted(by_sequence) != expected_sequences:
        raise ValueError("Glue global-dedup wave manifests are incomplete")
    if supplied_sequences != expected_sequences:
        raise ValueError("Glue global-dedup wave manifest order changed")

    bucket = str(release["storage"]["bucket"])
    namespace = str(release["storage"]["namespace"])
    current_fingerprint = str(release["release_fingerprint"])
    allowed_fingerprints = {current_fingerprint}
    migration = checkpoint.get("release_migration")
    if migration is not None:
        if not isinstance(migration, Mapping):
            raise ValueError("Glue global-dedup release migration changed")
        allowed_fingerprints.add(
            _valid_sha256(
                migration.get("from_release_fingerprint"),
                "migrated Glue map release",
            )
        )
    category_order = [str(row["category"]) for row in release["categories"]]
    category_totals: dict[str, dict[str, int]] = {
        name: {"documents": 0, "tokens": 0} for name in category_order
    }
    seen_manifest_hashes: set[str] = set()
    seen_manifest_keys: set[str] = set()
    seen_input_uris: set[str] = set()
    seen_object_hashes: set[str] = set()
    inputs: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    for sequence, ledger in enumerate(checkpoint["outputs"], start=1):
        manifest, path = by_sequence[sequence]
        digest = _manifest_digest(manifest)
        if digest != ledger.get("manifest_sha256"):
            raise ValueError("Glue global-dedup manifest differs from checkpoint ledger")
        if digest in seen_manifest_hashes:
            raise ValueError("Glue global-dedup repeats a wave manifest")
        seen_manifest_hashes.add(digest)
        expected = {
            "candidate_documents": ledger["candidate_documents"],
            "candidate_tokens": ledger["candidate_tokens"],
            "gate": "lumi_glue_production_map_wave_result_v0.1",
            "manifest_s3_key": ledger["manifest_s3_key"],
            "parquet_bytes": ledger["parquet_bytes"],
            "schema_version": SCHEMA_VERSION,
            "wave_sequence": sequence,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(f"Glue global-dedup wave changed {name}")
        manifest_key = _bounded_key(
            manifest.get("manifest_s3_key"),
            namespace=namespace,
            label="Glue map wave manifest",
        )
        if not manifest_key.startswith(f"{release['output']['result_prefix']}/"):
            raise ValueError("Glue map wave manifest left the map-result prefix")
        if manifest_key in seen_manifest_keys:
            raise ValueError("Glue global-dedup repeats a wave manifest key")
        seen_manifest_keys.add(manifest_key)
        manifest_release = _valid_sha256(
            manifest.get("release_fingerprint"), "Glue map wave release"
        )
        if manifest_release not in allowed_fingerprints:
            raise ValueError("Glue map wave belongs to an unapproved release")
        if f"/{manifest_release}/" not in f"/{manifest_key}/":
            raise ValueError("Glue map wave key differs from its release")
        categories = manifest.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("Glue map wave category outputs are missing")
        names = [row.get("category") for row in categories if isinstance(row, Mapping)]
        if len(names) != len(categories) or len(names) != len(set(names)):
            raise ValueError("Glue map wave category inventory changed")
        if names != [name for name in category_order if name in set(names)]:
            raise ValueError("Glue map wave category order changed")
        wave_documents = 0
        wave_tokens = 0
        wave_bytes = 0
        wave_objects = 0
        for row in categories:
            assert isinstance(row, Mapping)
            name = str(row["category"])
            documents = _non_negative_int(
                row.get("candidate_documents"), "wave candidate documents"
            )
            tokens = _non_negative_int(row.get("candidate_tokens"), "wave candidate tokens")
            parquet_bytes = _non_negative_int(row.get("parquet_bytes"), "wave Parquet bytes")
            if bool(tokens) != bool(documents) or bool(tokens) != bool(parquet_bytes):
                raise ValueError("Glue map wave empty-output semantics changed")
            if tokens:
                uri = _bounded_uri(
                    row.get("parquet_s3_uri"),
                    bucket=bucket,
                    prefix=(
                        f"{release['output']['mapped_prefix']}/{manifest_release}"
                    ),
                    label="Glue map Parquet output",
                )
                if uri in seen_input_uris:
                    raise ValueError("Glue global-dedup repeats a Parquet input")
                seen_input_uris.add(uri)
                object_count = _positive_int(
                    row.get("object_count"), "wave Parquet object count"
                )
                objects_sha256 = _valid_sha256(
                    row.get("objects_sha256"), "wave Parquet object inventory"
                )
                if objects_sha256 in seen_object_hashes:
                    raise ValueError("Glue global-dedup repeats a Parquet object inventory")
                seen_object_hashes.add(objects_sha256)
                inputs.append(
                    {
                        "candidate_documents": documents,
                        "candidate_tokens": tokens,
                        "category": name,
                        "manifest_sha256": digest,
                        "object_count": object_count,
                        "objects_sha256": objects_sha256,
                        "parquet_bytes": parquet_bytes,
                        "parquet_s3_uri": uri,
                        "wave_sequence": sequence,
                    }
                )
                wave_objects += object_count
            elif row.get("parquet_s3_uri") is not None or row.get("object_count") not in {
                None,
                0,
            }:
                raise ValueError("empty Glue map wave published Parquet")
            category_totals[name]["documents"] += documents
            category_totals[name]["tokens"] += tokens
            wave_documents += documents
            wave_tokens += tokens
            wave_bytes += parquet_bytes
        if (wave_documents, wave_tokens, wave_bytes) != (
            int(ledger["candidate_documents"]),
            int(ledger["candidate_tokens"]),
            int(ledger["parquet_bytes"]),
        ):
            raise ValueError("Glue global-dedup wave totals do not conserve the ledger")
        waves.append(
            {
                "candidate_documents": wave_documents,
                "candidate_tokens": wave_tokens,
                "manifest_file_sha256": _sha256_file(path),
                "manifest_s3_key": manifest_key,
                "manifest_sha256": digest,
                "object_count": wave_objects,
                "parent_checkpoint_sha256": _valid_sha256(
                    manifest.get("parent_checkpoint_sha256"),
                    "Glue map wave parent checkpoint",
                ),
                "parquet_bytes": wave_bytes,
                "release_fingerprint": manifest_release,
                "wave_sequence": sequence,
            }
        )
    for row in checkpoint["categories"]:
        name = str(row["category"])
        if category_totals[name] != {
            "documents": int(row["mapped_documents"]),
            "tokens": int(row["mapped_tokens"]),
        }:
            raise ValueError("Glue global-dedup category totals differ from checkpoint")
    mapped_documents = sum(int(row["candidate_documents"]) for row in inputs)
    mapped_tokens = sum(int(row["candidate_tokens"]) for row in inputs)
    mapped_bytes = sum(int(row["parquet_bytes"]) for row in inputs)
    if (mapped_documents, mapped_tokens, mapped_bytes) != (
        int(checkpoint["total_mapped_documents"]),
        int(checkpoint["total_mapped_tokens"]),
        int(checkpoint["mapped_parquet_bytes"]),
    ):
        raise ValueError("Glue global-dedup input index does not conserve checkpoint totals")
    total_candidate_tokens = mapped_tokens + int(release["resume"]["accepted_tokens"])
    if total_candidate_tokens > int(release["limits"]["target_total_candidate_tokens"]):
        raise ValueError("Glue global-dedup input exceeds the 150B hard cap")
    index: dict[str, Any] = {
        "accepted_prefix": {
            "documents": int(release["resume"]["accepted_documents"]),
            "tokens": int(release["resume"]["accepted_tokens"]),
        },
        "categories": [
            {
                "category": name,
                "mapped_documents": category_totals[name]["documents"],
                "mapped_tokens": category_totals[name]["tokens"],
            }
            for name in category_order
        ],
        "gate": "lumi_glue_production_global_dedup_input_v0.1",
        "inputs": inputs,
        "map_checkpoint_file_sha256": _sha256_file(map_checkpoint_path),
        "map_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "map_release_file_sha256": _sha256_file(map_release_path),
        "map_release_fingerprint": release["release_fingerprint"],
        "mapped_documents": mapped_documents,
        "mapped_parquet_bytes": mapped_bytes,
        "mapped_tokens": mapped_tokens,
        "schema_version": SCHEMA_VERSION,
        "storage": {"bucket": bucket, "namespace": namespace},
        "total_candidate_tokens": total_candidate_tokens,
        "waves": waves,
    }
    index["input_index_sha256"] = _fingerprint(index, "input_index_sha256")
    return validate_glue_global_dedup_input_index(index, release, checkpoint)


def validate_glue_global_dedup_input_index(
    index: Mapping[str, Any],
    map_release: Mapping[str, Any],
    map_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    validate_glue_production_mapping_release(map_release)
    validate_mapping_checkpoint(map_checkpoint, map_release)
    if map_checkpoint.get("status") != "ready_for_global_dedup":
        raise ValueError("Glue map checkpoint is not ready for global dedup")
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue global-dedup input schema")
    if index.get("gate") != "lumi_glue_production_global_dedup_input_v0.1":
        raise ValueError("unexpected Glue global-dedup input gate")
    if index.get("input_index_sha256") != _fingerprint(index, "input_index_sha256"):
        raise ValueError("Glue global-dedup input fingerprint changed")
    if index.get("map_release_fingerprint") != map_release["release_fingerprint"]:
        raise ValueError("Glue global-dedup input targets another map release")
    if index.get("map_checkpoint_sha256") != map_checkpoint["checkpoint_sha256"]:
        raise ValueError("Glue global-dedup input targets another map checkpoint")
    if index.get("storage") != map_release["storage"]:
        raise ValueError("Glue global-dedup input storage changed")
    accepted = index.get("accepted_prefix")
    if accepted != {
        "documents": int(map_release["resume"]["accepted_documents"]),
        "tokens": int(map_release["resume"]["accepted_tokens"]),
    }:
        raise ValueError("Glue global-dedup accepted prefix changed")
    inputs = index.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Glue global-dedup input rows are missing")
    uris: set[str] = set()
    inventories: set[str] = set()
    expected_categories = {str(row["category"]) for row in map_checkpoint["categories"]}
    allowed_fingerprints = {str(map_release["release_fingerprint"])}
    migration = map_checkpoint.get("release_migration")
    if isinstance(migration, Mapping):
        allowed_fingerprints.add(str(migration["from_release_fingerprint"]))
    wave_hashes = {
        int(row["wave_sequence"]): str(row["manifest_sha256"])
        for row in index.get("waves", [])
        if isinstance(row, Mapping)
    }
    category_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "tokens": 0}
    )
    documents = 0
    tokens = 0
    parquet_bytes = 0
    for row in inputs:
        if not isinstance(row, Mapping):
            raise ValueError("Glue global-dedup input row is invalid")
        uri = str(row.get("parquet_s3_uri"))
        if uri in uris:
            raise ValueError("Glue global-dedup repeats a Parquet input")
        uris.add(uri)
        if not any(
            uri.startswith(
                f"s3://{map_release['storage']['bucket']}/"
                f"{map_release['output']['mapped_prefix']}/{fingerprint}/"
            )
            for fingerprint in allowed_fingerprints
        ):
            raise ValueError("Glue global-dedup input left the mapped namespace")
        inventory = _valid_sha256(
            row.get("objects_sha256"), "Glue global-dedup object inventory"
        )
        if inventory in inventories:
            raise ValueError("Glue global-dedup repeats a Parquet object inventory")
        inventories.add(inventory)
        row_documents = _positive_int(
            row.get("candidate_documents"), "dedup input documents"
        )
        row_tokens = _positive_int(row.get("candidate_tokens"), "dedup input tokens")
        row_bytes = _positive_int(row.get("parquet_bytes"), "dedup input bytes")
        _positive_int(row.get("object_count"), "dedup input object count")
        wave_sequence = _positive_int(
            row.get("wave_sequence"), "dedup input wave sequence"
        )
        manifest_sha256 = _valid_sha256(
            row.get("manifest_sha256"), "dedup input manifest"
        )
        if wave_hashes.get(wave_sequence) != manifest_sha256:
            raise ValueError("Glue global-dedup input manifest binding changed")
        name = str(row.get("category"))
        if name not in expected_categories:
            raise ValueError("Glue global-dedup input category changed")
        category_totals[name]["documents"] += row_documents
        category_totals[name]["tokens"] += row_tokens
        documents += row_documents
        tokens += row_tokens
        parquet_bytes += row_bytes
    if (documents, tokens, parquet_bytes) != (
        int(map_checkpoint["total_mapped_documents"]),
        int(map_checkpoint["total_mapped_tokens"]),
        int(map_checkpoint["mapped_parquet_bytes"]),
    ):
        raise ValueError("Glue global-dedup input totals changed")
    if (index.get("mapped_documents"), index.get("mapped_tokens"), index.get(
        "mapped_parquet_bytes"
    )) != (documents, tokens, parquet_bytes):
        raise ValueError("Glue global-dedup input aggregates changed")
    expected_category_rows = [
        {
            "category": str(row["category"]),
            "mapped_documents": int(row["mapped_documents"]),
            "mapped_tokens": int(row["mapped_tokens"]),
        }
        for row in map_checkpoint["categories"]
    ]
    if index.get("categories") != expected_category_rows:
        raise ValueError("Glue global-dedup category ledger changed")
    for row in expected_category_rows:
        if category_totals[row["category"]] != {
            "documents": row["mapped_documents"],
            "tokens": row["mapped_tokens"],
        }:
            raise ValueError("Glue global-dedup category totals changed")
    waves = index.get("waves")
    if not isinstance(waves, list) or [row.get("wave_sequence") for row in waves] != list(
        range(1, int(map_checkpoint["checkpoint_sequence"]) + 1)
    ):
        raise ValueError("Glue global-dedup wave ledger changed")
    if [row.get("manifest_sha256") for row in waves] != [
        row["manifest_sha256"] for row in map_checkpoint["outputs"]
    ]:
        raise ValueError("Glue global-dedup wave manifests changed")
    total = tokens + int(accepted["tokens"])
    if index.get("total_candidate_tokens") != total:
        raise ValueError("Glue global-dedup total candidate tokens changed")
    if total > 150_000_000_000:
        raise ValueError("Glue global-dedup input exceeds the 150B hard cap")
    return dict(index)


def _manifest(directory: Path, paths: Sequence[Path]) -> None:
    rows = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(directory))):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(directory)}\n")
    _write_bytes_atomic(directory / PACKAGE_MANIFEST, "".join(rows).encode())


def _validate_map_semantics(
    map_release: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if map_release.get("storage") != {
        "bucket": plan["storage"]["bucket"],
        "namespace": plan["storage"]["namespace"],
    }:
        raise ValueError("Glue map and dedup source plan use different storage")
    if map_release.get("data_quality") != plan["data_quality"]:
        raise ValueError("Glue map and dedup source plan use different quality gates")
    expected = [
        {
            "accepted_tokens": int(row["accepted_tokens"]),
            "category": str(row["category"]),
            "clean_floor_tokens": int(row["floor_tokens"]),
            "ordinal": int(row["ordinal"]),
        }
        for row in plan["categories"]
    ]
    observed = [
        {
            "accepted_tokens": int(row["accepted_tokens"]),
            "category": str(row["category"]),
            "clean_floor_tokens": int(row["clean_floor_tokens"]),
            "ordinal": int(row["ordinal"]),
        }
        for row in map_release["categories"]
    ]
    if observed != expected:
        raise ValueError("Glue map and dedup source plan use different category semantics")
    if map_release.get("resume", {}).get("accepted_documents") != plan["resume"][
        "accepted_documents"
    ]:
        raise ValueError("Glue map and dedup source plan use different resume anchors")


def build_glue_production_global_dedup_package(
    *,
    contract_path: Path,
    repository_root: Path,
    source_commit: str,
    map_release_path: Path,
    map_checkpoint_path: Path,
    wave_manifest_paths: Sequence[Path],
    output_directory: Path,
) -> dict[str, Any]:
    """Build one deterministic release without authorizing or launching AWS."""

    root = repository_root.resolve()
    commit = _valid_commit(source_commit)
    contract = _read_contract(contract_path, root)
    map_release = _read_json(map_release_path)
    map_checkpoint = _read_json(map_checkpoint_path)
    input_index = build_glue_global_dedup_input_index(
        map_release_path=map_release_path,
        map_checkpoint_path=map_checkpoint_path,
        wave_manifest_paths=wave_manifest_paths,
    )
    plan_pin = contract["inputs"]["emr_resume_contract"]
    plan = build_emr_production_plan(
        root / str(plan_pin["path"]),
        repository_root=root,
        source_commit=commit,
    )
    _validate_map_semantics(map_release, plan)
    if input_index["total_candidate_tokens"] < int(
        contract["limits"]["candidate_total_floor_tokens"]
    ):
        raise ValueError("Glue global-dedup input does not reach the 120B floor")
    used_storage = int(map_release["stage"]["staged_compressed_bytes"]) + int(
        input_index["mapped_parquet_bytes"]
    )
    maximum_storage = int(contract["limits"]["hard_logical_storage_bytes"])
    storage_reserve = int(plan["limits"]["maximum_transient_source_bytes"])
    if used_storage + storage_reserve >= maximum_storage:
        raise ValueError("Glue global-dedup input already exhausts its storage cap")

    output_directory.mkdir(parents=True, exist_ok=True)
    code_path = output_directory / CODE_ARCHIVE
    implementation = _zip_code(root, code_path)
    source_script = root / "scripts" / JOB_SCRIPT
    packaged_script = output_directory / JOB_SCRIPT
    _write_bytes_atomic(packaged_script, source_script.read_bytes())
    implementation[str(source_script.relative_to(root))] = _sha256_file(source_script)
    packaged_map_release = output_directory / MAP_RELEASE_FILE
    packaged_checkpoint = output_directory / MAP_CHECKPOINT_FILE
    _write_bytes_atomic(packaged_map_release, map_release_path.read_bytes())
    _write_bytes_atomic(packaged_checkpoint, map_checkpoint_path.read_bytes())
    index_path = output_directory / INPUT_INDEX_FILE
    _write_bytes_atomic(
        index_path,
        (json.dumps(input_index, indent=2, sort_keys=True) + "\n").encode(),
    )
    packaged_manifests = []
    for path in wave_manifest_paths:
        manifest = _read_json(path)
        destination = output_directory / "wave-manifests" / (
            f"wave-{int(manifest['wave_sequence']):06d}.json"
        )
        _write_bytes_atomic(destination, path.read_bytes())
        packaged_manifests.append(destination)

    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    bootstrap = f"{namespace}/control/distributed/glue-production/global-dedup/bootstrap"
    code_sha = _sha256_file(code_path)
    script_sha = _sha256_file(packaged_script)
    index_sha = _sha256_file(index_path)
    map_release_sha = _sha256_file(packaged_map_release)
    checkpoint_sha = _sha256_file(packaged_checkpoint)
    initial_anchors = plan["artifacts"]["initial_anchors"]
    artifacts = {
        "code_archive": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/code/{code_sha}.zip",
            sha256=code_sha,
            size=code_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=code_path.name,
        ),
        "initial_anchors": _artifact(
            bucket=bucket,
            key=plan["storage"]["artifacts"]["initial_anchors"],
            sha256=str(initial_anchors["sha256"]),
            size=int(initial_anchors["bytes"]),
            state="existing_verified_production_object",
        ),
        "input_index": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/inputs/{index_sha}.json",
            sha256=index_sha,
            size=index_path.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=index_path.name,
        ),
        "job_script": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/jobs/{script_sha}.py",
            sha256=script_sha,
            size=packaged_script.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_script.name,
        ),
        "map_checkpoint": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/map-checkpoints/{checkpoint_sha}.json",
            sha256=checkpoint_sha,
            size=packaged_checkpoint.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_checkpoint.name,
        ),
        "map_release": _artifact(
            bucket=bucket,
            key=f"{bootstrap}/map-releases/{map_release_sha}.json",
            sha256=map_release_sha,
            size=packaged_map_release.stat().st_size,
            state="upload_required_before_authorized_launch",
            package_file=packaged_map_release.name,
        ),
    }
    jobs = {
        mode: {
            **dict(contract["jobs"][mode]),
            "execution_class": "STANDARD",
            "glue_version": contract["runtime"]["glue_version"],
            "maximum_concurrent_runs": 1,
            "number_of_workers": int(contract["jobs"][mode]["hard_maximum_dpu"]),
            "worker_type": contract["jobs"]["worker_type"],
        }
        for mode in sorted(MODES)
    }
    release: dict[str, Any] = {
        "artifacts": artifacts,
        "authorization": dict(contract["authorization"]),
        "categories": [
            {
                "accepted_documents": int(row["accepted_documents"]),
                "accepted_tokens": int(row["accepted_tokens"]),
                "category": str(row["category"]),
                "floor_tokens": int(row["floor_tokens"]),
                "ordinal": int(row["ordinal"]),
            }
            for row in plan["categories"]
        ],
        "data_quality": plan["data_quality"],
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_dedup": "not_authorized",
            "status": "global_dedup_package_ready_waiting_for_exact_authorization",
        },
        "deduplication": {
            "maximum_candidate_pairs": 50_000_000,
            "maximum_documents_without_usable_lsh_buckets": 0,
            "maximum_lsh_bucket_documents": 10_000,
            "maximum_oversized_lsh_buckets": 64,
            "maximum_oversized_lsh_memberships": 1_000_000,
            "maximum_pair_budget_lsh_buckets": 5_000_000,
            "maximum_pair_budget_lsh_memberships": 100_000_000,
            "maximum_rounds": 256,
            "oversized_lsh_bucket_policy": "drop_as_stop_bucket",
            "pair_budget_lsh_bucket_policy": "drop_as_stop_bucket",
        },
        "gate": "lumi_glue_production_global_dedup_release_v0.1",
        "implementation": implementation,
        "input_index_sha256": input_index["input_index_sha256"],
        "jobs": jobs,
        "limits": {
            **dict(contract["limits"]),
            "maximum_new_output_bytes": maximum_storage
            - used_storage
            - storage_reserve,
            "storage_reserve_bytes": storage_reserve,
        },
        "map": {
            "checkpoint_file_sha256": checkpoint_sha,
            "checkpoint_sha256": map_checkpoint["checkpoint_sha256"],
            "input_documents": input_index["mapped_documents"],
            "input_parquet_bytes": input_index["mapped_parquet_bytes"],
            "input_tokens": input_index["mapped_tokens"],
            "release_file_sha256": map_release_sha,
            "release_fingerprint": map_release["release_fingerprint"],
            "total_candidate_tokens": input_index["total_candidate_tokens"],
            "wave_count": len(input_index["waves"]),
        },
        "output": {
            "canary_prefix": f"{namespace}/control/distributed/glue-production/dedup-canary",
            "checkpoint_prefix": (
                f"{namespace}/control/distributed/glue-production/dedup-checkpoints"
            ),
            "production_prefix": f"{namespace}/deduplicated/glue-production",
            "result_prefix": f"{namespace}/control/distributed/glue-production/dedup-runs",
        },
        "region": contract["region"],
        "resume": {
            "accepted_documents": int(plan["resume"]["accepted_documents"]),
            "accepted_tokens": int(plan["resume"]["accepted_tokens"]),
        },
        "runtime": dict(contract["runtime"]),
        "schema_version": SCHEMA_VERSION,
        "source_commit": commit,
        "storage": {"bucket": bucket, "namespace": namespace},
    }
    release["release_fingerprint"] = _fingerprint(release, "release_fingerprint")
    validate_glue_production_global_dedup_release(release)
    release_path = output_directory / RELEASE_FILE
    _write_bytes_atomic(
        release_path,
        (json.dumps(release, indent=2, sort_keys=True) + "\n").encode(),
    )
    _manifest(
        output_directory,
        [
            code_path,
            index_path,
            packaged_checkpoint,
            packaged_map_release,
            packaged_script,
            release_path,
            *packaged_manifests,
        ],
    )
    return release


def validate_glue_production_global_dedup_release(
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if release.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue global-dedup release schema")
    if release.get("gate") != "lumi_glue_production_global_dedup_release_v0.1":
        raise ValueError("unexpected Glue global-dedup release gate")
    if release.get("release_fingerprint") != _fingerprint(
        release, "release_fingerprint"
    ):
        raise ValueError("Glue global-dedup release fingerprint changed")
    _valid_commit(release.get("source_commit"))
    if release.get("authorization") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": False,
        "paid_glue_mapping": False,
        "paid_glue_sharding": False,
        "paid_gpu_compute": False,
    }:
        raise ValueError("Glue global-dedup release authorizes paid work")
    jobs = release.get("jobs")
    if not isinstance(jobs, Mapping) or set(jobs) != MODES:
        raise ValueError("Glue global-dedup job modes changed")
    expected_jobs = {
        "canary": (8, 30, "1.760000"),
        "production": (100, 1_440, "1056.000000"),
    }
    for mode, (dpu, minutes, usd) in expected_jobs.items():
        job = jobs[mode]
        if (
            job.get("hard_maximum_dpu"),
            job.get("number_of_workers"),
            job.get("maximum_runtime_minutes"),
            job.get("maximum_compute_usd"),
            job.get("worker_type"),
            job.get("maximum_concurrent_runs"),
        ) != (dpu, dpu, minutes, usd, "G.1X", 1):
            raise ValueError(f"Glue global-dedup {mode} envelope changed")
    limits = release.get("limits", {})
    if limits.get("candidate_hard_maximum_tokens") != 150_000_000_000:
        raise ValueError("Glue global-dedup 150B cap changed")
    if limits.get("candidate_total_floor_tokens") != 120_000_000_000:
        raise ValueError("Glue global-dedup 120B floor changed")
    if limits.get("clean_category_floor_tokens") != 120_000_000_000:
        raise ValueError("Glue global-dedup category-floor total changed")
    if limits.get("final_training_tokens") != 80_000_000_000:
        raise ValueError("Glue global-dedup final 80B target changed")
    if release.get("deduplication") != {
        "maximum_candidate_pairs": 50_000_000,
        "maximum_documents_without_usable_lsh_buckets": 0,
        "maximum_lsh_bucket_documents": 10_000,
        "maximum_oversized_lsh_buckets": 64,
        "maximum_oversized_lsh_memberships": 1_000_000,
        "maximum_pair_budget_lsh_buckets": 5_000_000,
        "maximum_pair_budget_lsh_memberships": 100_000_000,
        "maximum_rounds": 256,
        "oversized_lsh_bucket_policy": "drop_as_stop_bucket",
        "pair_budget_lsh_bucket_policy": "drop_as_stop_bucket",
    }:
        raise ValueError("Glue global-dedup stop-bucket contract changed")
    maximum_output = _positive_int(
        limits.get("maximum_new_output_bytes"), "dedup output storage cap"
    )
    if limits.get("storage_reserve_bytes") != 100_000_000_000:
        raise ValueError("Glue global-dedup storage reserve changed")
    if maximum_output >= int(limits.get("hard_logical_storage_bytes", 0)):
        raise ValueError("Glue global-dedup output storage cap changed")
    categories = release.get("categories")
    if not isinstance(categories, list) or len(categories) != 7:
        raise ValueError("Glue global-dedup categories changed")
    if sum(int(row["floor_tokens"]) for row in categories) != 120_000_000_000:
        raise ValueError("Glue global-dedup category floors changed")
    if [row["ordinal"] for row in categories] != list(range(7)):
        raise ValueError("Glue global-dedup category order changed")
    map_evidence = release.get("map", {})
    _valid_sha256(map_evidence.get("checkpoint_sha256"), "dedup map checkpoint")
    _valid_sha256(map_evidence.get("checkpoint_file_sha256"), "dedup map checkpoint file")
    _valid_sha256(map_evidence.get("release_fingerprint"), "dedup map release")
    _valid_sha256(map_evidence.get("release_file_sha256"), "dedup map release file")
    if not 120_000_000_000 <= int(
        map_evidence.get("total_candidate_tokens", -1)
    ) <= 150_000_000_000:
        raise ValueError("Glue global-dedup input token total changed")
    if map_evidence.get("wave_count") != 12:
        raise ValueError("Glue global-dedup map wave count changed")
    _valid_sha256(release.get("input_index_sha256"), "dedup input index")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "code_archive",
        "initial_anchors",
        "input_index",
        "job_script",
        "map_checkpoint",
        "map_release",
    }:
        raise ValueError("Glue global-dedup artifacts changed")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Glue global-dedup artifact is invalid: {name}")
        _positive_int(artifact.get("bytes"), f"dedup artifact bytes: {name}")
        _valid_sha256(artifact.get("sha256"), f"dedup artifact hash: {name}")
        _bounded_uri(
            artifact.get("s3_uri"),
            bucket=str(release["storage"]["bucket"]),
            prefix=str(release["storage"]["namespace"]),
            label=f"dedup artifact: {name}",
        )
    return {
        "canary_maximum_usd": jobs["canary"]["maximum_compute_usd"],
        "production_maximum_usd": jobs["production"]["maximum_compute_usd"],
        "release_fingerprint": release["release_fingerprint"],
        "status": "verified_not_authorized",
    }


def expected_glue_global_dedup_approval(
    release: Mapping[str, Any], *, mode: str
) -> str:
    validate_glue_production_global_dedup_release(release)
    if mode not in MODES:
        raise ValueError("unsupported Glue global-dedup authorization mode")
    checkpoint = str(release["map"]["checkpoint_sha256"])
    if mode == "canary":
        return (
            "I approve one bounded Lumi 8-DPU, 30-minute/$1.76 Glue global-dedup "
            f"canary against checkpoint {checkpoint} only. No production dedup, "
            "sharding, GPU, or model training."
        )
    return (
        "I approve one bounded Lumi 100-DPU Glue global-dedup run against checkpoint "
        f"{checkpoint}, maximum 24 hours/$1,056. No sharding, GPU, or model training."
    )


def build_glue_global_dedup_authorization(
    release: Mapping[str, Any],
    *,
    mode: str,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_production_global_dedup_release(release)
    expected = expected_glue_global_dedup_approval(release, mode=mode)
    if approval_text != expected:
        raise ValueError("Glue global-dedup owner approval text does not match")
    issued = (now or datetime.now(UTC)).astimezone(UTC)
    job = release["jobs"][mode]
    authorization: dict[str, Any] = {
        "approval_text": approval_text,
        "bounds": {
            "authorized_runs": 1,
            "hard_maximum_dpu": int(job["hard_maximum_dpu"]),
            "input_index_sha256": release["input_index_sha256"],
            "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
            "maximum_runtime_minutes": int(job["maximum_runtime_minutes"]),
            "maximum_usd": job["maximum_compute_usd"],
        },
        "expires_at": (issued + timedelta(hours=24)).isoformat(),
        "gate": "lumi_glue_production_global_dedup_authorization_v0.1",
        "issued_at": issued.isoformat(),
        "mode": mode,
        "permissions": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_dedup": True,
            "paid_glue_mapping": False,
            "paid_glue_sharding": False,
            "paid_gpu_compute": False,
            "production_global_dedup": mode == "production",
        },
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": release["source_commit"],
    }
    authorization["authorization_sha256"] = _fingerprint(
        authorization, "authorization_sha256"
    )
    validate_glue_global_dedup_authorization(
        authorization, release, now=issued
    )
    return authorization


def validate_glue_global_dedup_authorization(
    authorization: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_glue_production_global_dedup_release(release)
    if authorization.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue global-dedup authorization schema")
    if authorization.get("gate") != (
        "lumi_glue_production_global_dedup_authorization_v0.1"
    ):
        raise ValueError("unexpected Glue global-dedup authorization gate")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("Glue global-dedup authorization fingerprint changed")
    mode = str(authorization.get("mode"))
    if mode not in MODES:
        raise ValueError("Glue global-dedup authorization mode changed")
    if authorization.get("approval_text") != expected_glue_global_dedup_approval(
        release, mode=mode
    ):
        raise ValueError("Glue global-dedup owner approval text changed")
    if authorization.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue global-dedup authorization targets another release")
    if authorization.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue global-dedup authorization source commit changed")
    if authorization.get("permissions") != {
        "full_4b_training": False,
        "model_training": False,
        "paid_glue_dedup": True,
        "paid_glue_mapping": False,
        "paid_glue_sharding": False,
        "paid_gpu_compute": False,
        "production_global_dedup": mode == "production",
    }:
        raise ValueError("Glue global-dedup authorization permissions changed")
    job = release["jobs"][mode]
    if authorization.get("bounds") != {
        "authorized_runs": 1,
        "hard_maximum_dpu": int(job["hard_maximum_dpu"]),
        "input_index_sha256": release["input_index_sha256"],
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "maximum_runtime_minutes": int(job["maximum_runtime_minutes"]),
        "maximum_usd": job["maximum_compute_usd"],
    }:
        raise ValueError("Glue global-dedup authorization bounds changed")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    expires = datetime.fromisoformat(str(authorization["expires_at"])).astimezone(UTC)
    if expires <= issued or expires - issued > timedelta(hours=24):
        raise ValueError("Glue global-dedup authorization lifetime changed")
    if current < issued - timedelta(minutes=5) or current >= expires:
        raise ValueError("Glue global-dedup authorization is not currently valid")
    return {
        "authorization_sha256": authorization["authorization_sha256"],
        "maximum_usd": job["maximum_compute_usd"],
        "mode": mode,
        "status": f"authorized_one_{mode}_global_dedup_run",
    }


def write_glue_global_dedup_authorization(
    release_path: Path,
    output_path: Path,
    *,
    mode: str,
    approval_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    release = _read_json(release_path)
    authorization = build_glue_global_dedup_authorization(
        release,
        mode=mode,
        approval_text=approval_text,
        now=now,
    )
    _write_bytes_atomic(
        output_path,
        (json.dumps(authorization, indent=2, sort_keys=True) + "\n").encode(),
    )
    return authorization


def glue_global_dedup_job_token(authorization: Mapping[str, Any]) -> str:
    digest = _valid_sha256(
        authorization.get("authorization_sha256"), "Glue global-dedup authorization"
    )
    return hashlib.sha256(f"glue-global-dedup:{digest}".encode()).hexdigest()[:32]


def _validate_result_inventory(
    value: Any,
    *,
    release: Mapping[str, Any],
    expected_prefix: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Glue global-dedup {label} inventory is missing")
    expected_uri = f"s3://{release['storage']['bucket']}/{expected_prefix}"
    if value.get("s3_uri") != expected_uri:
        raise ValueError(f"Glue global-dedup {label} left its bounded S3 prefix")
    _positive_int(value.get("object_bytes"), f"dedup {label} bytes")
    _positive_int(value.get("object_count"), f"dedup {label} objects")
    _valid_sha256(value.get("objects_sha256"), f"dedup {label} inventory")
    return dict(value)


def validate_glue_global_dedup_result(
    result: Mapping[str, Any],
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canary or production result before immutable publication."""

    validate_glue_global_dedup_authorization(authorization, release)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Glue global-dedup result schema")
    mode = str(authorization["mode"])
    expected_common = {
        "authorization_sha256": authorization["authorization_sha256"],
        "input_index_sha256": release["input_index_sha256"],
        "job_token": glue_global_dedup_job_token(authorization),
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "mode": mode,
        "release_fingerprint": release["release_fingerprint"],
    }
    for name, value in expected_common.items():
        if result.get(name) != value:
            raise ValueError(f"Glue global-dedup result changed {name}")
    metrics = result.get("deduplication")
    if not isinstance(metrics, Mapping):
        raise ValueError("Glue global-dedup metrics are missing")
    for name in (
        "accepted_documents",
        "candidate_documents",
        "exact_rejected_documents",
        "near_rejected_documents",
    ):
        _non_negative_int(metrics.get(name), f"dedup metric {name}")
    if int(metrics["accepted_documents"]) + int(
        metrics["exact_rejected_documents"]
    ) + int(metrics["near_rejected_documents"]) != int(
        metrics["candidate_documents"]
    ):
        raise ValueError("Glue global-dedup result does not conserve documents")
    for name in (
        "documents_without_usable_lsh_buckets",
        "lsh_bucket_memberships",
        "lsh_candidate_pairs",
        "lsh_pair_upper_bound_after_stop_buckets",
        "lsh_pair_upper_bound_before_stop_buckets",
        "maximum_lsh_bucket_documents_observed",
        "oversized_lsh_buckets",
        "oversized_lsh_memberships",
        "oversized_lsh_pair_upper_bound_removed",
        "pair_budget_lsh_buckets",
        "pair_budget_lsh_memberships",
        "pair_budget_lsh_pair_upper_bound_removed",
    ):
        _non_negative_int(metrics.get(name), f"dedup metric {name}")
    stop_contract = release["deduplication"]
    if int(metrics["oversized_lsh_buckets"]) > int(
        stop_contract["maximum_oversized_lsh_buckets"]
    ):
        raise ValueError("Glue global-dedup result exceeds its stop-bucket count cap")
    if int(metrics["oversized_lsh_memberships"]) > int(
        stop_contract["maximum_oversized_lsh_memberships"]
    ):
        raise ValueError("Glue global-dedup result exceeds its stop-bucket membership cap")
    if int(metrics["pair_budget_lsh_buckets"]) > int(
        stop_contract["maximum_pair_budget_lsh_buckets"]
    ):
        raise ValueError("Glue global-dedup result exceeds its pair-budget bucket cap")
    if int(metrics["pair_budget_lsh_memberships"]) > int(
        stop_contract["maximum_pair_budget_lsh_memberships"]
    ):
        raise ValueError("Glue global-dedup result exceeds its pair-budget membership cap")
    if int(metrics["documents_without_usable_lsh_buckets"]) > int(
        stop_contract["maximum_documents_without_usable_lsh_buckets"]
    ):
        raise ValueError("Glue global-dedup result leaves documents without LSH coverage")
    maximum_bucket_documents = int(metrics["maximum_lsh_bucket_documents_observed"])
    if int(metrics["candidate_documents"]) > 0 and maximum_bucket_documents < 1:
        raise ValueError("Glue global-dedup result has no observed LSH bucket maximum")
    if int(metrics["oversized_lsh_buckets"]) == 0:
        if maximum_bucket_documents > int(
            stop_contract["maximum_lsh_bucket_documents"]
        ):
            raise ValueError(
                "Glue global-dedup result has inconsistent stop-bucket metrics"
            )
    elif maximum_bucket_documents <= int(
        stop_contract["maximum_lsh_bucket_documents"]
    ):
        raise ValueError("Glue global-dedup result has inconsistent stop-bucket maximum")
    pair_upper_before = int(metrics["lsh_pair_upper_bound_before_stop_buckets"])
    pair_upper_after = int(metrics["lsh_pair_upper_bound_after_stop_buckets"])
    removed_pair_upper = int(
        metrics["oversized_lsh_pair_upper_bound_removed"]
    ) + int(metrics["pair_budget_lsh_pair_upper_bound_removed"])
    if pair_upper_before != pair_upper_after + removed_pair_upper:
        raise ValueError("Glue global-dedup result has inconsistent pair-budget accounting")
    if pair_upper_after > int(stop_contract["maximum_candidate_pairs"]):
        raise ValueError("Glue global-dedup result exceeds its pair upper-bound cap")
    if int(metrics["lsh_candidate_pairs"]) > pair_upper_after:
        raise ValueError("Glue global-dedup result exceeds its measured pair upper bound")
    if (
        int(metrics["oversized_lsh_memberships"])
        + int(metrics["pair_budget_lsh_memberships"])
        > int(metrics["lsh_bucket_memberships"])
    ):
        raise ValueError("Glue global-dedup result has inconsistent LSH memberships")
    for prefix in ("oversized", "pair_budget"):
        bucket_count = int(metrics[f"{prefix}_lsh_buckets"])
        memberships = int(metrics[f"{prefix}_lsh_memberships"])
        removed = int(metrics[f"{prefix}_lsh_pair_upper_bound_removed"])
        if (bucket_count == 0) != (memberships == 0 and removed == 0):
            raise ValueError("Glue global-dedup result has inconsistent stop-bucket telemetry")
    token = str(expected_common["job_token"])
    if mode == "canary":
        if result.get("gate") != (
            "lumi_glue_production_global_dedup_canary_result_v0.1"
        ):
            raise ValueError("unexpected Glue global-dedup canary result gate")
        if result.get("status") != "pass_structural_canary":
            raise ValueError("Glue global-dedup canary did not pass")
        input_documents = _positive_int(
            result.get("input_documents"), "dedup canary input documents"
        )
        _positive_int(result.get("input_tokens"), "dedup canary input tokens")
        if input_documents > int(
            release["jobs"]["canary"]["maximum_candidate_documents"]
        ):
            raise ValueError("Glue global-dedup canary exceeded its document cap")
        if int(metrics["candidate_documents"]) != input_documents:
            raise ValueError("Glue global-dedup canary metrics changed")
        prefix = f"{release['output']['canary_prefix']}/{token}"
        accepted = _validate_result_inventory(
            result.get("accepted_ids"),
            release=release,
            expected_prefix=f"{prefix}/accepted-ids",
            label="canary accepted IDs",
        )
        rejected = _validate_result_inventory(
            result.get("rejected_ids"),
            release=release,
            expected_prefix=f"{prefix}/rejected-ids",
            label="canary rejected IDs",
        )
        if int(accepted["object_bytes"]) + int(rejected["object_bytes"]) > int(
            release["limits"]["maximum_new_output_bytes"]
        ):
            raise ValueError("Glue global-dedup canary exceeded its storage cap")
        return dict(result)

    if result.get("gate") != "lumi_glue_production_global_dedup_result_v0.1":
        raise ValueError("unexpected Glue global-dedup production result gate")
    if result.get("status") != "pass_category_floors" or result.get(
        "total_deficit_tokens"
    ) != 0:
        raise ValueError("Glue global dedup did not satisfy every category floor")
    if int(metrics["candidate_documents"]) != int(release["map"]["input_documents"]):
        raise ValueError("Glue global-dedup production metrics changed")
    categories = result.get("categories")
    if not isinstance(categories, list) or [row.get("category") for row in categories] != [
        row["category"] for row in release["categories"]
    ]:
        raise ValueError("Glue global-dedup result categories changed")
    selected_documents = 0
    selected_tokens = 0
    for row, expected in zip(categories, release["categories"], strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("Glue global-dedup category result is invalid")
        if int(row.get("floor_tokens", -1)) != int(expected["floor_tokens"]):
            raise ValueError("Glue global-dedup category floor changed")
        row_documents = _non_negative_int(
            row.get("selected_documents"), "dedup selected documents"
        )
        row_tokens = _non_negative_int(row.get("selected_tokens"), "dedup selected tokens")
        if int(row.get("final_documents", -1)) != int(
            expected["accepted_documents"]
        ) + row_documents:
            raise ValueError("Glue global-dedup category document conservation failed")
        final_tokens = int(expected["accepted_tokens"]) + row_tokens
        if int(row.get("final_tokens", -1)) != final_tokens:
            raise ValueError("Glue global-dedup category token conservation failed")
        if not int(expected["floor_tokens"]) <= final_tokens <= int(
            expected["floor_tokens"]
        ) + int(release["limits"]["maximum_category_overshoot_tokens"]):
            raise ValueError("Glue global-dedup category left its floor envelope")
        if int(row.get("deficit_tokens", -1)) != 0:
            raise ValueError("Glue global-dedup category reports a deficit")
        if int(row.get("available_documents", -1)) < row_documents or int(
            row.get("available_tokens", -1)
        ) < row_tokens:
            raise ValueError("Glue global-dedup selected unavailable candidates")
        selected_documents += row_documents
        selected_tokens += row_tokens
    prefix = f"{release['output']['production_prefix']}/{token}"
    accepted = _validate_result_inventory(
        result.get("accepted_candidates"),
        release=release,
        expected_prefix=f"{prefix}/accepted-candidates",
        label="accepted candidates",
    )
    rejected = _validate_result_inventory(
        result.get("rejected_ids"),
        release=release,
        expected_prefix=f"{prefix}/rejected-ids",
        label="rejected IDs",
    )
    if int(accepted.get("documents", -1)) != selected_documents or int(
        accepted.get("tokens", -1)
    ) != selected_tokens:
        raise ValueError("Glue global-dedup accepted totals changed")
    if int(accepted["object_bytes"]) + int(rejected["object_bytes"]) > int(
        release["limits"]["maximum_new_output_bytes"]
    ):
        raise ValueError("Glue global-dedup exceeded its output storage cap")
    return dict(result)


def build_glue_global_dedup_job_definition(
    release: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    role_arn: str,
    release_s3_uri: str,
    authorization_s3_uri: str,
) -> dict[str, Any]:
    """Render one Glue job definition without creating or starting it."""

    validate_glue_global_dedup_authorization(authorization, release)
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", role_arn
    ) is None:
        raise ValueError("Glue global-dedup role ARN is invalid")
    bucket = str(release["storage"]["bucket"])
    namespace = str(release["storage"]["namespace"])
    files = [
        release_s3_uri,
        release["artifacts"]["input_index"]["s3_uri"],
        authorization_s3_uri,
        release["artifacts"]["initial_anchors"]["s3_uri"],
        release["artifacts"]["map_release"]["s3_uri"],
        release["artifacts"]["map_checkpoint"]["s3_uri"],
    ]
    for uri in files:
        _bounded_uri(uri, bucket=bucket, prefix=namespace, label="dedup bootstrap file")
    basenames = [Path(str(uri)).name for uri in files]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Glue global-dedup bootstrap basenames collide")
    for name in ("code_archive", "job_script"):
        _bounded_uri(
            release["artifacts"][name]["s3_uri"],
            bucket=bucket,
            prefix=namespace,
            label=f"dedup {name}",
        )
    mode = str(authorization["mode"])
    job = release["jobs"][mode]
    token = glue_global_dedup_job_token(authorization)
    return {
        "Command": {
            "Name": "glueetl",
            "PythonVersion": "3",
            "ScriptLocation": release["artifacts"]["job_script"]["s3_uri"],
        },
        "DefaultArguments": {
            "--LUMI_AUTHORIZATION_FILE": basenames[2],
            "--LUMI_INPUT_INDEX_FILE": basenames[1],
            "--LUMI_MAP_CHECKPOINT_FILE": basenames[5],
            "--LUMI_MAP_RELEASE_FILE": basenames[4],
            "--LUMI_RELEASE_FILE": basenames[0],
            "--additional-python-modules": release["runtime"][
                "additional_python_modules"
            ],
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-job-insights": "true",
            "--enable-metrics": "true",
            "--extra-files": ",".join(files),
            "--extra-py-files": release["artifacts"]["code_archive"]["s3_uri"],
            "--job-language": "python",
        },
        "Description": (
            f"Lumi bounded Glue {mode} global dedup; no mapping, sharding, GPU, "
            "or model training"
        ),
        "ExecutionClass": "STANDARD",
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
        "GlueVersion": release["runtime"]["glue_version"],
        "MaxRetries": 0,
        "Name": f"lumi-glue-global-dedup-{mode}-{token[:12]}",
        "NumberOfWorkers": int(job["number_of_workers"]),
        "Role": role_arn,
        "Tags": {
            "LumiInputIndex": str(release["input_index_sha256"]),
            "LumiScope": f"bounded-{mode}-global-dedup-only",
        },
        "Timeout": int(job["maximum_runtime_minutes"]),
        "WorkerType": job["worker_type"],
    }
