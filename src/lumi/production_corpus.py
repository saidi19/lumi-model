"""Single-writer, resumable production corpus acquisition for Lumi."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from array import array
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tokenizers import Tokenizer

from .aws_s3_canary import AwsCli
from .aws_s3_production import _put_immutable_object
from .data_pipeline import (
    PII_DETECTOR_VERSION,
    _extract_provenance,
    _record_id,
    filter_document,
    split_utf8,
)
from .data_quality import DataQualityGate
from .development_config import _source_provenance

SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
SHA256_HEX = set("0123456789abcdef")
IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "scripts/run_production_acquisition.py",
    "src/lumi/__init__.py",
    "src/lumi/production_corpus.py",
    "src/lumi/data_pipeline.py",
    "src/lumi/data_quality.py",
    "src/lumi/aws_s3_production.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _resolve(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _valid_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def _open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS categories(
            ordinal INTEGER NOT NULL,
            category TEXT PRIMARY KEY,
            floor_tokens INTEGER NOT NULL,
            accepted_tokens INTEGER NOT NULL DEFAULT 0,
            accepted_documents INTEGER NOT NULL DEFAULT 0,
            accepted_text_bytes INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS sources(
            ordinal INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            component TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_bytes INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            next_line INTEGER NOT NULL DEFAULT 0,
            raw_documents INTEGER NOT NULL DEFAULT 0,
            raw_text_bytes INTEGER NOT NULL DEFAULT 0,
            accepted_documents INTEGER NOT NULL DEFAULT 0,
            accepted_text_bytes INTEGER NOT NULL DEFAULT 0,
            accepted_tokens INTEGER NOT NULL DEFAULT 0,
            rejections TEXT NOT NULL DEFAULT '{}',
            pii_redactions TEXT NOT NULL DEFAULT '{}',
            language_labels TEXT NOT NULL DEFAULT '{}',
            UNIQUE(dataset_id, revision, source_file)
        );
        CREATE TABLE IF NOT EXISTS exact_content(
            content_sha256 TEXT PRIMARY KEY,
            document_id TEXT NOT NULL UNIQUE,
            source_ordinal INTEGER NOT NULL,
            source_line INTEGER NOT NULL,
            segment_index INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS exclusion_documents(
            document_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS exclusion_content(
            content_sha256 TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS exclusion_near_duplicate(
            fingerprint_sha256 TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS current_shard(
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            shard_id TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            clean_path TEXT NOT NULL,
            token_path TEXT NOT NULL,
            clean_offset INTEGER NOT NULL DEFAULT 0,
            token_bytes INTEGER NOT NULL DEFAULT 0,
            tokens INTEGER NOT NULL DEFAULT 0,
            documents INTEGER NOT NULL DEFAULT 0,
            text_bytes INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS shards(
            shard_id TEXT PRIMARY KEY,
            sequence INTEGER NOT NULL UNIQUE,
            category TEXT NOT NULL,
            clean_object_key TEXT NOT NULL UNIQUE,
            clean_sha256 TEXT NOT NULL UNIQUE,
            token_object_key TEXT NOT NULL UNIQUE,
            token_sha256 TEXT NOT NULL UNIQUE,
            provenance_object_key TEXT NOT NULL UNIQUE,
            provenance_sha256 TEXT NOT NULL UNIQUE,
            manifest_object_key TEXT NOT NULL UNIQUE,
            manifest_sha256 TEXT NOT NULL UNIQUE,
            documents INTEGER NOT NULL,
            text_bytes INTEGER NOT NULL,
            tokens INTEGER NOT NULL,
            committed_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def build_round_robin_source_schedule(
    source_manifest: Mapping[str, Any], mixture: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Spread early acquisition across every component in each category."""

    source_rows = source_manifest.get("sources")
    mixture_rows = mixture.get("mixture")
    if not isinstance(source_rows, list) or not isinstance(mixture_rows, list):
        raise ValueError("source manifest or mixture is missing")
    by_component = {str(row["component"]): row for row in source_rows}
    schedule = []
    categories = []
    ordinal = 0
    for category_ordinal, mixture_row in enumerate(mixture_rows):
        category = str(mixture_row["category"])
        target = int(mixture_row["target_tokens_at_80b"])
        floor = target * 3 // 2
        components = list(mixture_row["source_components"])
        queues: dict[str, deque[dict[str, Any]]] = {}
        for component in components:
            source = by_component.get(str(component))
            if source is None or source.get("category") != category:
                raise ValueError(f"mixture component is absent or miscategorized: {component}")
            files = sorted(source.get("data_files", []), key=lambda row: row["source_file"])
            if not files:
                raise ValueError(f"component has no pinned source files: {component}")
            queues[str(component)] = deque(files)
        while any(queues.values()):
            for component in components:
                queue = queues[str(component)]
                if not queue:
                    continue
                data_file = queue.popleft()
                source = by_component[str(component)]
                schedule.append(
                    {
                        "category": category,
                        "component": str(component),
                        "dataset_id": source["dataset_id"],
                        "license_policy_status": source["license_policy_status"],
                        "ordinal": ordinal,
                        "provenance_paths": list(source["provenance_paths"]),
                        "revision": source["dataset_revision"],
                        "source_bytes": int(data_file["source_bytes"]),
                        "source_file": data_file["source_file"],
                        "source_sha256": data_file["source_digest"]["value"],
                    }
                )
                ordinal += 1
        categories.append(
            {
                "category": category,
                "floor_tokens": floor,
                "ordinal": category_ordinal,
                "target_tokens": target,
            }
        )
    if len(schedule) != 758 or sum(row["source_bytes"] for row in schedule) != 499_817_285_384:
        raise ValueError("production source schedule differs from the approved inventory")
    if sum(row["floor_tokens"] for row in categories) != 120_000_000_000:
        raise ValueError("production category floors do not total 120B tokens")
    return schedule, categories


def _component_provenance(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = inventory.get("sources")
    if not isinstance(rows, list):
        raise ValueError("component inventory is missing")
    result = {}
    for row in rows:
        retained, static, required = _source_provenance(row)
        result[str(row["component"])] = {
            "required_non_null_provenance": required,
            "retain_provenance": retained,
            "static_provenance": static,
        }
    return result


def build_worker_release(
    contract_path: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    production_report: Path,
) -> dict[str, Any]:
    """Build the exact worker release without launching compute."""

    root = repository_root.resolve()
    contract = _read_json(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production worker schema")
    if contract.get("gate") != "production_corpus_worker_release_v0.1":
        raise ValueError("unexpected production worker gate")
    authorization = contract.get("authorization", {})
    if authorization.get("production_adaptive_acquisition") is not True:
        raise ValueError("production adaptive acquisition is not authorized")
    if any(
        authorization.get(name) is not False
        for name in ("full_4b_training", "model_training", "paid_gpu_compute")
    ):
        raise ValueError("production corpus worker must not authorize GPU or training")
    runtime = contract.get("runtime", {})
    expected_runtime = {
        "instance_initiated_shutdown_behavior": "terminate",
        "instance_type": "r7i.4xlarge",
        "instance_vcpus": 16,
        "instance_memory_gib": 128,
        "market": "on_demand",
        "maximum_runtime_hours": 72,
        "metadata_http_tokens": "required",
        "on_demand_usd_per_hour": "1.0584",
        "root_volume_gib": 30,
        "root_volume_type": "gp3",
        "state_volume_delete_on_termination": True,
        "state_volume_gib": 500,
        "state_volume_type": "gp3",
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"production worker runtime changed: {name}")
    if float(runtime["maximum_runtime_hours"]) * float(runtime["on_demand_usd_per_hour"]) >= 80:
        raise ValueError("production worker compute forecast exceeds its safe share of the cap")

    configured_inputs = contract.get("inputs")
    expected_inputs = {
        "aws_worker_pricing",
        "component_inventory",
        "data_mix",
        "data_quality",
        "filter_policy",
        "production_authorization",
        "source_file_manifest",
    }
    if not isinstance(configured_inputs, Mapping) or set(configured_inputs) != expected_inputs:
        raise ValueError("production worker input set changed")
    inputs = {}
    paths = {}
    for name, pin in sorted(configured_inputs.items()):
        path = _resolve(root, pin["path"], name)
        digest = _sha256_file(path)
        if digest != _valid_sha256(pin["sha256"], f"{name} SHA-256"):
            raise ValueError(f"production worker input changed: {name}")
        paths[name] = path
        inputs[name] = {"path": str(path.relative_to(root)), "sha256": digest}

    pricing = _read_json(paths["aws_worker_pricing"])
    if pricing.get("schema_version") != 1 or pricing.get("gate") != (
        "aws_production_worker_pricing_snapshot_v0.1"
    ):
        raise ValueError("production worker pricing snapshot is invalid")
    ec2_price = pricing.get("ec2", {})
    ebs_price = pricing.get("ebs", {})
    if (
        ec2_price.get("instance_type") != runtime["instance_type"]
        or ec2_price.get("vcpus") != runtime["instance_vcpus"]
        or ec2_price.get("memory_gib") != runtime["instance_memory_gib"]
        or Decimal(str(ec2_price.get("on_demand_usd_per_hour")))
        != Decimal(runtime["on_demand_usd_per_hour"])
    ):
        raise ValueError("production EC2 price does not match the worker runtime")
    if (
        ebs_price.get("volume_type") != "gp3"
        or Decimal(str(ebs_price.get("storage_usd_per_gb_month"))) != Decimal("0.08")
    ):
        raise ValueError("production gp3 price changed")
    compute_cost = Decimal(runtime["on_demand_usd_per_hour"]) * Decimal(
        runtime["maximum_runtime_hours"]
    )
    attached_gib = Decimal(runtime["root_volume_gib"] + runtime["state_volume_gib"])
    attached_storage_cost = (
        attached_gib
        * Decimal(str(ebs_price["storage_usd_per_gb_month"]))
        * Decimal(runtime["maximum_runtime_hours"])
        / Decimal(730)
    )
    worker_cost = compute_cost + attached_storage_cost
    if worker_cost >= Decimal("100.00"):
        raise ValueError("production worker EC2 plus attached EBS forecast exceeds $100")
    cost_forecast = {
        "attached_ebs_72h_usd": str(attached_storage_cost.quantize(Decimal("0.0001"))),
        "ec2_72h_usd": str(compute_cost.quantize(Decimal("0.0001"))),
        "maximum_72h_usd": str(worker_cost.quantize(Decimal("0.0001"))),
    }

    production = _read_json(production_report)
    expected_production_fingerprint = contract.get("production_plan_fingerprint")
    if production.get("plan_fingerprint") != expected_production_fingerprint:
        raise ValueError("production S3 report differs from the worker contract")
    if production.get("decision", {}).get("status") != "production_s3_ready_for_acquisition_worker":
        raise ValueError("production S3 namespace is not ready")
    if production.get("cost_policy", {}).get("initial_paid_cpu_compute_cap_usd") != "100.00":
        raise ValueError("production CPU cap changed")
    expected_acquisition = {
        "checkpoint_interval_records": 1000,
        "clean_shard_target_tokens": 100_000_000,
        "maximum_concurrent_workers": 1,
        "raw_source_payload_persistence": False,
        "source_order": "data_mix_category_then_component_then_source_file",
        "worker_local_scratch_hard_cap_bytes": 20_000_000_000,
        "worker_strategy": "single_writer_global_dedup_v0.1",
    }
    expected_limits = {
        "candidate_hard_maximum_tokens": 150_000_000_000,
        "candidate_total_floor_tokens": 120_000_000_000,
        "final_training_tokens": 80_000_000_000,
        "hard_logical_storage_bytes": 2_000_000_000_000,
        "maximum_category_overshoot_tokens": 100_000_000,
    }
    if production.get("acquisition") != expected_acquisition:
        raise ValueError("production acquisition policy changed")
    if production.get("limits") != expected_limits:
        raise ValueError("production acquisition limits changed")

    external = contract.get("external_artifacts", {})
    external_paths = {
        "frozen_tokenizer": tokenizer_path.resolve(),
        "training_exclusion_ledger": training_exclusion_ledger.resolve(),
    }
    external_report = {}
    for name, path in external_paths.items():
        pin = external.get(name, {})
        if not path.is_file():
            raise FileNotFoundError(f"missing production worker artifact: {path}")
        if path.stat().st_size != int(pin.get("bytes", -1)):
            raise ValueError(f"production worker artifact size changed: {name}")
        digest = _sha256_file(path)
        if digest != _valid_sha256(pin.get("sha256"), f"{name} SHA-256"):
            raise ValueError(f"production worker artifact hash changed: {name}")
        external_report[name] = {"bytes": path.stat().st_size, "sha256": digest}

    source_manifest = _read_json(paths["source_file_manifest"])
    mixture = _read_json(paths["data_mix"])
    schedule, categories = build_round_robin_source_schedule(source_manifest, mixture)
    implementation = {}
    for raw_path in IMPLEMENTATION_FILES:
        path = _resolve(root, raw_path, "worker implementation")
        implementation[raw_path] = _sha256_file(path)
    material = {
        "contract_sha256": _sha256_file(contract_path),
        "external_artifacts": external_report,
        "implementation": implementation,
        "inputs": inputs,
        "production_plan_fingerprint": expected_production_fingerprint,
        "schema_version": SCHEMA_VERSION,
    }
    return {
        "authorization": dict(authorization),
        "acquisition": expected_acquisition,
        "categories": categories,
        "cost_forecast": cost_forecast,
        "decision": {
            "full_4b_training": "not_authorized",
            "gpu_compute": "not_authorized",
            "status": "ready_to_launch_acquisition_worker",
        },
        "external_artifacts": external_report,
        "gate": contract["gate"],
        "implementation": implementation,
        "inputs": inputs,
        "limits": expected_limits,
        "production_plan_fingerprint": expected_production_fingerprint,
        "runtime": dict(runtime),
        "schedule": schedule,
        "schema_version": SCHEMA_VERSION,
        "worker_release_fingerprint": _sha256_bytes(_canonical_json_bytes(material)),
    }


def _initialize_state(
    connection: sqlite3.Connection,
    release: Mapping[str, Any],
    training_exclusion_ledger: Path,
) -> None:
    fingerprint = str(release["worker_release_fingerprint"])
    existing = _metadata_get(connection, "worker_release_fingerprint")
    if existing is not None and existing != fingerprint:
        raise ValueError("existing production state belongs to another worker release")
    with connection:
        _metadata_set(connection, "schema_version", STATE_SCHEMA_VERSION)
        _metadata_set(connection, "worker_release_fingerprint", fingerprint)
        _metadata_set(connection, "status", "running")
        _metadata_set(connection, "updated_at", _utc_now())
        if _metadata_get(connection, "started_at") is None:
            _metadata_set(connection, "started_at", _utc_now())
            _metadata_set(connection, "next_shard_sequence", 0)
        for category in release["categories"]:
            connection.execute(
                "INSERT OR IGNORE INTO categories(ordinal, category, floor_tokens) "
                "VALUES (?, ?, ?)",
                (category["ordinal"], category["category"], category["floor_tokens"]),
            )
            row = connection.execute(
                "SELECT ordinal, floor_tokens FROM categories WHERE category = ?",
                (category["category"],),
            ).fetchone()
            if row is None or (int(row["ordinal"]), int(row["floor_tokens"])) != (
                int(category["ordinal"]),
                int(category["floor_tokens"]),
            ):
                raise ValueError("existing production category contract changed")
        for source in release["schedule"]:
            connection.execute(
                "INSERT OR IGNORE INTO sources(ordinal, category, component, dataset_id, "
                "revision, source_file, source_bytes, source_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source["ordinal"],
                    source["category"],
                    source["component"],
                    source["dataset_id"],
                    source["revision"],
                    source["source_file"],
                    source["source_bytes"],
                    source["source_sha256"],
                ),
            )
        if connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0] != len(
            release["schedule"]
        ):
            raise ValueError("existing production source schedule changed")
        if connection.execute("SELECT COUNT(*) FROM exclusion_content").fetchone()[0] == 0:
            with training_exclusion_ledger.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    try:
                        document_id = str(row["document_id"])
                        content = _valid_sha256(row["content_sha256"], "excluded content")
                        near = _valid_sha256(
                            row["near_duplicate_sha256"], "excluded near duplicate"
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid training exclusion ledger line {line_number}"
                        ) from error
                    connection.execute(
                        "INSERT INTO exclusion_documents(document_id) VALUES (?)", (document_id,)
                    )
                    connection.execute(
                        "INSERT INTO exclusion_content(content_sha256) VALUES (?)", (content,)
                    )
                    connection.execute(
                        "INSERT INTO exclusion_near_duplicate(fingerprint_sha256) VALUES (?)",
                        (near,),
                    )


def _source_url(source: Mapping[str, Any]) -> str:
    dataset_id = quote(str(source["dataset_id"]), safe="/")
    revision = quote(str(source["revision"]), safe="")
    source_file = quote(str(source["source_file"]), safe="/")
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{source_file}"


def _scratch_bytes(scratch: Path) -> int:
    return sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file())


def _download_source(source: Mapping[str, Any], scratch: Path, scratch_cap: int) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    digest = str(source["source_sha256"])
    final = scratch / f"source-{digest}.json.gz"
    partial = scratch / f"source-{digest}.json.gz.part"
    expected_bytes = int(source["source_bytes"])
    if final.exists():
        if final.stat().st_size == expected_bytes and _sha256_file(final) == digest:
            return final
        final.unlink()
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    if _scratch_bytes(scratch) + max(0, expected_bytes - partial_bytes) > scratch_cap:
        raise RuntimeError("production worker scratch cap would be exceeded by source download")
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "10",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        _source_url(source),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"failed downloading pinned source: {source['source_file']}")
    if partial.stat().st_size != expected_bytes or _sha256_file(partial) != digest:
        raise RuntimeError(f"pinned source failed size or SHA-256: {source['source_file']}")
    partial.replace(final)
    return final


def _iter_gzip_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any] | None]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, row if isinstance(row, Mapping) else None


def _counter(value: str) -> Counter[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored production counter is invalid")
    return Counter({str(name): int(count) for name, count in parsed.items()})


def _current_shard(
    connection: sqlite3.Connection, state_directory: Path, category: str
) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM current_shard WHERE singleton = 1").fetchone()
    if row is not None:
        if row["category"] != category:
            raise ValueError("current production shard crosses categories")
        clean = state_directory / row["clean_path"]
        tokens = state_directory / row["token_path"]
        clean.parent.mkdir(parents=True, exist_ok=True)
        tokens.parent.mkdir(parents=True, exist_ok=True)
        clean.touch(exist_ok=True)
        tokens.touch(exist_ok=True)
        with clean.open("r+b") as handle:
            handle.truncate(int(row["clean_offset"]))
        with tokens.open("r+b") as handle:
            handle.truncate(int(row["token_bytes"]))
        return row
    sequence = int(_metadata_get(connection, "next_shard_sequence") or "0")
    shard_id = f"{category}-{sequence:06d}"
    clean_relative = f"scratch/{shard_id}.jsonl"
    token_relative = f"scratch/{shard_id}.uint16.bin"
    with connection:
        connection.execute(
            "INSERT INTO current_shard(singleton, shard_id, category, sequence, clean_path, "
            "token_path) VALUES (1, ?, ?, ?, ?, ?)",
            (shard_id, category, sequence, clean_relative, token_relative),
        )
    (state_directory / clean_relative).parent.mkdir(parents=True, exist_ok=True)
    (state_directory / clean_relative).touch(exist_ok=True)
    (state_directory / token_relative).touch(exist_ok=True)
    row = connection.execute("SELECT * FROM current_shard WHERE singleton = 1").fetchone()
    assert row is not None
    return row


def _excluded(connection: sqlite3.Connection, table: str, column: str, value: str) -> bool:
    return (
        connection.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ?",  # noqa: S608
            (value,),
        ).fetchone()
        is not None
    )


def _process_batch(
    connection: sqlite3.Connection,
    quality_gate: DataQualityGate,
    tokenizer: Tokenizer,
    *,
    source: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    source_row: sqlite3.Row,
    records: list[tuple[int, Mapping[str, Any] | None]],
    filters: Mapping[str, Any],
    state_directory: Path,
    hard_max_tokens: int,
) -> dict[str, int]:
    shard = _current_shard(connection, state_directory, str(source["category"]))
    clean_path = state_directory / shard["clean_path"]
    token_path = state_directory / shard["token_path"]
    clean_before = int(shard["clean_offset"])
    token_before = int(shard["token_bytes"])
    rejections = _counter(str(source_row["rejections"]))
    redactions = _counter(str(source_row["pii_redactions"]))
    languages = _counter(str(source_row["language_labels"]))
    raw_documents = 0
    raw_text_bytes = 0
    accepted: list[dict[str, Any]] = []
    local_hashes: set[str] = set()
    dataset = {"id": source["dataset_id"], "revision": source["revision"]}
    connection.execute("BEGIN IMMEDIATE")
    try:
        for source_line, upstream in records:
            raw_documents += 1
            if upstream is None:
                rejections["invalid_json"] += 1
                continue
            raw_text = upstream.get("text")
            if not isinstance(raw_text, str):
                rejections["missing_text"] += 1
                continue
            raw_text_bytes += len(raw_text.encode("utf-8"))
            provenance_source = {
                **source_provenance,
                "file": source["source_file"],
            }
            provenance, missing = _extract_provenance(upstream, provenance_source)
            if missing:
                for name in missing:
                    rejections[f"missing_provenance:{name}"] += 1
                continue
            filtered = filter_document(raw_text, filters)
            redactions.update(filtered.redactions)
            if filtered.rejection_reason is not None:
                rejections[filtered.rejection_reason] += 1
                continue
            assert filtered.text is not None
            for segment_index, text in enumerate(
                split_utf8(
                    filtered.text,
                    int(filters["max_segment_utf8_bytes"]),
                    int(filters["min_document_characters"]),
                )
            ):
                text_bytes = text.encode("utf-8")
                content_sha256 = _sha256_bytes(text_bytes)
                document_id = _record_id(
                    dataset, str(source["source_file"]), source_line, segment_index
                )
                if content_sha256 in local_hashes or _excluded(
                    connection, "exact_content", "content_sha256", content_sha256
                ):
                    rejections["exact_duplicate"] += 1
                    continue
                heldout_document = _excluded(
                    connection, "exclusion_documents", "document_id", document_id
                )
                heldout_content = _excluded(
                    connection, "exclusion_content", "content_sha256", content_sha256
                )
                if heldout_document or heldout_content:
                    rejections["heldout_exact"] += 1
                    continue
                evaluation = quality_gate.evaluate(text, str(source["category"]))
                if evaluation.rejection_reason is not None:
                    rejections[evaluation.rejection_reason] += 1
                    continue
                assert evaluation.metadata is not None
                near_sha256 = str(
                    evaluation.metadata["near_duplicate"]["fingerprint_sha256"]
                )
                if _excluded(
                    connection,
                    "exclusion_near_duplicate",
                    "fingerprint_sha256",
                    near_sha256,
                ):
                    rejections["heldout_near_duplicate"] += 1
                    continue
                quality_gate.accept(document_id, evaluation)
                connection.execute(
                    "INSERT INTO exact_content(content_sha256, document_id, source_ordinal, "
                    "source_line, segment_index) VALUES (?, ?, ?, ?, ?)",
                    (
                        content_sha256,
                        document_id,
                        source["ordinal"],
                        source_line,
                        segment_index,
                    ),
                )
                local_hashes.add(content_sha256)
                languages[evaluation.metadata["language"]["label"]] += 1
                accepted.append(
                    {
                        "content_sha256": content_sha256,
                        "document_id": document_id,
                        "evaluation": evaluation,
                        "pii_redactions": filtered.redactions,
                        "provenance": provenance,
                        "raw_sha256": filtered.raw_sha256,
                        "segment_index": segment_index,
                        "source_line": source_line,
                        "text": text,
                        "text_bytes": len(text_bytes),
                    }
                )

        encodings = tokenizer.encode_batch(
            [row["text"] for row in accepted], add_special_tokens=False
        )
        if len(encodings) != len(accepted):
            raise RuntimeError("production tokenizer returned the wrong batch size")
        total_before = connection.execute(
            "SELECT COALESCE(SUM(accepted_tokens), 0) FROM categories"
        ).fetchone()[0]
        accepted_tokens = sum(len(encoding.ids) for encoding in encodings)
        if int(total_before) + accepted_tokens > hard_max_tokens:
            raise RuntimeError("production acquisition would exceed the 150B hard maximum")

        clean_lines = []
        token_values = array("H")
        token_offset = int(shard["tokens"])
        for row, encoding in zip(accepted, encodings, strict=True):
            if any(token < 0 or token > 65_535 for token in encoding.ids):
                raise RuntimeError("production tokenizer emitted an invalid uint16 token")
            token_count = len(encoding.ids)
            record = {
                "content_sha256": row["content_sha256"],
                "id": row["document_id"],
                "provenance": {
                    "category": source["category"],
                    "component": source["component"],
                    "dataset_id": source["dataset_id"],
                    "dataset_revision": source["revision"],
                    "retained": row["provenance"],
                    "segment_index": row["segment_index"],
                    "source_file": source["source_file"],
                    "source_line": row["source_line"],
                    "source_sha256": source["source_sha256"],
                },
                "quality": row["evaluation"].metadata,
                "text": row["text"],
                "token_count": token_count,
                "token_offset": token_offset,
                "transformations": {
                    "normalization": "NFC",
                    "pii_detector": PII_DETECTOR_VERSION,
                    "pii_redactions": row["pii_redactions"],
                    "raw_sha256": row["raw_sha256"],
                },
            }
            clean_lines.append(_canonical_json_bytes(record) + b"\n")
            token_values.extend(encoding.ids)
            token_offset += token_count
        if sys.byteorder != "little":
            token_values.byteswap()
        token_bytes = token_values.tobytes()
        with clean_path.open("ab") as clean_handle, token_path.open("ab") as token_handle:
            for line in clean_lines:
                clean_handle.write(line)
            token_handle.write(token_bytes)
            clean_handle.flush()
            token_handle.flush()
            os.fsync(clean_handle.fileno())
            os.fsync(token_handle.fileno())
        clean_after = clean_path.stat().st_size
        token_after = token_path.stat().st_size
        if token_after != token_before + accepted_tokens * 2:
            raise RuntimeError("production uint16 shard byte count changed")
        accepted_documents = len(accepted)
        accepted_text_bytes = sum(int(row["text_bytes"]) for row in accepted)
        final_line = records[-1][0]
        connection.execute(
            "UPDATE sources SET status = 'running', next_line = ?, "
            "raw_documents = raw_documents + ?, raw_text_bytes = raw_text_bytes + ?, "
            "accepted_documents = accepted_documents + ?, "
            "accepted_text_bytes = accepted_text_bytes + ?, "
            "accepted_tokens = accepted_tokens + ?, rejections = ?, pii_redactions = ?, "
            "language_labels = ? WHERE ordinal = ?",
            (
                final_line,
                raw_documents,
                raw_text_bytes,
                accepted_documents,
                accepted_text_bytes,
                accepted_tokens,
                json.dumps(dict(sorted(rejections.items())), sort_keys=True),
                json.dumps(dict(sorted(redactions.items())), sort_keys=True),
                json.dumps(dict(sorted(languages.items())), sort_keys=True),
                source["ordinal"],
            ),
        )
        connection.execute(
            "UPDATE categories SET status = 'running', "
            "accepted_tokens = accepted_tokens + ?, accepted_documents = accepted_documents + ?, "
            "accepted_text_bytes = accepted_text_bytes + ? WHERE category = ?",
            (
                accepted_tokens,
                accepted_documents,
                accepted_text_bytes,
                source["category"],
            ),
        )
        connection.execute(
            "UPDATE current_shard SET clean_offset = ?, token_bytes = ?, "
            "tokens = tokens + ?, documents = documents + ?, text_bytes = text_bytes + ? "
            "WHERE singleton = 1",
            (
                clean_after,
                token_after,
                accepted_tokens,
                accepted_documents,
                accepted_text_bytes,
            ),
        )
        _metadata_set(connection, "updated_at", _utc_now())
        connection.commit()
        return {
            "accepted_documents": accepted_documents,
            "accepted_text_bytes": accepted_text_bytes,
            "accepted_tokens": accepted_tokens,
            "raw_documents": raw_documents,
            "raw_text_bytes": raw_text_bytes,
        }
    except BaseException:
        connection.rollback()
        with clean_path.open("r+b") as handle:
            handle.truncate(clean_before)
        with token_path.open("r+b") as handle:
            handle.truncate(token_before)
        raise


def _compress_zstd(source: Path, destination: Path) -> None:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("production acquisition requires the zstandard package") from error
    compressor = zstandard.ZstdCompressor(level=6, threads=-1)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        compressor.copy_stream(input_handle, output_handle, size=source.stat().st_size)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _write_provenance_parquet(clean_path: Path, destination: Path) -> set[str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("production acquisition requires pyarrow") from error
    schema = pa.schema(
        [
            ("document_id", pa.string()),
            ("content_sha256", pa.string()),
            ("category", pa.string()),
            ("component", pa.string()),
            ("dataset_id", pa.string()),
            ("dataset_revision", pa.string()),
            ("source_file", pa.string()),
            ("source_sha256", pa.string()),
            ("source_line", pa.int64()),
            ("segment_index", pa.int32()),
            ("token_offset", pa.int64()),
            ("token_count", pa.int32()),
            ("text_utf8_bytes", pa.int32()),
            ("near_duplicate_sha256", pa.string()),
            ("retained_provenance_json", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    rows: list[dict[str, Any]] = []
    source_files = set()
    try:
        with clean_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                provenance = record["provenance"]
                source_files.add(
                    f"{provenance['dataset_id']}@{provenance['dataset_revision']}:"
                    f"{provenance['source_file']}"
                )
                rows.append(
                    {
                        "category": provenance["category"],
                        "component": provenance["component"],
                        "content_sha256": record["content_sha256"],
                        "dataset_id": provenance["dataset_id"],
                        "dataset_revision": provenance["dataset_revision"],
                        "document_id": record["id"],
                        "near_duplicate_sha256": record["quality"]["near_duplicate"][
                            "fingerprint_sha256"
                        ],
                        "retained_provenance_json": json.dumps(
                            provenance["retained"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "segment_index": provenance["segment_index"],
                        "source_file": provenance["source_file"],
                        "source_line": provenance["source_line"],
                        "source_sha256": provenance["source_sha256"],
                        "text_utf8_bytes": len(record["text"].encode("utf-8")),
                        "token_count": record["token_count"],
                        "token_offset": record["token_offset"],
                    }
                )
                if len(rows) >= 50_000:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    return source_files


def _seal_current_shard(
    connection: sqlite3.Connection,
    cli: AwsCli,
    *,
    bucket: str,
    account_id: str,
    namespace: str,
    state_directory: Path,
    source_manifest_sha256: str,
    hard_logical_storage_bytes: int = 2_000_000_000_000,
) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM current_shard WHERE singleton = 1").fetchone()
    if row is None:
        return None
    if int(row["tokens"]) == 0:
        with connection:
            connection.execute("DELETE FROM current_shard WHERE singleton = 1")
        return None
    clean_path = state_directory / row["clean_path"]
    token_path = state_directory / row["token_path"]
    clean_zst = clean_path.with_suffix(".jsonl.zst")
    provenance_path = clean_path.with_suffix(".provenance.parquet")
    _compress_zstd(clean_path, clean_zst)
    source_files = _write_provenance_parquet(clean_path, provenance_path)
    clean_sha256 = _sha256_file(clean_zst)
    token_sha256 = _sha256_file(token_path)
    provenance_sha256 = _sha256_file(provenance_path)
    category = str(row["category"])
    clean_key = f"{namespace}/staging/clean/{category}/{clean_sha256}.jsonl.zst"
    token_key = f"{namespace}/staging/tokenized/{category}/{token_sha256}.bin"
    partition = f"shard-{clean_sha256[:3]}"
    provenance_key = (
        f"{namespace}/control/provenance/{partition}/{provenance_sha256}.parquet"
    )
    manifest = {
        "accepted_text_utf8_bytes": int(row["text_bytes"]),
        "category": category,
        "documents": int(row["documents"]),
        "object_key": clean_key,
        "object_bytes": clean_zst.stat().st_size,
        "object_sha256": clean_sha256,
        "provenance_index_key": provenance_key,
        "provenance_index_bytes": provenance_path.stat().st_size,
        "provenance_index_sha256": provenance_sha256,
        "removal_ledger_generation": 0,
        "schema_version": SCHEMA_VERSION,
        "shard_id": row["shard_id"],
        "source_files": sorted(source_files),
        "source_manifest_sha256": source_manifest_sha256,
        "token_dtype": "uint16_le",
        "token_object_key": token_key,
        "token_object_bytes": token_path.stat().st_size,
        "token_object_sha256": token_sha256,
        "tokens": int(row["tokens"]),
    }
    manifest_path = clean_path.with_suffix(".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    manifest_key = f"{namespace}/control/manifests/{manifest_sha256}.json"
    new_logical_bytes = sum(
        path.stat().st_size
        for path in (clean_zst, token_path, provenance_path, manifest_path)
    )
    existing_logical_bytes = int(_metadata_get(connection, "logical_storage_bytes") or "0")
    if existing_logical_bytes + new_logical_bytes > hard_logical_storage_bytes:
        raise RuntimeError("production acquisition would exceed the 2TB logical storage cap")
    for key, path in (
        (clean_key, clean_zst),
        (token_key, token_path),
        (provenance_key, provenance_path),
        (manifest_key, manifest_path),
    ):
        _put_immutable_object(
            cli,
            bucket=bucket,
            key=key,
            path=path,
            account_id=account_id,
        )
    with connection:
        connection.execute(
            "INSERT INTO shards(shard_id, sequence, category, clean_object_key, clean_sha256, "
            "token_object_key, token_sha256, provenance_object_key, provenance_sha256, "
            "manifest_object_key, manifest_sha256, documents, text_bytes, tokens, committed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["shard_id"],
                row["sequence"],
                category,
                clean_key,
                clean_sha256,
                token_key,
                token_sha256,
                provenance_key,
                provenance_sha256,
                manifest_key,
                manifest_sha256,
                row["documents"],
                row["text_bytes"],
                row["tokens"],
                _utc_now(),
            ),
        )
        connection.execute("DELETE FROM current_shard WHERE singleton = 1")
        _metadata_set(connection, "next_shard_sequence", int(row["sequence"]) + 1)
        _metadata_set(
            connection,
            "logical_storage_bytes",
            existing_logical_bytes + new_logical_bytes,
        )
        _metadata_set(connection, "updated_at", _utc_now())
    for path in (clean_path, token_path, clean_zst, provenance_path, manifest_path):
        path.unlink(missing_ok=True)
    return {**manifest, "manifest_object_key": manifest_key, "manifest_sha256": manifest_sha256}


def _progress(connection: sqlite3.Connection, release: Mapping[str, Any]) -> dict[str, Any]:
    categories = [
        dict(row)
        for row in connection.execute(
            "SELECT ordinal, category, floor_tokens, accepted_tokens, accepted_documents, "
            "accepted_text_bytes, status FROM categories ORDER BY ordinal"
        )
    ]
    current = connection.execute("SELECT * FROM current_shard WHERE singleton = 1").fetchone()
    return {
        "categories": categories,
        "current_shard": None if current is None else dict(current),
        "generated_at": _utc_now(),
        "hard_maximum_tokens": 150_000_000_000,
        "logical_storage_bytes": int(
            _metadata_get(connection, "logical_storage_bytes") or "0"
        ),
        "shards": int(connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0]),
        "source_files_complete": int(
            connection.execute(
                "SELECT COUNT(*) FROM sources WHERE status IN ('complete', 'partial_floor_reached')"
            ).fetchone()[0]
        ),
        "source_files_total": len(release["schedule"]),
        "status": _metadata_get(connection, "status"),
        "total_accepted_tokens": sum(int(row["accepted_tokens"]) for row in categories),
        "worker_release_fingerprint": release["worker_release_fingerprint"],
    }


def _upload_progress(
    connection: sqlite3.Connection,
    release: Mapping[str, Any],
    cli: AwsCli,
    *,
    bucket: str,
    account_id: str,
    namespace: str,
    state_directory: Path,
) -> dict[str, Any]:
    progress = _progress(connection, release)
    path = state_directory / "progress.json"
    _write_json_atomic(path, progress)
    digest = _sha256_file(path)
    key = f"{namespace}/control/progress/{digest}.json"
    _put_immutable_object(
        cli,
        bucket=bucket,
        key=key,
        path=path,
        account_id=account_id,
    )
    with connection:
        _metadata_set(connection, "latest_progress_key", key)
        _metadata_set(connection, "latest_progress_sha256", digest)
    return {**progress, "object_key": key, "sha256": digest}


def _state_checkpoint(
    connection: sqlite3.Connection,
    cli: AwsCli,
    *,
    bucket: str,
    account_id: str,
    namespace: str,
    state_directory: Path,
) -> dict[str, Any]:
    backup = state_directory / "state-checkpoint.sqlite3"
    compressed = state_directory / "state-checkpoint.sqlite3.zst"
    destination = sqlite3.connect(backup)
    try:
        connection.backup(destination)
    finally:
        destination.close()
    _compress_zstd(backup, compressed)
    digest = _sha256_file(compressed)
    key = f"{namespace}/control/checkpoints/{digest}.sqlite3.zst"
    result = _put_immutable_object(
        cli,
        bucket=bucket,
        key=key,
        path=compressed,
        account_id=account_id,
    )
    backup.unlink(missing_ok=True)
    compressed.unlink(missing_ok=True)
    with connection:
        _metadata_set(connection, "latest_checkpoint_key", key)
        _metadata_set(connection, "latest_checkpoint_sha256", digest)
        _metadata_set(connection, "latest_checkpoint_at", _utc_now())
    return result


def _run_production_acquisition_locked(
    contract_path: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    production_report: Path,
    state_directory: Path,
    profile: str = "default",
    maximum_runtime_seconds: int | None = None,
    batch_records: int = 32,
) -> dict[str, Any]:
    """Run or resume acquisition until every category floor or the time ceiling."""

    if batch_records < 1 or batch_records > 256:
        raise ValueError("batch_records must be in [1, 256]")
    release = build_worker_release(
        contract_path,
        repository_root=repository_root,
        tokenizer_path=tokenizer_path,
        training_exclusion_ledger=training_exclusion_ledger,
        production_report=production_report,
    )
    production = _read_json(production_report)
    bucket = str(production["bucket_identity"]["data_bucket"])
    namespace = str(production["provider"]["namespace"])
    cli = AwsCli(profile, str(production["provider"]["region"]))
    _, identity, _ = cli.call("sts", "get-caller-identity")
    account_id = str(identity["Account"])
    expected_account_hash = _sha256_bytes(account_id.encode())
    if expected_account_hash != production["bucket_identity"]["account_id_sha256"]:
        raise ValueError("production worker is running in the wrong AWS account")
    state_directory.mkdir(parents=True, exist_ok=True)
    scratch = state_directory / "scratch"
    scratch.mkdir(exist_ok=True)
    connection = _open_state(state_directory / "acquisition-state.sqlite3")
    quality_config = _read_json(
        _resolve(
            repository_root,
            release["inputs"]["data_quality"]["path"],
            "data quality",
        )
    )
    inventory = _read_json(
        _resolve(
            repository_root,
            release["inputs"]["component_inventory"]["path"],
            "component inventory",
        )
    )
    provenance = _component_provenance(inventory)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 65_536:
        raise ValueError("production tokenizer vocabulary size changed")
    filters = _read_json(
        _resolve(
            repository_root,
            release["inputs"]["filter_policy"]["path"],
            "filter policy",
        )
    ).get("filters")
    if not isinstance(filters, Mapping):
        raise ValueError("production filter policy is missing")
    quality_gate = DataQualityGate(
        quality_config,
        repository_root,
        near_duplicate_connection=connection,
        near_duplicate_table_prefix="production_near_duplicate",
    )
    started_monotonic = time.monotonic()
    stop_deadline = (
        None
        if maximum_runtime_seconds is None
        else started_monotonic + maximum_runtime_seconds
    )
    scratch_cap = int(release["acquisition"]["worker_local_scratch_hard_cap_bytes"])
    shard_target = int(release["acquisition"]["clean_shard_target_tokens"])
    hard_max_tokens = int(release["limits"]["candidate_hard_maximum_tokens"])
    hard_logical_storage_bytes = int(release["limits"]["hard_logical_storage_bytes"])
    try:
        _initialize_state(connection, release, training_exclusion_ledger)
        _upload_progress(
            connection,
            release,
            cli,
            bucket=bucket,
            account_id=account_id,
            namespace=namespace,
            state_directory=state_directory,
        )
        schedule_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source in release["schedule"]:
            schedule_by_category[source["category"]].append(source)
        for category_contract in release["categories"]:
            category = str(category_contract["category"])
            category_row = connection.execute(
                "SELECT * FROM categories WHERE category = ?", (category,)
            ).fetchone()
            assert category_row is not None
            if int(category_row["accepted_tokens"]) >= int(category_row["floor_tokens"]):
                with connection:
                    connection.execute(
                        "UPDATE categories SET status = 'complete' WHERE category = ?",
                        (category,),
                    )
                continue
            for source in schedule_by_category[category]:
                source_row = connection.execute(
                    "SELECT * FROM sources WHERE ordinal = ?", (source["ordinal"],)
                ).fetchone()
                assert source_row is not None
                if source_row["status"] in {"complete", "partial_floor_reached"}:
                    continue
                if stop_deadline is not None and time.monotonic() >= stop_deadline - 900:
                    raise TimeoutError("production worker reached its safe runtime checkpoint")
                raw_path = _download_source(source, scratch, scratch_cap)
                next_line = int(source_row["next_line"])
                pending: list[tuple[int, Mapping[str, Any] | None]] = []
                floor_reached = False
                for line_number, record in _iter_gzip_records(raw_path):
                    if line_number <= next_line:
                        continue
                    pending.append((line_number, record))
                    if len(pending) < batch_records:
                        continue
                    source_row = connection.execute(
                        "SELECT * FROM sources WHERE ordinal = ?", (source["ordinal"],)
                    ).fetchone()
                    assert source_row is not None
                    _process_batch(
                        connection,
                        quality_gate,
                        tokenizer,
                        source=source,
                        source_provenance=provenance[source["component"]],
                        source_row=source_row,
                        records=pending,
                        filters=filters,
                        state_directory=state_directory,
                        hard_max_tokens=hard_max_tokens,
                    )
                    pending = []
                    current = connection.execute(
                        "SELECT tokens FROM current_shard WHERE singleton = 1"
                    ).fetchone()
                    if current is not None and int(current["tokens"]) >= shard_target:
                        _seal_current_shard(
                            connection,
                            cli,
                            bucket=bucket,
                            account_id=account_id,
                            namespace=namespace,
                            state_directory=state_directory,
                            source_manifest_sha256=release["inputs"][
                                "source_file_manifest"
                            ]["sha256"],
                            hard_logical_storage_bytes=hard_logical_storage_bytes,
                        )
                        _upload_progress(
                            connection,
                            release,
                            cli,
                            bucket=bucket,
                            account_id=account_id,
                            namespace=namespace,
                            state_directory=state_directory,
                        )
                    category_row = connection.execute(
                        "SELECT * FROM categories WHERE category = ?", (category,)
                    ).fetchone()
                    assert category_row is not None
                    if int(category_row["accepted_tokens"]) >= int(
                        category_row["floor_tokens"]
                    ):
                        floor_reached = True
                        break
                    if stop_deadline is not None and time.monotonic() >= stop_deadline - 900:
                        raise TimeoutError("production worker reached its safe runtime checkpoint")
                if pending and not floor_reached:
                    source_row = connection.execute(
                        "SELECT * FROM sources WHERE ordinal = ?", (source["ordinal"],)
                    ).fetchone()
                    assert source_row is not None
                    _process_batch(
                        connection,
                        quality_gate,
                        tokenizer,
                        source=source,
                        source_provenance=provenance[source["component"]],
                        source_row=source_row,
                        records=pending,
                        filters=filters,
                        state_directory=state_directory,
                        hard_max_tokens=hard_max_tokens,
                    )
                    current = connection.execute(
                        "SELECT tokens FROM current_shard WHERE singleton = 1"
                    ).fetchone()
                    if current is not None and int(current["tokens"]) >= shard_target:
                        _seal_current_shard(
                            connection,
                            cli,
                            bucket=bucket,
                            account_id=account_id,
                            namespace=namespace,
                            state_directory=state_directory,
                            source_manifest_sha256=release["inputs"][
                                "source_file_manifest"
                            ]["sha256"],
                            hard_logical_storage_bytes=hard_logical_storage_bytes,
                        )
                    category_row = connection.execute(
                        "SELECT * FROM categories WHERE category = ?", (category,)
                    ).fetchone()
                    assert category_row is not None
                    floor_reached = int(category_row["accepted_tokens"]) >= int(
                        category_row["floor_tokens"]
                    )
                with connection:
                    connection.execute(
                        "UPDATE sources SET status = ? WHERE ordinal = ?",
                        (
                            "partial_floor_reached" if floor_reached else "complete",
                            source["ordinal"],
                        ),
                    )
                    _metadata_set(connection, "updated_at", _utc_now())
                raw_path.unlink(missing_ok=True)
                if floor_reached:
                    _seal_current_shard(
                        connection,
                        cli,
                        bucket=bucket,
                        account_id=account_id,
                        namespace=namespace,
                        state_directory=state_directory,
                        source_manifest_sha256=release["inputs"]["source_file_manifest"][
                            "sha256"
                        ],
                        hard_logical_storage_bytes=hard_logical_storage_bytes,
                    )
                    with connection:
                        connection.execute(
                            "UPDATE categories SET status = 'complete' WHERE category = ?",
                            (category,),
                        )
                    _upload_progress(
                        connection,
                        release,
                        cli,
                        bucket=bucket,
                        account_id=account_id,
                        namespace=namespace,
                        state_directory=state_directory,
                    )
                    break
                _upload_progress(
                    connection,
                    release,
                    cli,
                    bucket=bucket,
                    account_id=account_id,
                    namespace=namespace,
                    state_directory=state_directory,
                )
            category_row = connection.execute(
                "SELECT * FROM categories WHERE category = ?", (category,)
            ).fetchone()
            assert category_row is not None
            if int(category_row["accepted_tokens"]) < int(category_row["floor_tokens"]):
                raise RuntimeError(f"production category exhausted before its floor: {category}")
        _seal_current_shard(
            connection,
            cli,
            bucket=bucket,
            account_id=account_id,
            namespace=namespace,
            state_directory=state_directory,
            source_manifest_sha256=release["inputs"]["source_file_manifest"]["sha256"],
            hard_logical_storage_bytes=hard_logical_storage_bytes,
        )
        with connection:
            _metadata_set(connection, "status", "complete")
            _metadata_set(connection, "completed_at", _utc_now())
            _metadata_set(connection, "updated_at", _utc_now())
    except TimeoutError:
        _seal_current_shard(
            connection,
            cli,
            bucket=bucket,
            account_id=account_id,
            namespace=namespace,
            state_directory=state_directory,
            source_manifest_sha256=release["inputs"]["source_file_manifest"]["sha256"],
            hard_logical_storage_bytes=hard_logical_storage_bytes,
        )
        with connection:
            _metadata_set(connection, "status", "paused_at_runtime_cap")
            _metadata_set(connection, "updated_at", _utc_now())
    except BaseException as error:
        with connection:
            _metadata_set(connection, "status", "failed")
            _metadata_set(
                connection,
                "failure",
                json.dumps({"message": str(error), "type": type(error).__name__}),
            )
            _metadata_set(connection, "updated_at", _utc_now())
        raise
    finally:
        for raw_path in scratch.glob("source-*.json.gz"):
            raw_path.unlink(missing_ok=True)
        for partial_path in scratch.glob("source-*.json.gz.part"):
            partial_path.unlink(missing_ok=True)
        quality_gate.flush()
        _upload_progress(
            connection,
            release,
            cli,
            bucket=bucket,
            account_id=account_id,
            namespace=namespace,
            state_directory=state_directory,
        )
        _state_checkpoint(
            connection,
            cli,
            bucket=bucket,
            account_id=account_id,
            namespace=namespace,
            state_directory=state_directory,
        )
        quality_gate.close()
        connection.close()
    with sqlite3.connect(state_directory / "acquisition-state.sqlite3") as final_connection:
        final_connection.row_factory = sqlite3.Row
        return _progress(final_connection, release)


def run_production_acquisition(
    contract_path: Path,
    *,
    repository_root: Path,
    tokenizer_path: Path,
    training_exclusion_ledger: Path,
    production_report: Path,
    state_directory: Path,
    profile: str = "default",
    maximum_runtime_seconds: int | None = None,
    batch_records: int = 32,
) -> dict[str, Any]:
    """Run one exclusive, resumable acquisition worker for the state directory."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - production worker is Linux/macOS only
        raise RuntimeError("production acquisition requires POSIX file locking") from error

    state_directory.mkdir(parents=True, exist_ok=True)
    lock_path = state_directory / "acquisition.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another production acquisition worker owns {lock_path}"
            ) from error
        return _run_production_acquisition_locked(
            contract_path,
            repository_root=repository_root,
            tokenizer_path=tokenizer_path,
            training_exclusion_ledger=training_exclusion_ledger,
            production_report=production_report,
            state_directory=state_directory,
            profile=profile,
            maximum_runtime_seconds=maximum_runtime_seconds,
            batch_records=batch_records,
        )


def local_worker_status(state_directory: Path, release: Mapping[str, Any]) -> dict[str, Any]:
    database = state_directory / "acquisition-state.sqlite3"
    if not database.is_file():
        return {"status": "not_started"}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return _progress(connection, release)
    finally:
        connection.close()
