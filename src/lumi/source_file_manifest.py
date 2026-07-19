"""Resumable metadata-only source-file manifest for the Lumi corpus."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
DATABASE_NAME = "state.sqlite3"
SOURCE_FILES_NAME = "source-files.jsonl"
SNAPSHOT_NAME = "snapshot.json"
REPORT_NAME = "report.json"
MANIFEST_NAME = "manifest.sha256"
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_AUTHORIZATIONS = {
    "bulk_corpus_materialization",
    "create_cloud_resources",
    "full_4b_training",
    "paid_compute",
    "transfer_source_payload",
}
FORBIDDEN_METADATA_KEYS = {
    "access_key",
    "api_key",
    "credentials",
    "private_key",
    "raw_text",
    "secret_key",
    "session_token",
    "text",
}
IMPLEMENTATION_FILES = (
    "src/lumi/source_file_manifest.py",
    "scripts/run_source_file_manifest.py",
    "scripts/verify_source_file_manifest.py",
)

TreeFetcher = Callable[[str, str], Iterable[Any]]
ProgressCallback = Callable[[str], None]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_bytes_atomic(path, encoded.encode("utf-8"))


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _safe_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is unsafe: {value}")
    return value


def _resolve(repository_root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (repository_root / raw_path).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _output_path(repository_root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = (repository_root / raw_path).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} escapes the repository")
    return path


def _unique_by(
    items: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} has an invalid {key}")
        if value in result:
            raise ValueError(f"duplicate {label} {key}: {value}")
        result[value] = item
    return result


def _validate_no_payload_or_credentials(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_METADATA_KEYS:
                raise ValueError(f"{label} contains forbidden field {key}")
            _validate_no_payload_or_credentials(child, label)
    elif isinstance(value, list):
        for child in value:
            _validate_no_payload_or_credentials(child, label)


def _implementation_hashes(repository_root: Path) -> dict[str, str]:
    result = {}
    for raw_path in IMPLEMENTATION_FILES:
        path = repository_root / raw_path
        if not path.is_file():
            raise ValueError(f"missing source-manifest implementation: {path}")
        result[raw_path] = _sha256_file(path)
    return result


def _category_map(mixture: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    rows = mixture.get("mixture")
    if not isinstance(rows, list) or not rows:
        raise ValueError("mixture has no categories")
    for row in rows:
        category = row.get("category")
        components = row.get("source_components")
        if not isinstance(category, str) or not category:
            raise ValueError("mixture category is invalid")
        if not isinstance(components, list) or not components:
            raise ValueError(f"mixture category has no components: {category}")
        for component in components:
            if not isinstance(component, str) or not component:
                raise ValueError(f"mixture component is invalid in {category}")
            if component in result:
                raise ValueError(f"mixture repeats component {component}")
            result[component] = category
    return result


def _load_runtime(contract_path: Path, repository_root: Path) -> dict[str, Any]:
    specification = _read_json(contract_path)
    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source-file manifest schema")
    authorization = specification.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("source-file manifest authorization is missing")
    if authorization.get("fetch_public_repository_metadata") is not True:
        raise ValueError("public metadata fetch authorization must be explicit")
    if any(authorization.get(name) is not False for name in FORBIDDEN_AUTHORIZATIONS):
        raise ValueError(
            "source-file manifest must not authorize payload, cloud, compute, or training"
        )

    artifact_policy = specification.get("artifact_policy")
    if not isinstance(artifact_policy, dict):
        raise ValueError("source-file manifest artifact policy is missing")
    if artifact_policy.get("persist_corpus_payload") is not False:
        raise ValueError("source-file manifest must not persist corpus payload")
    if artifact_policy.get("persist_credentials") is not False:
        raise ValueError("source-file manifest must not persist credentials")
    maximum_artifact_bytes = _positive_int(
        artifact_policy.get("maximum_artifact_bytes"), "maximum artifact bytes"
    )
    if maximum_artifact_bytes > 100_000_000:
        raise ValueError("source-file metadata artifact cap exceeds 100MB")
    _positive_int(
        artifact_policy.get("minimum_free_bytes_before_start"),
        "minimum free bytes before start",
    )

    api = specification.get("metadata_api")
    if not isinstance(api, dict):
        raise ValueError("metadata API configuration is missing")
    if api != {
        "endpoint": "https://huggingface.co",
        "expand": False,
        "recursive": True,
        "repo_type": "dataset",
        "token": False,
    }:
        raise ValueError("metadata API must remain public, recursive, and metadata-only")

    tree_rules = specification.get("tree_rules")
    if not isinstance(tree_rules, dict):
        raise ValueError("tree rules are missing")
    suffixes = tree_rules.get("data_file_suffixes")
    if not isinstance(suffixes, list) or not suffixes or len(suffixes) != len(set(suffixes)):
        raise ValueError("data file suffixes are invalid")
    if not all(isinstance(suffix, str) and suffix.startswith(".") for suffix in suffixes):
        raise ValueError("data file suffixes must be explicit extensions")
    controls = tree_rules.get("allowed_control_files")
    if not isinstance(controls, list) or len(controls) != len(set(controls)):
        raise ValueError("allowed control files are invalid")
    for control in controls:
        _safe_path(control, "allowed control file")
    for field in (
        "fail_on_unknown_file",
        "require_git_blob_sha1",
        "require_git_lfs_sha256",
        "require_lfs_size_match",
    ):
        if tree_rules.get(field) is not True:
            raise ValueError(f"tree rule must remain enabled: {field}")

    inputs = specification.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "capacity_snapshot",
        "mixture",
        "source_inventory",
    }:
        raise ValueError("source-file manifest inputs changed")
    paths = {
        name: _resolve(repository_root, raw_path, name) for name, raw_path in inputs.items()
    }
    inventory_document = _read_json(paths["source_inventory"])
    capacity_document = _read_json(paths["capacity_snapshot"])
    mixture_document = _read_json(paths["mixture"])
    inventory = _unique_by(inventory_document.get("sources", []), "component", "source inventory")
    capacity = _unique_by(capacity_document.get("sources", []), "component", "capacity source")
    categories = _category_map(mixture_document)
    if set(inventory) != set(capacity) or set(inventory) != set(categories):
        raise ValueError("inventory, capacity snapshot, and mixture component sets differ")

    sources = []
    for component in sorted(inventory):
        source = inventory[component]
        measured = capacity[component]
        for field in ("dataset_id", "revision"):
            if source.get(field) != measured.get(field):
                raise ValueError(f"capacity {field} drifted for {component}")
        revision = source.get("revision")
        if not isinstance(revision, str) or SHA1_PATTERN.fullmatch(revision) is None:
            raise ValueError(f"source revision is not a pinned commit: {component}")
        expected_files = _positive_int(source.get("data_file_count"), f"{component} files")
        if expected_files != _positive_int(
            measured.get("tree_data_file_count"), f"{component} capacity files"
        ):
            raise ValueError(f"data file count drifted for {component}")
        expected_bytes = _positive_int(
            measured.get("tree_compressed_bytes"), f"{component} compressed bytes"
        )
        schema_anchor = _safe_path(source.get("schema_anchor"), f"{component} schema anchor")
        provenance_paths = sorted(
            {
                "id",
                "metadata.provenance",
                "source",
                *source.get("license_paths", []),
                *source.get("url_paths", []),
            }
        )
        if not all(isinstance(path, str) and path for path in provenance_paths):
            raise ValueError(f"provenance paths are invalid for {component}")
        source_level_basis = "source_level_basis" in str(source.get("status", ""))
        acquisition_status = (
            "blocked_pending_owner_cccc_license_decision_and_production_authorization"
            if source_level_basis
            else "blocked_pending_production_authorization"
        )
        sources.append(
            {
                "acquisition_status": acquisition_status,
                "category": categories[component],
                "component": component,
                "dataset_id": source["dataset_id"],
                "expected_bytes": expected_bytes,
                "expected_files": expected_files,
                "license_policy_status": source["status"],
                "provenance_paths": provenance_paths,
                "revision": revision,
                "schema_anchor": schema_anchor,
            }
        )

    expected_totals = specification.get("expected_totals")
    if not isinstance(expected_totals, dict):
        raise ValueError("source-file manifest expected totals are missing")
    observed_totals = {
        "compressed_source_bytes": sum(source["expected_bytes"] for source in sources),
        "data_files": sum(source["expected_files"] for source in sources),
        "sources": len(sources),
    }
    if expected_totals != observed_totals:
        raise ValueError("source-file manifest expected totals drifted")

    output = specification.get("output")
    if not isinstance(output, dict) or set(output) != {"repository_snapshot"}:
        raise ValueError("source-file manifest output contract changed")
    repository_snapshot = _output_path(
        repository_root, output["repository_snapshot"], "repository snapshot"
    )
    remaining_blockers = specification.get("remaining_authorization_blockers")
    if not isinstance(remaining_blockers, list) or set(remaining_blockers) != {
        "owner_decision_for_cccc_source_level_license_basis",
        "production_s3_namespace_and_adaptive_acquisition_approval",
    }:
        raise ValueError("remaining source acquisition blockers changed")

    input_hashes = {
        name: {"path": str(path.relative_to(repository_root)), "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    contract_sha256 = _sha256_file(contract_path)
    run_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "contract_sha256": contract_sha256,
                "inputs": input_hashes,
                "sources": sources,
                "tree_rules": tree_rules,
            }
        )
    )
    return {
        "api": api,
        "artifact_policy": artifact_policy,
        "authorization": authorization,
        "contract_path": str(contract_path.relative_to(repository_root)),
        "contract_sha256": contract_sha256,
        "expected_totals": observed_totals,
        "input_hashes": input_hashes,
        "remaining_blockers": remaining_blockers,
        "repository_snapshot": repository_snapshot,
        "run_fingerprint": run_fingerprint,
        "sources": sources,
        "specification": specification,
        "tree_rules": tree_rules,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_tree_entry(entry: Any, rules: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = _field(entry, "type")
    if kind is None:
        class_name = type(entry).__name__
        kind = "directory" if class_name == "RepoFolder" else "file"
    path = _safe_path(_field(entry, "path"), "repository tree path")
    if kind in {"directory", "folder"}:
        return None
    if kind not in {"file", None}:
        raise ValueError(f"unsupported repository tree entry type for {path}: {kind}")

    size = _nonnegative_int(_field(entry, "size"), f"repository file size for {path}")
    blob_id = _field(entry, "blob_id", _field(entry, "oid"))
    if not isinstance(blob_id, str) or SHA1_PATTERN.fullmatch(blob_id) is None:
        raise ValueError(f"repository file is missing a Git blob SHA-1: {path}")
    lfs = _field(entry, "lfs")
    xet_hash = _field(entry, "xet_hash")
    if xet_hash is not None and (
        not isinstance(xet_hash, str) or SHA256_PATTERN.fullmatch(xet_hash) is None
    ):
        raise ValueError(f"repository file has an invalid Xet hash: {path}")

    suffixes = tuple(rules["data_file_suffixes"])
    if path.endswith(suffixes):
        if size == 0:
            raise ValueError(f"source data file is empty: {path}")
        if lfs is None:
            raise ValueError(f"source data file has no LFS metadata: {path}")
        lfs_size = _positive_int(_field(lfs, "size"), f"LFS size for {path}")
        lfs_sha256 = _field(lfs, "sha256")
        if not isinstance(lfs_sha256, str) or SHA256_PATTERN.fullmatch(lfs_sha256) is None:
            raise ValueError(f"source data file has no LFS SHA-256: {path}")
        if lfs_size != size:
            raise ValueError(f"source data file LFS size differs from tree size: {path}")
        return {
            "blob_id": blob_id,
            "kind": "data",
            "lfs_sha256": lfs_sha256,
            "path": path,
            "size": size,
            "xet_hash": xet_hash,
        }

    if path not in set(rules["allowed_control_files"]):
        raise ValueError(f"unknown non-data repository file: {path}")
    return {
        "blob_id": blob_id,
        "kind": "control",
        "lfs_sha256": None,
        "path": path,
        "size": size,
        "xet_hash": xet_hash,
    }


def validate_source_tree(
    source: Mapping[str, Any], entries: Iterable[Any], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Normalize one pinned repository tree and fail on any incomplete metadata."""

    normalized = []
    paths: set[str] = set()
    for entry in entries:
        row = _normalize_tree_entry(entry, rules)
        if row is None:
            continue
        if row["path"] in paths:
            raise ValueError(f"repository tree repeats path: {row['path']}")
        paths.add(row["path"])
        normalized.append(row)
    normalized.sort(key=lambda row: row["path"])
    data_rows = [row for row in normalized if row["kind"] == "data"]
    if len(data_rows) != source["expected_files"]:
        raise ValueError(
            f"data file count differs for {source['component']}: "
            f"{len(data_rows)} != {source['expected_files']}"
        )
    source_bytes = sum(row["size"] for row in data_rows)
    if source_bytes != source["expected_bytes"]:
        raise ValueError(
            f"compressed byte total differs for {source['component']}: "
            f"{source_bytes} != {source['expected_bytes']}"
        )
    if source["schema_anchor"] not in {row["path"] for row in data_rows}:
        raise ValueError(f"schema anchor is absent for {source['component']}")
    return normalized


def _default_fetcher(runtime: Mapping[str, Any]) -> TreeFetcher:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required for the public metadata fetch; "
            "install the benchmark-build extra"
        ) from error

    api = HfApi(endpoint=runtime["api"]["endpoint"])

    def fetch(dataset_id: str, revision: str) -> Iterable[Any]:
        return api.list_repo_tree(
            dataset_id,
            recursive=True,
            expand=False,
            revision=revision,
            repo_type="dataset",
            token=False,
        )

    return fetch


@contextmanager
def _exclusive_lock(path: Path) -> Iterable[None]:
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another source-file manifest process holds the run lock") from error
        yield


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS manifest_metadata ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS manifest_sources ("
        "component TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, revision TEXT NOT NULL, "
        "expected_files INTEGER NOT NULL, expected_bytes INTEGER NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', observed_files INTEGER, "
        "observed_bytes INTEGER, control_files INTEGER, last_error TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS manifest_files ("
        "component TEXT NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL, "
        "size INTEGER NOT NULL, blob_id TEXT NOT NULL, lfs_sha256 TEXT, xet_hash TEXT, "
        "PRIMARY KEY (component, path), "
        "FOREIGN KEY (component) REFERENCES manifest_sources(component))"
    )
    connection.commit()
    return connection


def _metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM manifest_metadata WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else str(row[0])


def _metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO manifest_metadata (key, value) VALUES (?, ?)",
        (key, str(value)),
    )


def _initialize_state(connection: sqlite3.Connection, runtime: Mapping[str, Any]) -> None:
    fingerprint = _metadata_get(connection, "run_fingerprint")
    if fingerprint is not None and fingerprint != runtime["run_fingerprint"]:
        raise ValueError("source-file manifest state belongs to a different pinned run")
    if fingerprint is None:
        _metadata_set(connection, "schema_version", SCHEMA_VERSION)
        _metadata_set(connection, "run_fingerprint", runtime["run_fingerprint"])
    for source in runtime["sources"]:
        existing = connection.execute(
            "SELECT dataset_id, revision, expected_files, expected_bytes "
            "FROM manifest_sources WHERE component = ?",
            (source["component"],),
        ).fetchone()
        expected = (
            source["dataset_id"],
            source["revision"],
            source["expected_files"],
            source["expected_bytes"],
        )
        if existing is None:
            connection.execute(
                "INSERT INTO manifest_sources "
                "(component, dataset_id, revision, expected_files, expected_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (source["component"], *expected),
            )
        elif tuple(existing) != expected:
            raise ValueError(f"source-file manifest state drifted for {source['component']}")
    known = {
        str(row[0]) for row in connection.execute("SELECT component FROM manifest_sources")
    }
    expected_components = {source["component"] for source in runtime["sources"]}
    if known != expected_components:
        raise ValueError("source-file manifest state contains an unexpected component")
    connection.commit()


def _rows_for_source(
    connection: sqlite3.Connection, source: Mapping[str, Any], rules: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stored = connection.execute(
        "SELECT path, kind, size, blob_id, lfs_sha256, xet_hash "
        "FROM manifest_files WHERE component = ? ORDER BY path",
        (source["component"],),
    ).fetchall()
    pseudo_entries = []
    for row in stored:
        if row["kind"] == "data":
            lfs = {"sha256": row["lfs_sha256"], "size": row["size"]}
        elif row["kind"] == "control":
            lfs = None
        else:
            raise ValueError(f"state has an invalid file kind for {source['component']}")
        pseudo_entries.append(
            {
                "blob_id": row["blob_id"],
                "lfs": lfs,
                "path": row["path"],
                "size": row["size"],
                "type": "file",
                "xet_hash": row["xet_hash"],
            }
        )
    return validate_source_tree(source, pseudo_entries, rules)


def _fetch_with_retries(
    fetcher: TreeFetcher,
    source: Mapping[str, Any],
    *,
    retries: int,
    retry_delay_seconds: float,
) -> list[Any]:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry delay must be non-negative")
    for attempt in range(retries + 1):
        try:
            return list(fetcher(source["dataset_id"], source["revision"]))
        except Exception as error:
            if attempt >= retries:
                raise RuntimeError(
                    f"metadata fetch failed for {source['component']} after {attempt + 1} attempts"
                ) from error
            time.sleep(retry_delay_seconds * (2**attempt))
    raise AssertionError("unreachable metadata retry state")


def _store_source_rows(
    connection: sqlite3.Connection,
    source: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    data_rows = [row for row in rows if row["kind"] == "data"]
    control_rows = [row for row in rows if row["kind"] == "control"]
    with connection:
        connection.execute(
            "DELETE FROM manifest_files WHERE component = ?", (source["component"],)
        )
        connection.executemany(
            "INSERT INTO manifest_files "
            "(component, path, kind, size, blob_id, lfs_sha256, xet_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    source["component"],
                    row["path"],
                    row["kind"],
                    row["size"],
                    row["blob_id"],
                    row["lfs_sha256"],
                    row["xet_hash"],
                )
                for row in rows
            ],
        )
        connection.execute(
            "UPDATE manifest_sources SET status = 'complete', observed_files = ?, "
            "observed_bytes = ?, control_files = ?, last_error = NULL WHERE component = ?",
            (
                len(data_rows),
                sum(row["size"] for row in data_rows),
                len(control_rows),
                source["component"],
            ),
        )


def _process_sources(
    connection: sqlite3.Connection,
    runtime: Mapping[str, Any],
    fetcher: TreeFetcher,
    *,
    retries: int,
    retry_delay_seconds: float,
    progress: ProgressCallback,
) -> None:
    total = len(runtime["sources"])
    for index, source in enumerate(runtime["sources"], start=1):
        status_row = connection.execute(
            "SELECT status FROM manifest_sources WHERE component = ?",
            (source["component"],),
        ).fetchone()
        if status_row is None:
            raise ValueError(f"source-file state lost {source['component']}")
        if status_row["status"] == "complete":
            rows = _rows_for_source(connection, source, runtime["tree_rules"])
            progress(
                f"[{index}/{total}] resume verified {source['component']} "
                f"({sum(row['kind'] == 'data' for row in rows)} files)"
            )
            continue
        progress(f"[{index}/{total}] fetching metadata for {source['component']}")
        try:
            entries = _fetch_with_retries(
                fetcher,
                source,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            rows = validate_source_tree(source, entries, runtime["tree_rules"])
            _store_source_rows(connection, source, rows)
        except Exception as error:
            with connection:
                connection.execute(
                    "UPDATE manifest_sources SET status = 'pending', last_error = ? "
                    "WHERE component = ?",
                    (f"{type(error).__name__}: {error}", source["component"]),
                )
            raise
        progress(
            f"[{index}/{total}] sealed {source['component']} "
            f"({sum(row['kind'] == 'data' for row in rows)} files)"
        )


def _data_file_record(source: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acquisition_authorized": False,
        "acquisition_status": source["acquisition_status"],
        "category": source["category"],
        "component": source["component"],
        "dataset_id": source["dataset_id"],
        "dataset_revision": source["revision"],
        "git_blob_sha1": row["blob_id"],
        "license_policy_status": source["license_policy_status"],
        "provenance_paths": source["provenance_paths"],
        "schema_version": SCHEMA_VERSION,
        "source_bytes": row["size"],
        "source_digest": {
            "algorithm": "sha256",
            "basis": "git_lfs_object",
            "value": row["lfs_sha256"],
        },
        "source_file": row["path"],
        "xet_hash": row["xet_hash"],
    }


def _build_snapshot(
    connection: sqlite3.Connection, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    source_rows = []
    total_control_files = 0
    for source in runtime["sources"]:
        status = connection.execute(
            "SELECT status, observed_files, observed_bytes, control_files "
            "FROM manifest_sources WHERE component = ?",
            (source["component"],),
        ).fetchone()
        if status is None or status["status"] != "complete":
            raise RuntimeError(f"source metadata is incomplete for {source['component']}")
        rows = _rows_for_source(connection, source, runtime["tree_rules"])
        data_rows = [row for row in rows if row["kind"] == "data"]
        control_rows = [row for row in rows if row["kind"] == "control"]
        if (
            status["observed_files"] != len(data_rows)
            or status["observed_bytes"] != sum(row["size"] for row in data_rows)
            or status["control_files"] != len(control_rows)
        ):
            raise ValueError(f"source state summary differs for {source['component']}")
        total_control_files += len(control_rows)
        source_rows.append(
            {
                "acquisition_status": source["acquisition_status"],
                "category": source["category"],
                "component": source["component"],
                "compressed_source_bytes": sum(row["size"] for row in data_rows),
                "data_file_count": len(data_rows),
                "data_files": [
                    {
                        "git_blob_sha1": row["blob_id"],
                        "source_bytes": row["size"],
                        "source_digest": {
                            "algorithm": "sha256",
                            "basis": "git_lfs_object",
                            "value": row["lfs_sha256"],
                        },
                        "source_file": row["path"],
                        "xet_hash": row["xet_hash"],
                    }
                    for row in data_rows
                ],
                "dataset_id": source["dataset_id"],
                "dataset_revision": source["revision"],
                "license_policy_status": source["license_policy_status"],
                "provenance_paths": source["provenance_paths"],
                "repository_control_files": [
                    {
                        "git_blob_sha1": row["blob_id"],
                        "path": row["path"],
                        "size": row["size"],
                        "xet_hash": row["xet_hash"],
                    }
                    for row in control_rows
                ],
                "schema_anchor": source["schema_anchor"],
            }
        )
    snapshot = {
        "authorization": dict(runtime["authorization"]),
        "metadata_source": {
            "endpoint": runtime["api"]["endpoint"],
            "expand": False,
            "method": "huggingface_hub.HfApi.list_repo_tree",
            "recursive": True,
            "repo_type": "dataset",
            "revision_policy": "exact_40_character_commit_sha",
            "token": False,
        },
        "remaining_authorization_blockers": list(runtime["remaining_blockers"]),
        "run_fingerprint": runtime["run_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "sources": source_rows,
        "status": "complete_metadata_only_no_payload_authorized",
        "summary": {
            **runtime["expected_totals"],
            "corpus_payload_bytes_downloaded": 0,
            "repository_control_files": total_control_files,
        },
        "verified_at": runtime["specification"]["verified_at"],
        "version": runtime["specification"]["version"],
    }
    _validate_no_payload_or_credentials(snapshot, "source-file snapshot")
    return snapshot


def _flatten_snapshot(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for source in snapshot["sources"]:
        source_context = {
            "acquisition_status": source["acquisition_status"],
            "category": source["category"],
            "component": source["component"],
            "dataset_id": source["dataset_id"],
            "license_policy_status": source["license_policy_status"],
            "provenance_paths": source["provenance_paths"],
            "revision": source["dataset_revision"],
        }
        for row in source["data_files"]:
            normalized = {
                "blob_id": row["git_blob_sha1"],
                "lfs_sha256": row["source_digest"]["value"],
                "path": row["source_file"],
                "size": row["source_bytes"],
                "xet_hash": row["xet_hash"],
            }
            records.append(_data_file_record(source_context, normalized))
    records.sort(key=lambda row: (row["component"], row["source_file"]))
    return records


def _source_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def _build_report(
    runtime: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    database_path: Path,
    source_files_path: Path,
    snapshot_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    summary = dict(snapshot["summary"])
    report = {
        "authorization": dict(runtime["authorization"]),
        "decision": {
            "bulk_corpus_materialization": "not_authorized",
            "cloud_resource_creation": "not_authorized",
            "complete_source_file_manifest": "pass",
            "full_4b_training": "not_authorized",
            "next_action": "owner_cccc_license_decision_then_production_acquisition_approval",
            "paid_compute": "not_authorized",
            "source_payload_transfer": "not_authorized",
        },
        "evidence": {
            "database_bytes": database_path.stat().st_size,
            "database_sha256": _sha256_file(database_path),
            "repository_snapshot_sha256": _sha256_file(runtime["repository_snapshot"]),
            "snapshot_bytes": snapshot_path.stat().st_size,
            "snapshot_sha256": _sha256_file(snapshot_path),
            "source_files_jsonl_bytes": source_files_path.stat().st_size,
            "source_files_jsonl_sha256": _sha256_file(source_files_path),
        },
        "gate": runtime["specification"]["gate"],
        "implementation": _implementation_hashes(repository_root),
        "inputs": {
            "contract": {
                "path": runtime["contract_path"],
                "sha256": runtime["contract_sha256"],
            },
            **runtime["input_hashes"],
        },
        "metadata_only_guarantees": {
            "api_expand": False,
            "api_token": False,
            "corpus_payload_bytes_downloaded": 0,
            "persist_corpus_payload": False,
            "persist_credentials": False,
            "upstream_content_identity": "git_lfs_sha256",
        },
        "remaining_authorization_blockers": list(runtime["remaining_blockers"]),
        "run_fingerprint": runtime["run_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "status": "pass_metadata_complete_acquisition_still_blocked",
        "summary": summary,
        "verified_at": runtime["specification"]["verified_at"],
        "version": runtime["specification"]["version"],
    }
    report["report_fingerprint"] = _sha256_bytes(
        _canonical_json_bytes(
            {
                "authorization": report["authorization"],
                "decision": report["decision"],
                "evidence": report["evidence"],
                "implementation": report["implementation"],
                "inputs": report["inputs"],
                "run_fingerprint": report["run_fingerprint"],
                "summary": report["summary"],
            }
        )
    )
    _validate_no_payload_or_credentials(report, "source-file report")
    return report


def _check_disk(output: Path, runtime: Mapping[str, Any]) -> None:
    minimum_free = runtime["artifact_policy"]["minimum_free_bytes_before_start"]
    if shutil.disk_usage(output).free < minimum_free:
        raise RuntimeError("free disk space is below the source-file manifest safety floor")


def _artifact_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def _seal_manifest(output: Path) -> None:
    names = [DATABASE_NAME, REPORT_NAME, SNAPSHOT_NAME, SOURCE_FILES_NAME]
    lines = [f"{_sha256_file(output / name)}  {name}" for name in sorted(names)]
    _write_bytes_atomic(output / MANIFEST_NAME, ("\n".join(lines) + "\n").encode("utf-8"))


def run_source_file_manifest(
    contract_path: Path,
    output: Path,
    *,
    repository_root: Path | None = None,
    repository_snapshot: Path | None = None,
    resume: bool = False,
    retries: int = 3,
    retry_delay_seconds: float = 1.0,
    fetch_tree: TreeFetcher | None = None,
    progress: ProgressCallback = print,
) -> dict[str, Any]:
    """Fetch and seal public repository metadata without downloading corpus payloads."""

    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    runtime = _load_runtime(contract_path, root)
    if repository_snapshot is not None:
        runtime["repository_snapshot"] = repository_snapshot.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _check_disk(output, runtime)
    database_path = output / DATABASE_NAME
    if database_path.exists() and not resume:
        raise RuntimeError("source-file manifest state exists; pass --resume to reuse it")
    fetcher = fetch_tree or _default_fetcher(runtime)

    with _exclusive_lock(output / ".run.lock"):
        connection = _open_database(database_path)
        try:
            _initialize_state(connection, runtime)
            _process_sources(
                connection,
                runtime,
                fetcher,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
                progress=progress,
            )
            snapshot = _build_snapshot(connection, runtime)
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        finally:
            connection.close()

        snapshot_path = output / SNAPSHOT_NAME
        source_files_path = output / SOURCE_FILES_NAME
        snapshot_bytes = (
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_bytes_atomic(snapshot_path, snapshot_bytes)
        _write_bytes_atomic(runtime["repository_snapshot"], snapshot_bytes)
        records = _flatten_snapshot(snapshot)
        _write_bytes_atomic(source_files_path, _source_jsonl_bytes(records))
        report = _build_report(
            runtime,
            snapshot,
            database_path=database_path,
            source_files_path=source_files_path,
            snapshot_path=snapshot_path,
            repository_root=root,
        )
        _write_json_atomic(output / REPORT_NAME, report)
        _seal_manifest(output)
        if _artifact_bytes(output) > runtime["artifact_policy"]["maximum_artifact_bytes"]:
            raise RuntimeError("source-file manifest artifacts exceeded the hard size cap")
    return report


def _read_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise ValueError("source-file manifest SHA file is malformed")
        digest, name = match.groups()
        if name in result:
            raise ValueError(f"source-file manifest repeats artifact {name}")
        result[name] = digest
    expected = {DATABASE_NAME, REPORT_NAME, SNAPSHOT_NAME, SOURCE_FILES_NAME}
    if set(result) != expected:
        raise ValueError("source-file manifest SHA file has an unexpected artifact set")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid source-file JSONL line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"source-file JSONL line {line_number} is not an object")
            records.append(record)
    return records


def verify_source_file_manifest(
    output: Path,
    *,
    contract_path: Path,
    repository_root: Path | None = None,
    repository_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Independently verify hashes, SQLite state, flattened records, and pinned inputs."""

    output = output.resolve()
    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    runtime = _load_runtime(contract_path, root)
    if repository_snapshot is not None:
        runtime["repository_snapshot"] = repository_snapshot.resolve()
    manifest = _read_manifest(output / MANIFEST_NAME)
    for name, expected_digest in manifest.items():
        path = output / name
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise ValueError(f"source-file artifact hash does not match: {name}")

    snapshot_path = output / SNAPSHOT_NAME
    snapshot = _read_json(snapshot_path)
    _validate_no_payload_or_credentials(snapshot, "source-file snapshot")
    if _sha256_file(runtime["repository_snapshot"]) != _sha256_file(snapshot_path):
        raise ValueError("repository source-file snapshot differs from sealed evidence")
    expected_records = _flatten_snapshot(snapshot)
    records = _read_jsonl(output / SOURCE_FILES_NAME)
    if records != expected_records:
        raise ValueError("source-files JSONL differs from the canonical snapshot")
    if len(records) != runtime["expected_totals"]["data_files"]:
        raise ValueError("source-files JSONL count differs from the pinned total")
    _validate_no_payload_or_credentials(records, "source-files JSONL")

    database_path = output / DATABASE_NAME
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if _metadata_get(connection, "run_fingerprint") != runtime["run_fingerprint"]:
            raise ValueError("source-file state fingerprint differs from the pinned run")
        database_snapshot = _build_snapshot(connection, runtime)
    finally:
        connection.close()
    if database_snapshot != snapshot:
        raise ValueError("source-file SQLite state differs from the canonical snapshot")

    report_path = output / REPORT_NAME
    report = _read_json(report_path)
    expected_report = _build_report(
        runtime,
        snapshot,
        database_path=database_path,
        source_files_path=output / SOURCE_FILES_NAME,
        snapshot_path=snapshot_path,
        repository_root=root,
    )
    if report != expected_report:
        raise ValueError("source-file report does not reproduce from pinned evidence")
    if report["summary"] != snapshot["summary"]:
        raise ValueError("source-file report and snapshot summaries differ")
    return report
