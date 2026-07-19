"""Resumable, text-free retention canaries for the tightest Lumi corpus domains."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import time
import urllib.error
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .corpus_capacity import build_corpus_capacity_report
from .data_pipeline import (
    PII_DETECTOR_VERSION,
    _extract_provenance,
    _iter_source_records,
    _record_id,
    _source_dataset,
    filter_document,
    split_utf8,
)
from .data_quality import DataQualityGate, QualityEvaluation
from .development_config import compose_development_config

SCHEMA_VERSION = 1
DATABASE_NAME = "retention-state.sqlite3"
REPORT_NAME = "report.json"
MANIFEST_NAME = "manifest.sha256"
MAX_TOTAL_RAW_TEXT_BYTES = 192_000_000
MAX_CATEGORY_RAW_TEXT_BYTES = 64_000_000
MAX_SOURCE_RAW_TEXT_BYTES = 64_000_000
MAX_ARTIFACT_BYTES = 2_147_483_648
IMPLEMENTATION_FILES = (
    "src/lumi/corpus_capacity.py",
    "src/lumi/corpus_retention.py",
    "src/lumi/data_pipeline.py",
    "src/lumi/data_quality.py",
    "src/lumi/development_config.py",
)
NETWORK_ERRORS = (EOFError, OSError, TimeoutError, urllib.error.URLError)

RecordIteratorFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any], float],
    Iterator[tuple[int, Mapping[str, Any] | None]],
]
TokenCounter = Callable[[list[str]], list[int]]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _implementation_hashes(repository_root: Path) -> dict[str, str]:
    hashes = {}
    for relative_path in IMPLEMENTATION_FILES:
        path = repository_root / relative_path
        if not path.is_file():
            raise ValueError(f"missing retention implementation file: {path}")
        hashes[relative_path] = _sha256_file(path)
    return hashes


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _resolve(repository_root: Path, value: str, label: str) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} escapes the repository: {value}")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return path


def _artifact_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def _check_disk(output: Path, disk_policy: Mapping[str, Any]) -> None:
    minimum_free = _positive_int(
        disk_policy["minimum_free_bytes_before_start"], "minimum free bytes"
    )
    if shutil.disk_usage(output).free < minimum_free:
        raise RuntimeError("free disk space is below the retention-canary safety floor")
    maximum_artifacts = _positive_int(
        disk_policy["maximum_artifact_bytes"], "maximum artifact bytes"
    )
    if maximum_artifacts > MAX_ARTIFACT_BYTES:
        raise ValueError("retention-canary artifact cap exceeds the compiled safety ceiling")
    if _artifact_bytes(output) > maximum_artifacts:
        raise RuntimeError("retention-canary artifacts exceeded the hard disk cap")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another retention-canary process holds the run lock") from error
        yield


def _metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM retention_metadata WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else str(row[0])


def _metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO retention_metadata (key, value) VALUES (?, ?)",
        (key, str(value)),
    )


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retention_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retention_sources ("
        "component TEXT PRIMARY KEY, category TEXT NOT NULL, dataset_id TEXT NOT NULL, "
        "revision TEXT NOT NULL, source_file TEXT NOT NULL, raw_limit INTEGER NOT NULL, "
        "next_line INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending', "
        "raw_documents INTEGER NOT NULL DEFAULT 0, raw_text_bytes INTEGER NOT NULL DEFAULT 0, "
        "accepted_documents INTEGER NOT NULL DEFAULT 0, "
        "accepted_text_bytes INTEGER NOT NULL DEFAULT 0, "
        "accepted_tokens INTEGER NOT NULL DEFAULT 0, rejections TEXT NOT NULL DEFAULT '{}', "
        "pii_redactions TEXT NOT NULL DEFAULT '{}', "
        "language_labels TEXT NOT NULL DEFAULT '{}')"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS retention_exact_content ("
        "content_sha256 TEXT PRIMARY KEY, document_id TEXT NOT NULL UNIQUE, "
        "component TEXT NOT NULL, source_line INTEGER NOT NULL, segment_index INTEGER NOT NULL)"
    )
    connection.commit()
    return connection


def _load_runtime(
    contract_path: Path, repository_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
    str,
]:
    specification = _read_json(contract_path)
    if specification.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus retention canary schema")
    authorization = specification["authorization"]
    if authorization.get("stream_bounded_canaries") is not True:
        raise ValueError("bounded canary authorization must be explicit")
    for forbidden in ("bulk_corpus_materialization", "full_4b_training", "paid_compute"):
        if authorization.get(forbidden) is not False:
            raise ValueError("retention canary must not authorize bulk data, training, or compute")
    disk_policy = specification["disk_policy"]
    if disk_policy.get("persist_raw_text") is not False:
        raise ValueError("retention canary must not persist raw text")
    if disk_policy.get("persist_accepted_text") is not False:
        raise ValueError("retention canary must not persist accepted text")

    paths = {
        name: _resolve(repository_root, value, name)
        for name, value in specification["inputs"].items()
    }
    source_config = compose_development_config(
        paths["source_plan"], repository_root=repository_root
    )
    available_sources = {
        source["component"]: {"category": entry["category"], **source}
        for entry in source_config["mixture"]
        for source in entry["sources"]
    }
    canaries = specification["canaries"]
    if not isinstance(canaries, list) or not canaries:
        raise ValueError("retention canary requires source entries")
    selected = []
    seen_components: set[str] = set()
    observed_category_limits: Counter[str] = Counter()
    for canary in canaries:
        component = canary["component"]
        if component in seen_components:
            raise ValueError(f"retention canary repeats component {component}")
        seen_components.add(component)
        source = available_sources.get(component)
        if source is None:
            raise ValueError(f"retention canary uses unknown component {component}")
        if source["category"] != canary["category"]:
            raise ValueError(f"retention canary category mismatch for {component}")
        raw_limit = _positive_int(
            canary["raw_text_utf8_bytes_limit"], f"{component} raw byte limit"
        )
        if raw_limit > MAX_SOURCE_RAW_TEXT_BYTES:
            raise ValueError(f"{component} exceeds the per-source canary safety ceiling")
        observed_category_limits[source["category"]] += raw_limit
        selected.append({**source, "raw_text_utf8_bytes_limit": raw_limit})
    expected_category_limits = {
        category: _positive_int(value, f"{category} category raw limit")
        for category, value in specification["category_raw_text_utf8_bytes"].items()
    }
    if any(value > MAX_CATEGORY_RAW_TEXT_BYTES for value in expected_category_limits.values()):
        raise ValueError("a category exceeds the retention-canary safety ceiling")
    if sum(expected_category_limits.values()) > MAX_TOTAL_RAW_TEXT_BYTES:
        raise ValueError("retention canary exceeds the compiled total raw-byte ceiling")
    if dict(observed_category_limits) != expected_category_limits:
        raise ValueError("source canary limits do not match category limits")

    capacity_report = build_corpus_capacity_report(
        paths["capacity_contract"], repository_root=repository_root
    )
    if capacity_report["decision"]["bounded_source_canaries"] != "ready":
        raise ValueError("capacity planning gate does not permit bounded source canaries")
    snapshot = _read_json(paths["published_capacity_snapshot"])
    snapshot_sources = {row["component"]: row for row in snapshot["sources"]}
    if not seen_components <= set(snapshot_sources):
        raise ValueError("published capacity snapshot does not cover every canary")
    for source in selected:
        snapshot_source = snapshot_sources[source["component"]]
        if source["dataset_id"] != snapshot_source["dataset_id"]:
            raise ValueError(f"capacity dataset drifted for {source['component']}")
        if source["revision"] != snapshot_source["revision"]:
            raise ValueError(f"capacity revision drifted for {source['component']}")

    input_hashes = {
        "contract": _sha256_file(contract_path),
        **{name: _sha256_file(path) for name, path in sorted(paths.items())},
    }
    implementation_hashes = _implementation_hashes(repository_root)
    fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "implementation": implementation_hashes,
                "inputs": input_hashes,
                "pii_detector": PII_DETECTOR_VERSION,
                "selected_sources": selected,
            }
        )
    )
    return (
        specification,
        paths,
        source_config,
        selected,
        capacity_report,
        snapshot_sources,
        fingerprint,
    )


def _default_token_counter(tokenizer_path: Path, expected: Mapping[str, Any]) -> TokenCounter:
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ValueError(f"missing frozen tokenizer: {tokenizer_json}")
    if _sha256_file(tokenizer_json) != expected["tokenizer_json_sha256"]:
        raise ValueError("frozen tokenizer hash does not match its contract")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    if tokenizer.get_vocab_size() != expected["vocab_size"]:
        raise ValueError("frozen tokenizer vocabulary size changed")

    def count(texts: list[str]) -> list[int]:
        return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]

    return count


def _initialize_run(
    connection: sqlite3.Connection,
    *,
    fingerprint: str,
    selected_sources: list[dict[str, Any]],
) -> None:
    existing_fingerprint = _metadata_get(connection, "fingerprint")
    if existing_fingerprint is not None and existing_fingerprint != fingerprint:
        raise ValueError("existing retention state has a different run fingerprint")
    with connection:
        _metadata_set(connection, "fingerprint", fingerprint)
        _metadata_set(connection, "schema_version", SCHEMA_VERSION)
        connection.execute("DELETE FROM retention_metadata WHERE key = 'failure'")
        if _metadata_get(connection, "started_at") is None:
            _metadata_set(connection, "started_at", _utc_now())
            _metadata_set(connection, "network_retries", 0)
        _metadata_set(connection, "status", "running")
        _metadata_set(connection, "updated_at", _utc_now())
        for source in selected_sources:
            connection.execute(
                "INSERT OR IGNORE INTO retention_sources "
                "(component, category, dataset_id, revision, source_file, raw_limit) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source["component"],
                    source["category"],
                    source["dataset_id"],
                    source["revision"],
                    source["file"],
                    source["raw_text_utf8_bytes_limit"],
                ),
            )
            row = connection.execute(
                "SELECT * FROM retention_sources WHERE component = ?",
                (source["component"],),
            ).fetchone()
            assert row is not None
            expected = (
                source["category"],
                source["dataset_id"],
                source["revision"],
                source["file"],
                source["raw_text_utf8_bytes_limit"],
            )
            observed = (
                row["category"],
                row["dataset_id"],
                row["revision"],
                row["source_file"],
                row["raw_limit"],
            )
            if observed != expected:
                component = source["component"]
                raise ValueError(f"existing retention source contract changed: {component}")


def _counter_from_json(value: str) -> Counter[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored retention counter is invalid")
    return Counter({str(key): int(count) for key, count in parsed.items()})


def _process_record(
    connection: sqlite3.Connection,
    quality_gate: DataQualityGate,
    token_counter: TokenCounter,
    *,
    dataset: Mapping[str, Any],
    source: Mapping[str, Any],
    source_line: int,
    upstream_record: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> bool:
    row = connection.execute(
        "SELECT * FROM retention_sources WHERE component = ?", (source["component"],)
    ).fetchone()
    assert row is not None
    if source_line != int(row["next_line"]) + 1:
        raise ValueError(f"non-contiguous source line for {source['component']}: {source_line}")

    raw_text_bytes = 0
    rejections: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    language_labels: Counter[str] = Counter()
    accepted: list[tuple[str, str, int, bytes, QualityEvaluation]] = []
    if upstream_record is None:
        rejections["invalid_json"] += 1
    else:
        raw_text = upstream_record.get("text")
        if not isinstance(raw_text, str):
            rejections["missing_text"] += 1
        else:
            raw_text_bytes = len(raw_text.encode("utf-8"))
            _, missing_provenance = _extract_provenance(upstream_record, source)
            if missing_provenance:
                for name in missing_provenance:
                    rejections[f"missing_provenance:{name}"] += 1
            else:
                filtered = filter_document(raw_text, policy)
                redactions.update(filtered.redactions)
                if filtered.rejection_reason is not None:
                    rejections[filtered.rejection_reason] += 1
                else:
                    assert filtered.text is not None
                    local_hashes: set[str] = set()
                    source_dataset = _source_dataset(dataset, source)
                    for segment_index, segment in enumerate(
                        split_utf8(
                            filtered.text,
                            int(policy["max_segment_utf8_bytes"]),
                            int(policy["min_document_characters"]),
                        )
                    ):
                        segment_bytes = segment.encode("utf-8")
                        content_hash = _sha256_bytes(segment_bytes)
                        duplicate = (
                            content_hash in local_hashes
                            or connection.execute(
                                "SELECT 1 FROM retention_exact_content WHERE content_sha256 = ?",
                                (content_hash,),
                            ).fetchone()
                        )
                        if duplicate:
                            rejections["exact_duplicate"] += 1
                            continue
                        document_id = _record_id(
                            source_dataset, source["file"], source_line, segment_index
                        )
                        evaluation = quality_gate.evaluate(segment, source["category"])
                        if evaluation.rejection_reason is not None:
                            rejections[evaluation.rejection_reason] += 1
                            continue
                        assert evaluation.metadata is not None
                        language_labels[evaluation.metadata["language"]["label"]] += 1
                        accepted.append(
                            (document_id, content_hash, segment_index, segment_bytes, evaluation)
                        )
                        local_hashes.add(content_hash)

    token_counts = token_counter([item[3].decode("utf-8") for item in accepted])
    if len(token_counts) != len(accepted) or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in token_counts
    ):
        raise ValueError("token counter returned invalid results")

    stored_rejections = _counter_from_json(row["rejections"])
    stored_redactions = _counter_from_json(row["pii_redactions"])
    stored_languages = _counter_from_json(row["language_labels"])
    stored_rejections.update(rejections)
    stored_redactions.update(redactions)
    stored_languages.update(language_labels)
    accepted_text_bytes = sum(len(item[3]) for item in accepted)
    accepted_tokens = sum(token_counts)
    next_raw_bytes = int(row["raw_text_bytes"]) + raw_text_bytes
    completed = next_raw_bytes >= int(row["raw_limit"])
    with connection:
        for document_id, content_hash, segment_index, _, evaluation in accepted:
            quality_gate.accept(document_id, evaluation)
            connection.execute(
                "INSERT INTO retention_exact_content "
                "(content_sha256, document_id, component, source_line, segment_index) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    content_hash,
                    document_id,
                    source["component"],
                    source_line,
                    segment_index,
                ),
            )
        connection.execute(
            "UPDATE retention_sources SET next_line = ?, status = ?, "
            "raw_documents = raw_documents + 1, raw_text_bytes = raw_text_bytes + ?, "
            "accepted_documents = accepted_documents + ?, "
            "accepted_text_bytes = accepted_text_bytes + ?, "
            "accepted_tokens = accepted_tokens + ?, rejections = ?, pii_redactions = ?, "
            "language_labels = ? WHERE component = ?",
            (
                source_line,
                "complete" if completed else "running",
                raw_text_bytes,
                len(accepted),
                accepted_text_bytes,
                accepted_tokens,
                json.dumps(dict(sorted(stored_rejections.items())), sort_keys=True),
                json.dumps(dict(sorted(stored_redactions.items())), sort_keys=True),
                json.dumps(dict(sorted(stored_languages.items())), sort_keys=True),
                source["component"],
            ),
        )
        _metadata_set(connection, "updated_at", _utc_now())
    return completed


def _increment_network_retry(connection: sqlite3.Connection) -> None:
    with connection:
        retries = int(_metadata_get(connection, "network_retries") or "0") + 1
        _metadata_set(connection, "network_retries", retries)
        _metadata_set(connection, "updated_at", _utc_now())


def _run_source(
    connection: sqlite3.Connection,
    quality_gate: DataQualityGate,
    token_counter: TokenCounter,
    *,
    dataset: Mapping[str, Any],
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    record_iterator_factory: RecordIteratorFactory,
    timeout_seconds: float,
    retries: int,
    retry_forever: bool,
    retry_sleep: Callable[[float], None],
    disk_check: Callable[[], None],
) -> None:
    row = connection.execute(
        "SELECT status FROM retention_sources WHERE component = ?", (source["component"],)
    ).fetchone()
    assert row is not None
    if row["status"] == "complete":
        return
    attempt = 0
    while True:
        try:
            for source_line, upstream_record in record_iterator_factory(
                dataset, source, timeout_seconds
            ):
                cursor = connection.execute(
                    "SELECT next_line FROM retention_sources WHERE component = ?",
                    (source["component"],),
                ).fetchone()
                assert cursor is not None
                if source_line <= int(cursor["next_line"]):
                    continue
                completed = _process_record(
                    connection,
                    quality_gate,
                    token_counter,
                    dataset=dataset,
                    source=source,
                    source_line=source_line,
                    upstream_record=upstream_record,
                    policy=policy,
                )
                if source_line % 100 == 0:
                    disk_check()
                if completed:
                    return
            with connection:
                connection.execute(
                    "UPDATE retention_sources SET status = 'exhausted' WHERE component = ?",
                    (source["component"],),
                )
            raise RuntimeError(
                f"pinned source {source['component']} exhausted before its raw byte limit"
            )
        except NETWORK_ERRORS as error:
            if not retry_forever and attempt >= retries:
                raise RuntimeError(
                    f"failed streaming retention source {source['component']}"
                ) from error
            _increment_network_retry(connection)
            retry_sleep(min(2 ** min(attempt, 6), 60))
            attempt += 1


def _build_report(
    connection: sqlite3.Connection,
    *,
    specification: Mapping[str, Any],
    paths: Mapping[str, Path],
    contract_path: Path,
    selected_sources: list[dict[str, Any]],
    capacity_report: Mapping[str, Any],
    snapshot_sources: Mapping[str, Mapping[str, Any]],
    fingerprint: str,
    tokenizer_verified: bool,
    repository_root: Path,
) -> dict[str, Any]:
    capacity_categories = {row["category"]: row for row in capacity_report["categories"]}
    source_rows = []
    category_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for source in selected_sources:
        row = connection.execute(
            "SELECT * FROM retention_sources WHERE component = ?", (source["component"],)
        ).fetchone()
        assert row is not None
        raw_bytes = int(row["raw_text_bytes"])
        accepted_tokens = int(row["accepted_tokens"])
        published_bytes = int(snapshot_sources[source["component"]]["published_utf8_bytes"])
        projected_tokens = published_bytes * accepted_tokens // raw_bytes
        source_report = {
            "accepted_documents": int(row["accepted_documents"]),
            "accepted_text_retention_fraction": int(row["accepted_text_bytes"]) / raw_bytes,
            "accepted_text_utf8_bytes": int(row["accepted_text_bytes"]),
            "accepted_token_yield_per_raw_utf8_byte": accepted_tokens / raw_bytes,
            "accepted_tokens": accepted_tokens,
            "category": source["category"],
            "component": source["component"],
            "dataset_id": source["dataset_id"],
            "language_labels": json.loads(row["language_labels"]),
            "next_line": int(row["next_line"]),
            "pii_redactions": json.loads(row["pii_redactions"]),
            "projected_post_filter_tokens": projected_tokens,
            "published_utf8_bytes": published_bytes,
            "raw_documents": int(row["raw_documents"]),
            "raw_text_utf8_bytes": raw_bytes,
            "raw_text_utf8_bytes_limit": int(row["raw_limit"]),
            "rejections": json.loads(row["rejections"]),
            "revision": source["revision"],
            "source_file": source["file"],
            "status": row["status"],
        }
        source_rows.append(source_report)
        totals = category_totals[source["category"]]
        for name in (
            "accepted_documents",
            "accepted_text_utf8_bytes",
            "accepted_tokens",
            "projected_post_filter_tokens",
            "published_utf8_bytes",
            "raw_documents",
            "raw_text_utf8_bytes",
            "raw_text_utf8_bytes_limit",
        ):
            totals[name] += int(source_report[name])

    category_rows = []
    for category in sorted(category_totals):
        totals = category_totals[category]
        minimum = int(capacity_categories[category]["minimum_candidate_tokens"])
        projected = int(totals["projected_post_filter_tokens"])
        category_rows.append(
            {
                "accepted_documents": int(totals["accepted_documents"]),
                "accepted_text_retention_fraction": (
                    totals["accepted_text_utf8_bytes"] / totals["raw_text_utf8_bytes"]
                ),
                "accepted_text_utf8_bytes": int(totals["accepted_text_utf8_bytes"]),
                "accepted_tokens": int(totals["accepted_tokens"]),
                "category": category,
                "minimum_candidate_tokens": minimum,
                "projected_post_filter_tokens": projected,
                "projection_headroom_fraction": projected / minimum - 1.0,
                "projection_passes": projected >= minimum,
                "published_utf8_bytes": int(totals["published_utf8_bytes"]),
                "raw_documents": int(totals["raw_documents"]),
                "raw_text_utf8_bytes": int(totals["raw_text_utf8_bytes"]),
                "raw_text_utf8_bytes_limit": int(totals["raw_text_utf8_bytes_limit"]),
            }
        )
    projections_pass = all(row["projection_passes"] for row in category_rows)
    sources_complete = all(row["status"] == "complete" for row in source_rows)
    tokenizer_contract = _read_json(paths["tokenizer_contract"])
    input_hashes = {
        "contract": {
            "path": _portable_path(contract_path, repository_root),
            "sha256": _sha256_file(contract_path),
        },
        **{
            name: {
                "path": _portable_path(path, repository_root),
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(paths.items())
        },
    }
    return {
        "authorization": dict(specification["authorization"]),
        "categories": category_rows,
        "decision": {
            "bulk_corpus_materialization": "not_authorized",
            "capacity_projection": "pass" if projections_pass else "fail",
            "full_4b_training": "not_authorized",
            "next_action": (
                "object_storage_and_acquisition_design"
                if projections_pass
                else "revise_tight_category_sources"
            ),
            "paid_compute": "not_authorized",
            "retention_canary": ("pass" if sources_complete and tokenizer_verified else "fail"),
        },
        "disk_policy": dict(specification["disk_policy"]),
        "gate": specification["gate"],
        "inputs": input_hashes,
        "implementation": {
            path: {"path": path, "sha256": digest}
            for path, digest in sorted(_implementation_hashes(repository_root).items())
        },
        "limitations": [
            "Deterministic first-shard canaries are not random samples of each full source.",
            "Published capacity projections assume the measured first-shard yield generalizes.",
            "Each configured raw-byte target may be exceeded by the final whole document.",
            "Cross-source deduplication makes measured source yield dependent on source order.",
            "PII, secret, language, benchmark, and near-duplicate detectors are conservative.",
            "A passing projection does not approve bulk materialization or model training.",
        ],
        "network_retries": int(_metadata_get(connection, "network_retries") or "0"),
        "run_fingerprint": fingerprint,
        "schema_version": SCHEMA_VERSION,
        "sources": source_rows,
        "status": "complete" if sources_complete else "incomplete",
        "summary": {
            "accepted_documents": sum(row["accepted_documents"] for row in source_rows),
            "accepted_text_utf8_bytes": sum(row["accepted_text_utf8_bytes"] for row in source_rows),
            "accepted_tokens": sum(row["accepted_tokens"] for row in source_rows),
            "raw_documents": sum(row["raw_documents"] for row in source_rows),
            "raw_text_utf8_bytes": sum(row["raw_text_utf8_bytes"] for row in source_rows),
            "raw_text_utf8_bytes_limit": sum(
                row["raw_text_utf8_bytes_limit"] for row in source_rows
            ),
            "tokenizer_json_sha256": tokenizer_contract["baseline"]["tokenizer_json_sha256"],
            "tokenizer_verified": tokenizer_verified,
        },
        "version": specification["version"],
    }


def _seal(output: Path) -> None:
    entries = []
    for name in (DATABASE_NAME, REPORT_NAME):
        path = output / name
        if not path.is_file():
            raise ValueError(f"cannot seal missing retention artifact: {path}")
        entries.append(f"{_sha256_file(path)}  {name}")
    (output / MANIFEST_NAME).write_text("\n".join(entries) + "\n", encoding="utf-8")


def verify_retention_canary(output: Path, *, repository_root: Path | None = None) -> dict[str, Any]:
    manifest_path = output / MANIFEST_NAME
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(manifest_lines) != 2:
        raise ValueError("retention manifest must contain exactly two entries")
    entries = {}
    for line in manifest_lines:
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("retention manifest contains an invalid SHA-256 digest")
        entries[name] = digest
    if set(entries) != {DATABASE_NAME, REPORT_NAME}:
        raise ValueError("retention manifest has an unexpected artifact set")
    for name, expected in entries.items():
        if _sha256_file(output / name) != expected:
            raise ValueError(f"retention artifact hash mismatch: {name}")
    report = _read_json(output / REPORT_NAME)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "complete":
        raise ValueError("retention report is not a complete supported report")
    authorization = report.get("authorization", {})
    if authorization.get("stream_bounded_canaries") is not True or any(
        authorization.get(name) is not False
        for name in ("bulk_corpus_materialization", "full_4b_training", "paid_compute")
    ):
        raise ValueError("retention report authorization is unsafe")
    disk_policy = report.get("disk_policy", {})
    if disk_policy.get("persist_raw_text") is not False:
        raise ValueError("retention report permits raw-text persistence")
    if disk_policy.get("persist_accepted_text") is not False:
        raise ValueError("retention report permits accepted-text persistence")

    runtime = None
    if repository_root is not None:
        root = repository_root.resolve()
        for entry in [*report["inputs"].values(), *report["implementation"].values()]:
            path = Path(entry["path"])
            path = path if path.is_absolute() else root / path
            if not path.is_file() or _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"retention input hash mismatch: {entry['path']}")
        contract_entry = report["inputs"]["contract"]
        contract_path = Path(contract_entry["path"])
        contract_path = contract_path if contract_path.is_absolute() else root / contract_path
        runtime = _load_runtime(contract_path, root)
        if runtime[-1] != report["run_fingerprint"]:
            raise ValueError("retention runtime fingerprint cannot be reproduced")

    database_uri = f"file:{(output / DATABASE_NAME).resolve()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("retention database integrity check failed")
        if _metadata_get(connection, "status") != "complete":
            raise ValueError("retention database is not complete")
        if _metadata_get(connection, "fingerprint") != report["run_fingerprint"]:
            raise ValueError("retention database fingerprint does not match the report")
        database_sources = {
            row["component"]: row
            for row in connection.execute("SELECT * FROM retention_sources").fetchall()
        }
        source_count = len(database_sources)
        accepted_count = connection.execute(
            "SELECT COALESCE(SUM(accepted_documents), 0) FROM retention_sources"
        ).fetchone()[0]
        exact_count = connection.execute("SELECT COUNT(*) FROM retention_exact_content").fetchone()[
            0
        ]
        signature_count = connection.execute(
            "SELECT COUNT(*) FROM retention_near_duplicate_signatures"
        ).fetchone()[0]
    if source_count != len(report["sources"]):
        raise ValueError("retention source count does not match the report")
    if accepted_count != exact_count or accepted_count != signature_count:
        raise ValueError("retention accepted-document indexes disagree")

    report_sources = {row["component"]: row for row in report["sources"]}
    if len(report_sources) != len(report["sources"]) or set(report_sources) != set(
        database_sources
    ):
        raise ValueError("retention report source identities disagree")
    for component, source in report_sources.items():
        stored = database_sources[component]
        direct_fields = {
            "accepted_documents": "accepted_documents",
            "accepted_text_utf8_bytes": "accepted_text_bytes",
            "accepted_tokens": "accepted_tokens",
            "category": "category",
            "dataset_id": "dataset_id",
            "next_line": "next_line",
            "raw_documents": "raw_documents",
            "raw_text_utf8_bytes": "raw_text_bytes",
            "raw_text_utf8_bytes_limit": "raw_limit",
            "revision": "revision",
            "source_file": "source_file",
            "status": "status",
        }
        for report_name, stored_name in direct_fields.items():
            if source[report_name] != stored[stored_name]:
                raise ValueError(f"retention source metric mismatch: {component}/{report_name}")
        for report_name, stored_name in (
            ("language_labels", "language_labels"),
            ("pii_redactions", "pii_redactions"),
            ("rejections", "rejections"),
        ):
            if source[report_name] != json.loads(stored[stored_name]):
                raise ValueError(f"retention source counter mismatch: {component}/{report_name}")
        raw_bytes = int(source["raw_text_utf8_bytes"])
        if raw_bytes <= 0 or source["status"] != "complete":
            raise ValueError(f"retention source did not complete: {component}")
        if source["accepted_text_retention_fraction"] != (
            source["accepted_text_utf8_bytes"] / raw_bytes
        ):
            raise ValueError(f"retention text fraction mismatch: {component}")
        if source["accepted_token_yield_per_raw_utf8_byte"] != (
            source["accepted_tokens"] / raw_bytes
        ):
            raise ValueError(f"retention token yield mismatch: {component}")
        if source["projected_post_filter_tokens"] != (
            source["published_utf8_bytes"] * source["accepted_tokens"] // raw_bytes
        ):
            raise ValueError(f"retention source projection mismatch: {component}")

    if runtime is not None:
        selected = {source["component"]: source for source in runtime[3]}
        snapshot_sources = runtime[5]
        if set(selected) != set(report_sources):
            raise ValueError("retention selected-source contract disagrees with the report")
        for component, source in report_sources.items():
            expected = selected[component]
            for name in ("category", "dataset_id", "revision"):
                if source[name] != expected[name]:
                    raise ValueError(f"retention source contract mismatch: {component}/{name}")
            if source["source_file"] != expected["file"]:
                raise ValueError(f"retention source contract mismatch: {component}/file")
            if source["published_utf8_bytes"] != int(
                snapshot_sources[component]["published_utf8_bytes"]
            ):
                raise ValueError(f"retention source capacity mismatch: {component}")

    summary = report["summary"]
    summary_fields = {
        "accepted_documents": "accepted_documents",
        "accepted_text_utf8_bytes": "accepted_text_utf8_bytes",
        "accepted_tokens": "accepted_tokens",
        "raw_documents": "raw_documents",
        "raw_text_utf8_bytes": "raw_text_utf8_bytes",
        "raw_text_utf8_bytes_limit": "raw_text_utf8_bytes_limit",
    }
    for summary_name, source_name in summary_fields.items():
        if summary[summary_name] != sum(
            int(source[source_name]) for source in report_sources.values()
        ):
            raise ValueError(f"retention summary mismatch: {summary_name}")
    if summary["raw_text_utf8_bytes_limit"] > MAX_TOTAL_RAW_TEXT_BYTES:
        raise ValueError("retention report exceeds the total raw-byte safety ceiling")

    categories = {row["category"]: row for row in report["categories"]}
    if len(categories) != len(report["categories"]):
        raise ValueError("retention report repeats a category")
    for category, category_report in categories.items():
        members = [source for source in report_sources.values() if source["category"] == category]
        if not members:
            raise ValueError(f"retention category has no sources: {category}")
        for name in (
            "accepted_documents",
            "accepted_text_utf8_bytes",
            "accepted_tokens",
            "projected_post_filter_tokens",
            "published_utf8_bytes",
            "raw_documents",
            "raw_text_utf8_bytes",
            "raw_text_utf8_bytes_limit",
        ):
            if category_report[name] != sum(int(source[name]) for source in members):
                raise ValueError(f"retention category mismatch: {category}/{name}")
        if category_report["accepted_text_retention_fraction"] != (
            category_report["accepted_text_utf8_bytes"] / category_report["raw_text_utf8_bytes"]
        ):
            raise ValueError(f"retention category text fraction mismatch: {category}")
        passes = (
            category_report["projected_post_filter_tokens"]
            >= category_report["minimum_candidate_tokens"]
        )
        if category_report["projection_passes"] is not passes:
            raise ValueError(f"retention category decision mismatch: {category}")

    projections_pass = all(row["projection_passes"] for row in categories.values())
    decision = report["decision"]
    expected_retention = "pass" if summary["tokenizer_verified"] else "fail"
    if decision["retention_canary"] != expected_retention:
        raise ValueError("retention completion decision mismatch")
    if decision["capacity_projection"] != ("pass" if projections_pass else "fail"):
        raise ValueError("retention capacity decision mismatch")
    if decision["next_action"] != (
        "object_storage_and_acquisition_design"
        if projections_pass
        else "revise_tight_category_sources"
    ):
        raise ValueError("retention next-action decision mismatch")
    for name in ("bulk_corpus_materialization", "full_4b_training", "paid_compute"):
        if decision.get(name) != "not_authorized":
            raise ValueError(f"retention decision unexpectedly authorizes {name}")
    return report


def retention_progress(output: Path) -> dict[str, Any]:
    database = output / DATABASE_NAME
    if not database.is_file():
        return {"status": "not_started"}
    database_uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT component, category, status, next_line, raw_limit, raw_text_bytes, "
            "accepted_documents, accepted_text_bytes, accepted_tokens "
            "FROM retention_sources ORDER BY rowid"
        ).fetchall()
        return {
            "artifact_bytes": _artifact_bytes(output),
            "network_retries": int(_metadata_get(connection, "network_retries") or "0"),
            "sources": [dict(row) for row in rows],
            "status": _metadata_get(connection, "status"),
            "updated_at": _metadata_get(connection, "updated_at"),
        }


def run_retention_canary(
    contract_path: Path,
    tokenizer_path: Path | None,
    output: Path,
    *,
    repository_root: Path | None = None,
    resume: bool = False,
    timeout_seconds: float = 60.0,
    retries: int = 2,
    retry_forever: bool = False,
    record_iterator_factory: RecordIteratorFactory = _iter_source_records,
    token_counter_override: TokenCounter | None = None,
    allow_unverified_test_tokenizer: bool = False,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run or resume the bounded canaries and seal text-free evidence."""

    if retries < 0:
        raise ValueError("retries cannot be negative")
    contract_path = contract_path.resolve()
    root = repository_root.resolve() if repository_root else contract_path.parents[1]
    runtime = _load_runtime(contract_path, root)
    specification, paths, source_config, selected, capacity_report, snapshot, fingerprint = runtime
    output.mkdir(parents=True, exist_ok=True)
    _check_disk(output, specification["disk_policy"])
    database_path = output / DATABASE_NAME
    if database_path.exists() and not resume:
        raise FileExistsError("retention state exists; pass resume=True")
    tokenizer_contract = _read_json(paths["tokenizer_contract"])["baseline"]
    tokenizer_verified = token_counter_override is None
    if token_counter_override is None:
        if tokenizer_path is None:
            raise ValueError("the frozen tokenizer path is required")
        token_counter = _default_token_counter(tokenizer_path, tokenizer_contract)
    else:
        if not allow_unverified_test_tokenizer:
            raise ValueError("token counter overrides are test-only")
        token_counter = token_counter_override

    with _exclusive_lock(output / ".run.lock"):
        connection = _open_database(database_path)
        quality_gate = DataQualityGate(
            source_config["quality"],
            root,
            near_duplicate_connection=connection,
            near_duplicate_table_prefix="retention_near_duplicate",
        )
        try:
            _initialize_run(connection, fingerprint=fingerprint, selected_sources=selected)
            for source in selected:
                _run_source(
                    connection,
                    quality_gate,
                    token_counter,
                    dataset=source_config["dataset"],
                    source=source,
                    policy=source_config["filters"],
                    record_iterator_factory=record_iterator_factory,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_forever=retry_forever,
                    retry_sleep=retry_sleep,
                    disk_check=lambda: _check_disk(output, specification["disk_policy"]),
                )
            with connection:
                _metadata_set(connection, "status", "complete")
                _metadata_set(connection, "updated_at", _utc_now())
            report = _build_report(
                connection,
                specification=specification,
                paths=paths,
                contract_path=contract_path,
                selected_sources=selected,
                capacity_report=capacity_report,
                snapshot_sources=snapshot,
                fingerprint=fingerprint,
                tokenizer_verified=tokenizer_verified,
                repository_root=root,
            )
            _write_json_atomic(output / REPORT_NAME, report)
            quality_gate.flush()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.commit()
        except Exception as error:
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
            quality_gate.close()
            connection.close()
        _check_disk(output, specification["disk_policy"])
        _seal(output)
        return verify_retention_canary(output, repository_root=root)
