"""Leak-resistant, resumable held-out data construction for Lumi evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .packed_data import (
    MANIFEST_FILENAME,
    PACKED_DATA_SCHEMA_VERSION,
    TOKEN_BYTES,
    TOKEN_DTYPE,
    TOKEN_ENDIANNESS,
    validate_packed_manifest,
)

HELDOUT_SELECTION_MODE = "post_training_hash_ranked"
HELDOUT_STATE_FILENAME = "heldout-state.sqlite3"
SELECTION_LEDGER_FILENAME = "selection-ledger.jsonl"
TRAINING_LEDGER_FILENAME = "training-exclusion-ledger.jsonl"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "bytes": path.stat().st_size,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def _tokenizer_artifact(path: Path) -> dict[str, Any]:
    files = []
    for name, required in (
        ("tokenizer.json", True),
        ("tokenizer_config.json", True),
        ("lumi_tokenizer_metadata.json", False),
    ):
        candidate = path / name
        if required and not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.is_file():
            artifact = _file_artifact(candidate)
            files.append({"bytes": artifact["bytes"], "name": name, "sha256": artifact["sha256"]})
    return {"files": files, "path": str(path.resolve())}


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _stratum_id(category: str, component: str) -> str:
    return json.dumps([category, component], ensure_ascii=False, separators=(",", ":"))


def _allocate_evaluation_strata(
    training_manifest: dict[str, Any], *, target_tokens: int, sequence_length: int
) -> tuple[list[dict[str, Any]], int]:
    total_sequences = target_tokens // sequence_length
    if total_sequences < 1:
        raise ValueError("held-out target must contain at least one complete sequence")
    training_strata = training_manifest.get("selection", {}).get("strata", [])
    if not training_strata:
        raise ValueError("training manifest has no source strata")
    total_weight = sum(item["target_text_utf8_bytes"] for item in training_strata)
    allocated = 0
    strata = []
    for item in training_strata:
        numerator = total_sequences * item["target_text_utf8_bytes"]
        sequences = numerator // total_weight
        strata.append(
            {
                "allocation_remainder": numerator % total_weight,
                "category": item["category"],
                "component": item["component"],
                "id": item["id"],
                "target_sequences": sequences,
                "target_text_utf8_bytes": item["target_text_utf8_bytes"],
                "training_target_tokens": item["target_tokens"],
            }
        )
        allocated += sequences
    leftovers = total_sequences - allocated
    order = sorted(
        strata,
        key=lambda item: (
            -item["allocation_remainder"],
            item["category"],
            item["component"],
        ),
    )
    for item in order[:leftovers]:
        item["target_sequences"] += 1
    for item in strata:
        item.pop("allocation_remainder")
        if item["target_sequences"] < 1:
            raise ValueError(
                "held-out target is too small to allocate every source stratum at least once"
            )
        item["target_tokens"] = item["target_sequences"] * sequence_length
    return strata, total_sequences * sequence_length


def _record_identity(record: dict[str, Any]) -> tuple[str, str, str | None]:
    document_id = record.get("id")
    content_sha256 = record.get("content_sha256")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("source record has no stable document ID")
    if (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValueError(f"source record {document_id!r} has an invalid content SHA-256")
    near_duplicate = record.get("quality", {}).get("near_duplicate", {}).get(
        "fingerprint_sha256"
    )
    if near_duplicate is not None and not isinstance(near_duplicate, str):
        raise ValueError(f"source record {document_id!r} has an invalid near-duplicate hash")
    return document_id, content_sha256, near_duplicate


def _record_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source record is missing provenance")
    category = provenance.get("category")
    component = provenance.get("component")
    if not isinstance(category, str) or not isinstance(component, str):
        raise ValueError("source record has an invalid provenance stratum")
    return category, component, _stratum_id(category, component)


def _selection_rank(seed: int, document_id: str, content_sha256: str) -> str:
    material = f"{seed}\0{document_id}\0{content_sha256}".encode()
    return hashlib.sha256(material).hexdigest()


def _rank_is_candidate(rank: str, denominator: int) -> bool:
    return int(rank[:16], 16) % denominator == 0


def _connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_documents (
            document_id TEXT PRIMARY KEY,
            content_sha256 TEXT NOT NULL UNIQUE,
            near_duplicate_sha256 TEXT,
            stratum_id TEXT NOT NULL,
            input_offset INTEGER NOT NULL,
            tokens_used INTEGER NOT NULL,
            truncated INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            document_id TEXT PRIMARY KEY,
            content_sha256 TEXT NOT NULL UNIQUE,
            near_duplicate_sha256 TEXT,
            stratum_id TEXT NOT NULL,
            rank TEXT NOT NULL,
            input_offset INTEGER NOT NULL,
            line_bytes INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS candidates_stratum_rank
            ON candidates(stratum_id, rank, document_id);
        CREATE UNIQUE INDEX IF NOT EXISTS candidates_near_duplicate
            ON candidates(near_duplicate_sha256)
            WHERE near_duplicate_sha256 IS NOT NULL;
        """
    )
    return connection


def _read_metadata(connection: sqlite3.Connection, key: str) -> Any | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else json.loads(row["value"])


def _write_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
    )


def _load_exclusion_sets(
    connection: sqlite3.Connection,
) -> tuple[set[str], set[str], set[str]]:
    document_ids: set[str] = set()
    content_hashes: set[str] = set()
    near_duplicate_hashes: set[str] = set()
    for row in connection.execute(
        "SELECT document_id, content_sha256, near_duplicate_sha256 FROM training_documents"
    ):
        document_ids.add(row["document_id"])
        content_hashes.add(row["content_sha256"])
        if row["near_duplicate_sha256"]:
            near_duplicate_hashes.add(row["near_duplicate_sha256"])
    return document_ids, content_hashes, near_duplicate_hashes


def _state_template(strata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_documents": 0,
        "documents_seen": 0,
        "duplicate_candidates_excluded": 0,
        "eligible_documents": 0,
        "input_offset": 0,
        "status": "scanning",
        "training_documents": {item["id"]: 0 for item in strata},
        "training_tokens": {item["id"]: 0 for item in strata},
        "training_truncated": {item["id"]: False for item in strata},
    }


def _validate_training_replay(
    state: dict[str, Any], training_manifest: dict[str, Any]
) -> None:
    expected = {item["id"]: item for item in training_manifest["statistics"]["strata"]}
    for stratum_id, item in expected.items():
        if state["training_tokens"].get(stratum_id) != item["tokens"]:
            raise RuntimeError(f"training token replay changed for {stratum_id}")
        if state["training_documents"].get(stratum_id) != item["documents_packed"]:
            raise RuntimeError(f"training document replay changed for {stratum_id}")
        if state["training_truncated"].get(stratum_id) != item["truncated_final_document"]:
            raise RuntimeError(f"training truncation replay changed for {stratum_id}")


def _scan_source(
    *,
    connection: sqlite3.Connection,
    state: dict[str, Any],
    input_path: Path,
    tokenizer,
    strata: list[dict[str, Any]],
    seed: int,
    candidate_fraction_denominator: int,
    checkpoint_documents: int,
    interrupt_after_documents: int | None,
) -> dict[str, Any]:
    strata_by_id = {item["id"]: item for item in strata}
    excluded_ids, excluded_content, excluded_near = _load_exclusion_sets(connection)
    reached_eof = False
    with input_path.open("rb") as input_handle:
        input_handle.seek(state["input_offset"])
        while not reached_eof:
            connection.execute("BEGIN IMMEDIATE")
            processed = 0
            try:
                while processed < checkpoint_documents:
                    line_offset = input_handle.tell()
                    raw_line = input_handle.readline()
                    if not raw_line:
                        reached_eof = True
                        break
                    next_offset = input_handle.tell()
                    record = json.loads(raw_line)
                    text = record.get("text")
                    if not isinstance(text, str):
                        raise TypeError("source record text must be a string")
                    _, _, stratum_id = _record_stratum(record)
                    stratum = strata_by_id.get(stratum_id)
                    if stratum is None:
                        raise ValueError(f"source record has an unknown stratum {stratum_id}")

                    state["documents_seen"] += 1
                    state["input_offset"] = next_offset
                    processed += 1
                    training_remaining = (
                        stratum["training_target_tokens"]
                        - state["training_tokens"][stratum_id]
                    )
                    if text and training_remaining > 0:
                        document_id, content_sha256, near_duplicate = _record_identity(record)
                        token_count = 1 + len(
                            tokenizer.backend_tokenizer.encode(
                                text, add_special_tokens=False
                            ).ids
                        )
                        tokens_used = min(token_count, training_remaining)
                        truncated = token_count > training_remaining
                        if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_sha256:
                            raise ValueError(f"content SHA-256 changed for {document_id}")
                        try:
                            connection.execute(
                                "INSERT INTO training_documents VALUES(?, ?, ?, ?, ?, ?, ?)",
                                (
                                    document_id,
                                    content_sha256,
                                    near_duplicate,
                                    stratum_id,
                                    line_offset,
                                    tokens_used,
                                    int(truncated),
                                ),
                            )
                        except sqlite3.IntegrityError as error:
                            raise ValueError(
                                "training replay contains duplicate documents"
                            ) from error
                        excluded_ids.add(document_id)
                        excluded_content.add(content_sha256)
                        if near_duplicate:
                            excluded_near.add(near_duplicate)
                        state["training_tokens"][stratum_id] += tokens_used
                        state["training_documents"][stratum_id] += 1
                        state["training_truncated"][stratum_id] |= truncated
                    elif text:
                        state["eligible_documents"] += 1
                        document_id, content_sha256, near_duplicate = _record_identity(record)
                        rank = _selection_rank(seed, document_id, content_sha256)
                        if _rank_is_candidate(rank, candidate_fraction_denominator):
                            overlap = (
                                document_id in excluded_ids
                                or content_sha256 in excluded_content
                                or (
                                    near_duplicate is not None
                                    and near_duplicate in excluded_near
                                )
                            )
                            if overlap:
                                state["duplicate_candidates_excluded"] += 1
                            else:
                                try:
                                    connection.execute(
                                        "INSERT INTO candidates VALUES(?, ?, ?, ?, ?, ?, ?)",
                                        (
                                            document_id,
                                            content_sha256,
                                            near_duplicate,
                                            stratum_id,
                                            rank,
                                            line_offset,
                                            len(raw_line),
                                        ),
                                    )
                                    state["candidate_documents"] += 1
                                except sqlite3.IntegrityError:
                                    state["duplicate_candidates_excluded"] += 1

                    if (
                        interrupt_after_documents is not None
                        and state["documents_seen"] >= interrupt_after_documents
                    ):
                        break
                _write_metadata(connection, "state", state)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            if (
                interrupt_after_documents is not None
                and state["documents_seen"] >= interrupt_after_documents
            ):
                raise RuntimeError("simulated held-out scan interruption")
    state["status"] = "scanned"
    _write_metadata(connection, "state", state)
    connection.commit()
    return state


class _ShardWriter:
    def __init__(self, output: Path, *, sequence_length: int, sequences_per_shard: int) -> None:
        self.output = output
        self.capacity = sequence_length * sequences_per_shard
        self.tokens_written = 0
        self._handle = None
        self._shard_index = -1

    def _ensure_handle(self):
        shard_index = self.tokens_written // self.capacity
        if self._handle is not None and shard_index == self._shard_index:
            return self._handle
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        self._shard_index = shard_index
        self._handle = (self.output / f"shard-{shard_index:05d}.bin").open("ab")
        return self._handle

    def append(self, token_ids: list[int]) -> None:
        written = 0
        while written < len(token_ids):
            offset = self.tokens_written % self.capacity
            take = min(len(token_ids) - written, self.capacity - offset)
            values = array("I", token_ids[written : written + take])
            if values.itemsize != TOKEN_BYTES:
                raise RuntimeError("platform unsigned-int width does not match packed format")
            if sys.byteorder != TOKEN_ENDIANNESS:
                values.byteswap()
            self._ensure_handle().write(values.tobytes())
            self.tokens_written += take
            written += take

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None


def _write_ledger_artifact(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "bytes": path.stat().st_size,
        "name": path.name,
        "records": len(rows),
        "sha256": _sha256_file(path),
    }


def _read_candidate_record(input_handle, row: sqlite3.Row) -> dict[str, Any]:
    input_handle.seek(row["input_offset"])
    raw_line = input_handle.read(row["line_bytes"])
    if len(raw_line) != row["line_bytes"]:
        raise RuntimeError("held-out candidate source line changed length")
    record = json.loads(raw_line)
    document_id, content_sha256, near_duplicate = _record_identity(record)
    if (
        document_id != row["document_id"]
        or content_sha256 != row["content_sha256"]
        or near_duplicate != row["near_duplicate_sha256"]
    ):
        raise RuntimeError("held-out candidate identity changed after scanning")
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != content_sha256:
        raise RuntimeError(f"held-out candidate content changed for {document_id}")
    return record


def _build_packed_output(
    *,
    connection: sqlite3.Connection,
    output: Path,
    input_path: Path,
    tokenizer,
    build_specification: dict[str, Any],
    build_fingerprint: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    temporary = output / ".packing"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    for stale in output.glob("shard-*.bin"):
        stale.unlink()
    for name in (MANIFEST_FILENAME, SELECTION_LEDGER_FILENAME, TRAINING_LEDGER_FILENAME):
        (output / name).unlink(missing_ok=True)

    packing = build_specification["packing"]
    selection = build_specification["selection"]
    writer = _ShardWriter(
        temporary,
        sequence_length=packing["sequence_length"],
        sequences_per_shard=packing["sequences_per_shard"],
    )
    training_rows = [
        {
            "content_sha256": row["content_sha256"],
            "document_id": row["document_id"],
            "near_duplicate_sha256": row["near_duplicate_sha256"],
            "stratum_id": row["stratum_id"],
            "tokens_used": row["tokens_used"],
            "truncated": bool(row["truncated"]),
        }
        for row in connection.execute(
            "SELECT * FROM training_documents ORDER BY input_offset"
        )
    ]
    training_ledger = _write_ledger_artifact(
        temporary / TRAINING_LEDGER_FILENAME, training_rows
    )
    excluded_ids = {row["document_id"] for row in training_rows}
    excluded_content = {row["content_sha256"] for row in training_rows}
    excluded_near = {
        row["near_duplicate_sha256"]
        for row in training_rows
        if row["near_duplicate_sha256"]
    }

    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_content: set[str] = set()
    selected_near: set[str] = set()
    stratum_statistics = []
    try:
        with input_path.open("rb") as input_handle:
            for stratum in selection["strata"]:
                remaining = stratum["target_tokens"]
                sequence_start = writer.tokens_written // packing["sequence_length"]
                documents = 0
                truncated_final_document = False
                candidate_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM candidates WHERE stratum_id = ?",
                    (stratum["id"],),
                ).fetchone()["count"]
                candidates = connection.execute(
                    "SELECT * FROM candidates WHERE stratum_id = ? "
                    "ORDER BY rank, document_id",
                    (stratum["id"],),
                )
                for candidate in candidates:
                    if remaining == 0:
                        break
                    record = _read_candidate_record(input_handle, candidate)
                    category, component, stratum_id = _record_stratum(record)
                    if stratum_id != stratum["id"]:
                        raise RuntimeError("held-out candidate stratum changed after scanning")
                    text_ids = tokenizer.backend_tokenizer.encode(
                        record["text"], add_special_tokens=False
                    ).ids
                    token_ids = [build_specification["tokenizer"]["doc_token_id"], *text_ids]
                    take = min(len(token_ids), remaining)
                    token_ids = token_ids[:take]
                    truncated = take < 1 + len(text_ids)
                    writer.append(token_ids)
                    remaining -= take
                    documents += 1
                    truncated_final_document = truncated
                    document_id = candidate["document_id"]
                    content_sha256 = candidate["content_sha256"]
                    near_duplicate = candidate["near_duplicate_sha256"]
                    if (
                        document_id in excluded_ids
                        or content_sha256 in excluded_content
                        or (near_duplicate is not None and near_duplicate in excluded_near)
                        or document_id in selected_ids
                        or content_sha256 in selected_content
                        or (near_duplicate is not None and near_duplicate in selected_near)
                    ):
                        raise RuntimeError("held-out selection contains a document overlap")
                    selected_ids.add(document_id)
                    selected_content.add(content_sha256)
                    if near_duplicate:
                        selected_near.add(near_duplicate)
                    selected_rows.append(
                        {
                            "category": category,
                            "component": component,
                            "content_sha256": content_sha256,
                            "document_id": document_id,
                            "near_duplicate_sha256": near_duplicate,
                            "rank": candidate["rank"],
                            "stratum_id": stratum_id,
                            "tokens_used": take,
                            "truncated": truncated,
                        }
                    )
                if remaining:
                    raise RuntimeError(
                        f"candidate fraction did not provide enough tokens for {stratum['id']}: "
                        f"missing {remaining}"
                    )
                sequence_end = writer.tokens_written // packing["sequence_length"]
                if sequence_end - sequence_start != stratum["target_sequences"]:
                    raise RuntimeError("held-out source stratum is not sequence aligned")
                stratum_statistics.append(
                    {
                        **stratum,
                        "candidate_documents": candidate_count,
                        "documents_selected": documents,
                        "sequence_end": sequence_end,
                        "sequence_start": sequence_start,
                        "tokens": stratum["target_tokens"],
                        "truncated_final_document": truncated_final_document,
                    }
                )
    finally:
        writer.close()

    selection_ledger = _write_ledger_artifact(
        temporary / SELECTION_LEDGER_FILENAME, selected_rows
    )
    shards = []
    total_sequences = 0
    for path in sorted(temporary.glob("shard-*.bin")):
        size = path.stat().st_size
        tokens = size // TOKEN_BYTES
        if size % TOKEN_BYTES or tokens % packing["sequence_length"]:
            raise RuntimeError(f"held-out shard is not sequence aligned: {path}")
        sequences = tokens // packing["sequence_length"]
        shards.append(
            {
                "bytes": size,
                "name": path.name,
                "sequences": sequences,
                "sha256": _sha256_file(path),
                "tokens": tokens,
            }
        )
        total_sequences += sequences
    if writer.tokens_written != packing["target_tokens_effective"]:
        raise RuntimeError("held-out packed token count changed")

    manifest = {
        **build_specification,
        "build_fingerprint": build_fingerprint,
        "completed_at": _utc_now(),
        "end_reason": "heldout_target",
        "leakage": {
            "content_sha256_overlap": 0,
            "document_id_overlap": 0,
            "near_duplicate_sha256_overlap": 0,
            "training_documents": len(training_rows),
            "validation_documents": len(selected_rows),
        },
        "schema_version": PACKED_DATA_SCHEMA_VERSION,
        "selection": {**selection, "strata": stratum_statistics},
        "selection_ledger": selection_ledger,
        "shards": shards,
        "statistics": {
            "candidate_documents": state["candidate_documents"],
            "documents_scanned": state["documents_seen"],
            "documents_selected": len(selected_rows),
            "duplicate_candidates_excluded": state["duplicate_candidates_excluded"],
            "eligible_documents": state["eligible_documents"],
            "sequences": total_sequences,
            "strata": stratum_statistics,
            "tokens": writer.tokens_written,
        },
        "status": "complete",
        "training_exclusion_ledger": training_ledger,
    }
    _write_json_atomic(temporary / MANIFEST_FILENAME, manifest)
    validate_heldout_manifest(temporary / MANIFEST_FILENAME)

    for path in sorted(temporary.iterdir(), key=lambda item: item.name == MANIFEST_FILENAME):
        os.replace(path, output / path.name)
    temporary.rmdir()
    return validate_heldout_manifest(output / MANIFEST_FILENAME)


def _read_ledger(path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"held-out ledger size changed: {path}")
    if _sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"held-out ledger SHA-256 changed: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if len(rows) != artifact["records"]:
        raise ValueError(f"held-out ledger record count changed: {path}")
    return rows


def validate_heldout_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate packed shards, pinned ledgers, sequence ranges, and zero overlap."""

    manifest_path = Path(manifest_path)
    manifest = validate_packed_manifest(manifest_path)
    if manifest.get("selection", {}).get("mode") != HELDOUT_SELECTION_MODE:
        raise ValueError("manifest is not a Lumi held-out evaluation artifact")
    training_rows = _read_ledger(
        manifest_path.parent / manifest["training_exclusion_ledger"]["name"],
        manifest["training_exclusion_ledger"],
    )
    selected_rows = _read_ledger(
        manifest_path.parent / manifest["selection_ledger"]["name"],
        manifest["selection_ledger"],
    )

    def values(rows: list[dict[str, Any]], key: str) -> set[str]:
        return {row[key] for row in rows if row.get(key)}

    def require_unique(rows: list[dict[str, Any]], key: str, ledger: str) -> None:
        present = [row[key] for row in rows if row.get(key)]
        if len(present) != len(set(present)):
            raise ValueError(f"{ledger} repeats {key}")

    for key in ("document_id", "content_sha256", "near_duplicate_sha256"):
        require_unique(training_rows, key, "training exclusion ledger")
        require_unique(selected_rows, key, "held-out selection ledger")

    overlaps = {
        "content_sha256_overlap": len(
            values(training_rows, "content_sha256")
            & values(selected_rows, "content_sha256")
        ),
        "document_id_overlap": len(
            values(training_rows, "document_id") & values(selected_rows, "document_id")
        ),
        "near_duplicate_sha256_overlap": len(
            values(training_rows, "near_duplicate_sha256")
            & values(selected_rows, "near_duplicate_sha256")
        ),
    }
    recorded_overlap_changed = any(
        manifest["leakage"].get(key) != value for key, value in overlaps.items()
    )
    if any(overlaps.values()) or recorded_overlap_changed:
        raise ValueError(f"held-out ledger overlaps training data: {overlaps}")
    if manifest["leakage"]["training_documents"] != len(training_rows):
        raise ValueError("training exclusion ledger count changed")
    if manifest["leakage"]["validation_documents"] != len(selected_rows):
        raise ValueError("held-out selection ledger count changed")

    expected_start = 0
    selected_tokens = {item["id"]: 0 for item in manifest["selection"]["strata"]}
    for row in selected_rows:
        if row["stratum_id"] not in selected_tokens:
            raise ValueError("held-out ledger has an unknown source stratum")
        selected_tokens[row["stratum_id"]] += row["tokens_used"]
    for item in manifest["selection"]["strata"]:
        if item["sequence_start"] != expected_start:
            raise ValueError("held-out source sequence ranges are not contiguous")
        if item["sequence_end"] - item["sequence_start"] != item["target_sequences"]:
            raise ValueError("held-out source sequence range changed")
        if selected_tokens[item["id"]] != item["target_tokens"]:
            raise ValueError("held-out ledger token total changed")
        expected_start = item["sequence_end"]
    if expected_start != manifest["statistics"]["sequences"]:
        raise ValueError("held-out sequence ranges do not cover the packed artifact")
    return manifest


def build_heldout_evaluation(
    *,
    input_path: str | Path,
    tokenizer_path: str | Path,
    training_manifest_path: str | Path,
    output: str | Path,
    target_tokens: int,
    sequence_length: int,
    sequences_per_shard: int = 256,
    seed: int = 17,
    candidate_fraction_denominator: int = 16,
    checkpoint_documents: int = 1_000,
    expected_vocab_size: int | None = None,
    expected_doc_token_id: int = 3,
    resume: bool = False,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Replay the exact training selection, then pack hash-ranked unseen documents."""

    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    training_manifest_path = Path(training_manifest_path)
    output = Path(output)
    if sequence_length < 2 or sequences_per_shard < 1:
        raise ValueError("held-out packing dimensions must be positive")
    if candidate_fraction_denominator < 1 or checkpoint_documents < 1:
        raise ValueError("candidate denominator and checkpoint size must be positive")
    training_manifest = validate_packed_manifest(training_manifest_path)
    if training_manifest.get("selection", {}).get("mode") != "source_manifest_stratified":
        raise ValueError("held-out construction requires a stratified training manifest")
    input_artifact = _file_artifact(input_path)
    expected_input = training_manifest["input"]
    if (
        input_artifact["bytes"] != expected_input["bytes"]
        or input_artifact["sha256"] != expected_input["sha256"]
    ):
        raise ValueError("held-out source does not match the training input artifact")
    tokenizer_artifact = _tokenizer_artifact(tokenizer_path)
    if tokenizer_artifact["files"] != training_manifest["tokenizer"]["files"]:
        raise ValueError("held-out tokenizer files do not match the training artifact")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = len(tokenizer)
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError("held-out tokenizer vocabulary size changed")
    if tokenizer.get_vocab().get("<|doc|>") != expected_doc_token_id:
        raise ValueError("held-out tokenizer document token changed")
    if vocab_size != training_manifest["tokenizer"]["vocab_size"]:
        raise ValueError("held-out and training tokenizer vocabularies differ")

    strata, effective_target = _allocate_evaluation_strata(
        training_manifest,
        target_tokens=target_tokens,
        sequence_length=sequence_length,
    )
    training_manifest_artifact = _file_artifact(training_manifest_path)
    build_specification = {
        "format": {
            "dtype": TOKEN_DTYPE,
            "endianness": TOKEN_ENDIANNESS,
            "layout": "flat_fixed_sequences",
            "token_bytes": TOKEN_BYTES,
        },
        "input": input_artifact,
        "packing": {
            "checkpoint_documents": checkpoint_documents,
            "max_documents": None,
            "sequence_length": sequence_length,
            "sequences_per_shard": sequences_per_shard,
            "target_tokens_effective": effective_target,
            "target_tokens_requested": target_tokens,
            "text_key": "text",
        },
        "selection": {
            "candidate_fraction_denominator": candidate_fraction_denominator,
            "mode": HELDOUT_SELECTION_MODE,
            "seed": seed,
            "strata": strata,
        },
        "tokenizer": {
            **tokenizer_artifact,
            "doc_token": "<|doc|>",
            "doc_token_id": expected_doc_token_id,
            "vocab_size": vocab_size,
        },
        "training_artifact": {
            "build_fingerprint": training_manifest["build_fingerprint"],
            "documents": training_manifest["statistics"]["documents_packed"],
            "manifest_bytes": training_manifest_artifact["bytes"],
            "manifest_sha256": training_manifest_artifact["sha256"],
            "tokens": training_manifest["statistics"]["tokens"],
        },
    }
    fingerprint_specification = json.loads(json.dumps(build_specification))
    fingerprint_specification["input"].pop("path")
    fingerprint_specification["tokenizer"].pop("path")
    build_fingerprint = _fingerprint(fingerprint_specification)

    manifest_path = output / MANIFEST_FILENAME
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"held-out manifest already exists: {manifest_path}")
        cached = validate_heldout_manifest(manifest_path)
        if cached["build_fingerprint"] != build_fingerprint:
            raise ValueError("cached held-out artifact does not match the requested build")
        return cached
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / HELDOUT_STATE_FILENAME
    if state_path.exists() and not resume:
        raise FileExistsError(f"held-out state exists; pass --resume: {state_path}")
    unexpected = [
        path for path in output.iterdir() if path.name not in {HELDOUT_STATE_FILENAME}
    ]
    if unexpected and not resume:
        raise FileExistsError(f"held-out output is not empty: {output}")

    connection = _connect_state(state_path)
    try:
        stored_fingerprint = _read_metadata(connection, "build_fingerprint")
        state = _read_metadata(connection, "state")
        if stored_fingerprint is None:
            state = _state_template(strata)
            connection.execute("BEGIN IMMEDIATE")
            _write_metadata(connection, "build_fingerprint", build_fingerprint)
            _write_metadata(connection, "state", state)
            connection.commit()
        elif stored_fingerprint != build_fingerprint:
            raise ValueError("held-out checkpoint does not match the requested build")
        elif state is None:
            raise ValueError("held-out checkpoint has no durable state")

        assert state is not None
        if state["status"] == "scanning":
            state = _scan_source(
                connection=connection,
                state=state,
                input_path=input_path,
                tokenizer=tokenizer,
                strata=strata,
                seed=seed,
                candidate_fraction_denominator=candidate_fraction_denominator,
                checkpoint_documents=checkpoint_documents,
                interrupt_after_documents=_interrupt_after_documents,
            )
        if state["status"] not in {"scanned", "complete"}:
            raise ValueError("held-out checkpoint has an invalid status")
        _validate_training_replay(state, training_manifest)
        manifest = _build_packed_output(
            connection=connection,
            output=output,
            input_path=input_path,
            tokenizer=tokenizer,
            build_specification=build_specification,
            build_fingerprint=build_fingerprint,
            state=state,
        )
        state["status"] = "complete"
        connection.execute("BEGIN IMMEDIATE")
        _write_metadata(connection, "state", state)
        connection.commit()
        return manifest
    finally:
        connection.close()
