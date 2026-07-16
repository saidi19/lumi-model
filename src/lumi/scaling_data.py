"""Deterministic, held-out-safe source indexing for Lumi scaling datasets."""

from __future__ import annotations

import hashlib
import json
import os
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
    TOKEN_ENDIANNESS,
    validate_packed_manifest,
)
from .scaling_pilot import SCALING_SELECTION_MODE, load_scaling_pilot_plan

SCALING_DATA_INDEX_SCHEMA_VERSION = 1
SCALING_DATA_PROOF_SCHEMA_VERSION = 1
INDEX_DATABASE_FILENAME = "source-index.sqlite3"
INDEX_MANIFEST_FILENAME = "manifest.json"
PACK_STATE_FILENAME = "scaling-packing-state.sqlite3"
SELECTION_PROOF_FILENAME = "selection-proof.json"
ALLOCATION_MODE = "eligible_token_proportional_largest_remainder"
SCALING_TOKEN_BYTES = 2
SCALING_TOKEN_DTYPE = "uint16"
SCALING_TOKEN_ARRAY_CODE = "H"


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


def _artifact_without_path(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "path"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stratum_id(category: str, component: str) -> str:
    return json.dumps([category, component], ensure_ascii=False, separators=(",", ":"))


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
    if near_duplicate is not None and (
        not isinstance(near_duplicate, str)
        or len(near_duplicate) != 64
        or any(character not in "0123456789abcdef" for character in near_duplicate)
    ):
        raise ValueError(f"source record {document_id!r} has an invalid near-duplicate hash")
    return document_id, content_sha256, near_duplicate


def _record_stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source record is missing provenance")
    category = provenance.get("category")
    component = provenance.get("component")
    if not isinstance(category, str) or not category:
        raise ValueError("source record has an invalid category")
    if not isinstance(component, str) or not component:
        raise ValueError("source record has an invalid component")
    return category, component, _stratum_id(category, component)


def _selection_rank(seed: int, document_id: str, content_sha256: str) -> str:
    material = f"{seed}\0{document_id}\0{content_sha256}".encode()
    return hashlib.sha256(material).hexdigest()


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
            artifact["name"] = name
            artifact.pop("path")
            files.append(artifact)
    return {"files": files, "path": str(path.resolve())}


def _source_strata(source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    strata = []
    seen: set[str] = set()
    for source in source_manifest.get("source_files", []):
        if not source.get("used"):
            continue
        category = source.get("category")
        component = source.get("component")
        statistics = source.get("statistics", {})
        documents = statistics.get("documents")
        target_bytes = statistics.get("target_text_utf8_bytes")
        if not isinstance(category, str) or not isinstance(component, str):
            raise ValueError("source manifest contains an invalid source stratum")
        if not isinstance(documents, int) or documents < 1:
            raise ValueError("source manifest stratum has an invalid document count")
        if not isinstance(target_bytes, int) or target_bytes < 1:
            raise ValueError("source manifest stratum has an invalid byte target")
        identifier = _stratum_id(category, component)
        if identifier in seen:
            raise ValueError(f"source manifest repeats stratum {identifier}")
        seen.add(identifier)
        strata.append(
            {
                "category": category,
                "component": component,
                "documents": documents,
                "id": identifier,
                "target_text_utf8_bytes": target_bytes,
            }
        )
    if not strata:
        raise ValueError("source manifest contains no used strata")
    return strata


def _load_heldout_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows = []
    indexes: dict[str, dict[str, str]] = {
        "document_id": {},
        "content_sha256": {},
        "near_duplicate_sha256": {},
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            document_id = row.get("document_id")
            content_sha256 = row.get("content_sha256")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"held-out ledger row {line_number} has no document ID")
            if not isinstance(content_sha256, str) or len(content_sha256) != 64:
                raise ValueError(f"held-out ledger row {line_number} has no content hash")
            for key in indexes:
                value = row.get(key)
                if value is None and key == "near_duplicate_sha256":
                    continue
                if not isinstance(value, str) or not value:
                    raise ValueError(f"held-out ledger row {line_number} has invalid {key}")
                if value in indexes[key]:
                    raise ValueError(f"held-out ledger repeats {key}")
                indexes[key][value] = document_id
            rows.append(row)
    if not rows:
        raise ValueError("held-out selection ledger is empty")
    return rows, indexes


def _heldout_match(
    *,
    document_id: str,
    content_sha256: str,
    near_duplicate_sha256: str | None,
    stratum_id: str,
    rows_by_id: dict[str, dict[str, Any]],
    indexes: dict[str, dict[str, str]],
) -> tuple[bool, bool]:
    direct_matches = {
        value
        for value in (
            indexes["document_id"].get(document_id),
            indexes["content_sha256"].get(content_sha256),
        )
        if value is not None
    }
    near_match = (
        indexes["near_duplicate_sha256"].get(near_duplicate_sha256)
        if near_duplicate_sha256
        else None
    )
    if direct_matches:
        if len(direct_matches) != 1:
            raise ValueError("source identity maps to multiple held-out documents")
        expected_id = next(iter(direct_matches))
        if near_match is not None and near_match != expected_id:
            raise ValueError("source identity and near-duplicate family disagree")
        expected = rows_by_id[expected_id]
        if (
            expected["document_id"] != document_id
            or expected["content_sha256"] != content_sha256
            or expected.get("near_duplicate_sha256") != near_duplicate_sha256
            or expected.get("stratum_id") != stratum_id
        ):
            raise ValueError("source identity only partially matches a held-out document")
        return True, True
    # A different document in the same near-duplicate family must also be
    # excluded, but it is not one of the exact ledger rows.
    if near_match is not None:
        return True, False
    return False, False


def create_scaling_data_contract(
    *,
    input_path: str | Path,
    tokenizer_path: str | Path,
    source_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    heldout_selection_ledger_path: str | Path,
    token_count_report_path: str | Path,
    seed: int,
    final_target_tokens: int,
    sequence_length: int,
    expected_vocab_size: int,
    expected_doc_token_id: int = 3,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source evidence and return a reusable, path-aware index contract."""

    paths = {
        "input": Path(input_path),
        "tokenizer": Path(tokenizer_path),
        "source_manifest": Path(source_manifest_path),
        "heldout_manifest": Path(heldout_manifest_path),
        "heldout_ledger": Path(heldout_selection_ledger_path),
        "token_count_report": Path(token_count_report_path),
    }
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError("selection seed must be a positive integer")
    if final_target_tokens < sequence_length or sequence_length < 2:
        raise ValueError("final target and sequence length are invalid")
    if expected_vocab_size < 4 or expected_vocab_size > 65_536:
        raise ValueError("expected vocabulary size is invalid")

    input_artifact = _file_artifact(paths["input"])
    source_manifest_artifact = _file_artifact(paths["source_manifest"])
    source_manifest = _read_json(paths["source_manifest"])
    if source_manifest.get("status") != "complete":
        raise ValueError("source manifest is not complete")
    source_output = source_manifest.get("output", {})
    if source_output.get("file_bytes") != input_artifact["bytes"]:
        raise ValueError("source manifest byte count does not match the input")
    if source_output.get("sha256") != input_artifact["sha256"]:
        raise ValueError("source manifest hash does not match the input")
    source_documents = source_manifest.get("statistics", {}).get("documents")
    if not isinstance(source_documents, int) or source_documents < 1:
        raise ValueError("source manifest has an invalid document count")
    strata = _source_strata(source_manifest)
    if sum(item["documents"] for item in strata) != source_documents:
        raise ValueError("source stratum document counts do not match the source total")

    heldout_manifest_artifact = _file_artifact(paths["heldout_manifest"])
    heldout_manifest = _read_json(paths["heldout_manifest"])
    if heldout_manifest.get("status") != "complete":
        raise ValueError("held-out manifest is not complete")
    heldout_ledger_artifact = _file_artifact(paths["heldout_ledger"])
    ledger_spec = heldout_manifest.get("selection_ledger", {})
    heldout_rows, _ = _load_heldout_rows(paths["heldout_ledger"])
    if ledger_spec.get("sha256") != heldout_ledger_artifact["sha256"]:
        raise ValueError("held-out manifest does not pin the supplied selection ledger")
    if ledger_spec.get("bytes") != heldout_ledger_artifact["bytes"]:
        raise ValueError("held-out selection-ledger byte count changed")
    if ledger_spec.get("records") != len(heldout_rows):
        raise ValueError("held-out selection-ledger record count changed")

    token_count_artifact = _file_artifact(paths["token_count_report"])
    token_count_report = _read_json(paths["token_count_report"])
    selected_tokenizer = token_count_report.get("tokenizers", {}).get("65k", {})
    observed_tokens = selected_tokenizer.get("overall", {}).get("tokens")
    if not isinstance(observed_tokens, int) or observed_tokens < 1:
        raise ValueError("token-count report has no 65K observed-token total")
    if selected_tokenizer.get("artifact", {}).get("vocab_size") != expected_vocab_size:
        raise ValueError("token-count report vocabulary size changed")

    tokenizer_artifact = _tokenizer_artifact(paths["tokenizer"])
    tokenizer_json = next(
        item for item in tokenizer_artifact["files"] if item["name"] == "tokenizer.json"
    )
    if (
        selected_tokenizer.get("artifact", {}).get("tokenizer_json_sha256")
        != tokenizer_json["sha256"]
    ):
        raise ValueError("token-count report references a different tokenizer")
    heldout_tokenizer = heldout_manifest.get("tokenizer")
    if not isinstance(heldout_tokenizer, dict):
        raise ValueError("held-out manifest has no tokenizer contract")
    if heldout_tokenizer.get("files") != tokenizer_artifact["files"]:
        raise ValueError("held-out manifest references a different tokenizer")
    if heldout_tokenizer.get("vocab_size") != expected_vocab_size:
        raise ValueError("held-out tokenizer vocabulary size changed")
    if heldout_tokenizer.get("doc_token_id") != expected_doc_token_id:
        raise ValueError("held-out tokenizer document-token ID changed")
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    if len(tokenizer) != expected_vocab_size:
        raise ValueError("tokenizer vocabulary size changed")
    if tokenizer.get_vocab().get("<|doc|>") != expected_doc_token_id:
        raise ValueError("tokenizer document-token ID changed")

    plan_artifact = None
    if plan_path is not None:
        plan_path = Path(plan_path)
        plan = load_scaling_pilot_plan(plan_path)
        plan_artifact = _file_artifact(plan_path)
        plan_data = plan["data"]
        expected = plan_data["source_sample"]
        heldout_expected = plan_data["heldout"]
        if source_manifest_artifact["sha256"] != expected["manifest_sha256"]:
            raise ValueError("source manifest does not match the scaling plan")
        if input_artifact["sha256"] != expected["sample_sha256"]:
            raise ValueError("source input does not match the scaling plan")
        if source_documents != expected["documents"]:
            raise ValueError("source document count does not match the scaling plan")
        if observed_tokens != expected["observed_tokens"]:
            raise ValueError("observed-token count does not match the scaling plan")
        if token_count_artifact["sha256"] != expected["token_count_report_sha256"]:
            raise ValueError("token-count report does not match the scaling plan")
        if heldout_manifest_artifact["sha256"] != heldout_expected["manifest_sha256"]:
            raise ValueError("held-out manifest does not match the scaling plan")
        if heldout_ledger_artifact["sha256"] != heldout_expected["selection_ledger_sha256"]:
            raise ValueError("held-out ledger does not match the scaling plan")
        if len(heldout_rows) != heldout_expected["documents"]:
            raise ValueError("held-out document count does not match the scaling plan")
        if seed != plan_data["selection"]["seed"]:
            raise ValueError("selection seed does not match the scaling plan")
        if final_target_tokens != plan_data["selection"]["target_tokens_requested"]:
            raise ValueError("final target does not match the scaling plan")
        if sequence_length != plan_data["sequence_length"]:
            raise ValueError("sequence length does not match the scaling plan")
        if len(strata) != plan_data["selection"]["source_strata"]:
            raise ValueError("source-stratum count does not match the scaling plan")

    return {
        "heldout": {
            "documents": len(heldout_rows),
            "ledger": heldout_ledger_artifact,
            "manifest": heldout_manifest_artifact,
        },
        "input": input_artifact,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "plan": plan_artifact,
        "schema_version": SCALING_DATA_INDEX_SCHEMA_VERSION,
        "selection": {
            "final_target_tokens": final_target_tokens,
            "mode": SCALING_SELECTION_MODE,
            "seed": seed,
            "sequence_length": sequence_length,
            "strata": strata,
        },
        "source": {
            "documents": source_documents,
            "manifest": source_manifest_artifact,
        },
        "token_count_report": {
            "artifact": token_count_artifact,
            "categories": selected_tokenizer.get("categories", {}),
            "observed_text_tokens": observed_tokens,
        },
        "tokenizer": {
            **tokenizer_artifact,
            "doc_token": "<|doc|>",
            "doc_token_id": expected_doc_token_id,
            "vocab_size": expected_vocab_size,
        },
    }


def _contract_fingerprint_specification(contract: dict[str, Any]) -> dict[str, Any]:
    specification = json.loads(json.dumps(contract))
    specification.pop("paths", None)
    for key in ("input",):
        if isinstance(specification.get(key), dict):
            specification[key].pop("path", None)
    for group, field in (
        ("heldout", "ledger"),
        ("heldout", "manifest"),
        ("source", "manifest"),
        ("token_count_report", "artifact"),
    ):
        artifact = specification.get(group, {}).get(field)
        if isinstance(artifact, dict):
            artifact.pop("path", None)
    if isinstance(specification.get("plan"), dict):
        specification["plan"].pop("path", None)
    if isinstance(specification.get("tokenizer"), dict):
        specification["tokenizer"].pop("path", None)
    return specification


def _connect_index(path: Path) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS documents (
            input_offset INTEGER PRIMARY KEY,
            line_bytes INTEGER NOT NULL,
            document_id TEXT NOT NULL UNIQUE,
            content_sha256 TEXT NOT NULL UNIQUE,
            near_duplicate_sha256 TEXT,
            stratum_id TEXT NOT NULL,
            rank TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            heldout INTEGER NOT NULL CHECK(heldout IN (0, 1)),
            heldout_exact INTEGER NOT NULL CHECK(heldout_exact IN (0, 1))
        );
        CREATE INDEX IF NOT EXISTS documents_near_duplicate
            ON documents(near_duplicate_sha256)
            WHERE near_duplicate_sha256 IS NOT NULL;
        CREATE INDEX IF NOT EXISTS documents_stratum_rank
            ON documents(stratum_id, heldout, rank, document_id);
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


def _initial_index_state(strata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents_seen": 0,
        "input_offset": 0,
        "source_text_tokens": 0,
        "status": "scanning",
        "strata": {
            item["id"]: {
                "documents": 0,
                "eligible_documents": 0,
                "eligible_tokens": 0,
                "heldout_documents": 0,
                "heldout_exact_documents": 0,
                "heldout_tokens": 0,
                "text_tokens": 0,
                "tokens": 0,
            }
            for item in strata
        },
    }


def _finalize_sqlite(connection: sqlite3.Connection) -> None:
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("PRAGMA journal_mode=DELETE")


def _index_statistics(state: dict[str, Any], strata: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for specification in strata:
        values = state["strata"][specification["id"]]
        items.append({**specification, **values})
    return {
        "documents": state["documents_seen"],
        "eligible_documents": sum(item["eligible_documents"] for item in items),
        "eligible_tokens": sum(item["eligible_tokens"] for item in items),
        "heldout_documents": sum(item["heldout_documents"] for item in items),
        "heldout_exact_documents": sum(
            item["heldout_exact_documents"] for item in items
        ),
        "heldout_tokens": sum(item["heldout_tokens"] for item in items),
        "source_text_tokens": state["source_text_tokens"],
        "strata": items,
        "tokens_with_document_markers": sum(item["tokens"] for item in items),
    }


def _validate_index_totals(contract: dict[str, Any], statistics: dict[str, Any]) -> None:
    if statistics["documents"] != contract["source"]["documents"]:
        raise RuntimeError("indexed document count does not match the sealed source")
    if statistics["source_text_tokens"] != contract["token_count_report"][
        "observed_text_tokens"
    ]:
        raise RuntimeError("indexed text-token total does not match the frozen tokenizer audit")
    if statistics["heldout_exact_documents"] != contract["heldout"]["documents"]:
        raise RuntimeError("not every held-out document was found and excluded")
    if statistics["eligible_tokens"] < contract["selection"]["final_target_tokens"]:
        raise RuntimeError(
            "eligible source capacity cannot cover the planned final target: "
            f"{statistics['eligible_tokens']} < {contract['selection']['final_target_tokens']}"
        )
    expected = {item["id"]: item for item in contract["selection"]["strata"]}
    actual = {item["id"]: item for item in statistics["strata"]}
    if set(actual) != set(expected):
        raise RuntimeError("indexed source strata changed")
    for identifier, specification in expected.items():
        if actual[identifier]["documents"] != specification["documents"]:
            raise RuntimeError(f"indexed document count changed for {identifier}")

    expected_categories = contract["token_count_report"].get("categories", {})
    if expected_categories:
        category_totals: dict[str, dict[str, int]] = {}
        for item in statistics["strata"]:
            totals = category_totals.setdefault(item["category"], {"documents": 0, "tokens": 0})
            totals["documents"] += item["documents"]
            totals["tokens"] += item["text_tokens"]
        for category, specification in expected_categories.items():
            if category_totals.get(category) != {
                "documents": specification.get("documents"),
                "tokens": specification.get("tokens"),
            }:
                raise RuntimeError(f"indexed tokenizer totals changed for category {category}")


def validate_scaling_source_index(manifest_path: str | Path) -> dict[str, Any]:
    """Validate one complete immutable scaling source index."""

    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCALING_DATA_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported scaling source-index schema version")
    if manifest.get("status") != "complete":
        raise ValueError("scaling source index is not complete")
    database = manifest.get("database")
    if not isinstance(database, dict) or Path(database.get("name", "")).name != database.get(
        "name"
    ):
        raise ValueError("scaling source-index database artifact is invalid")
    database_path = manifest_path.parent / database["name"]
    if not database_path.is_file() or database_path.stat().st_size != database.get("bytes"):
        raise ValueError("scaling source-index database size changed")
    if _sha256_file(database_path) != database.get("sha256"):
        raise ValueError("scaling source-index database hash changed")

    uri = f"file:{database_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if _read_metadata(connection, "build_fingerprint") != manifest.get(
            "build_fingerprint"
        ):
            raise ValueError("source-index database fingerprint changed")
        state = _read_metadata(connection, "state")
        if not isinstance(state, dict) or state.get("status") != "complete":
            raise ValueError("source-index database is not sealed")
        aggregate = connection.execute(
            "SELECT COUNT(*) AS documents, "
            "COALESCE(SUM(token_count), 0) AS tokens, "
            "COALESCE(SUM(CASE WHEN heldout = 0 THEN 1 ELSE 0 END), 0) "
            "AS eligible_documents, "
            "COALESCE(SUM(CASE WHEN heldout = 0 THEN token_count ELSE 0 END), 0) "
            "AS eligible_tokens, "
            "COALESCE(SUM(heldout), 0) AS heldout_documents "
            ", COALESCE(SUM(heldout_exact), 0) AS heldout_exact_documents "
            "FROM documents"
        ).fetchone()
        statistics = manifest["statistics"]
        for key in (
            "documents",
            "eligible_documents",
            "eligible_tokens",
            "heldout_documents",
            "heldout_exact_documents",
        ):
            if aggregate[key] != statistics[key]:
                raise ValueError(f"source-index database {key} changed")
        if aggregate["tokens"] != statistics["tokens_with_document_markers"]:
            raise ValueError("source-index database token total changed")
    finally:
        connection.close()
    return manifest


def build_scaling_source_index(
    *,
    contract: dict[str, Any],
    output: str | Path,
    checkpoint_documents: int = 256,
    checkpoint_text_bytes: int = 16 * 1024 * 1024,
    progress_documents: int = 10_000,
    resume: bool = False,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Scan/tokenize the sealed source once and build a reusable capacity index."""

    output = Path(output)
    if contract.get("schema_version") != SCALING_DATA_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported scaling data contract")
    for value, name in (
        (checkpoint_documents, "checkpoint-documents"),
        (checkpoint_text_bytes, "checkpoint-text-bytes"),
        (progress_documents, "progress-documents"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    input_path = Path(contract["paths"]["input"])
    tokenizer_path = Path(contract["paths"]["tokenizer"])
    ledger_path = Path(contract["paths"]["heldout_ledger"])
    if _file_artifact(input_path) != contract["input"]:
        raise ValueError("source input changed after contract validation")
    if _file_artifact(ledger_path) != contract["heldout"]["ledger"]:
        raise ValueError("held-out ledger changed after contract validation")
    if _tokenizer_artifact(tokenizer_path) != {
        "files": contract["tokenizer"]["files"],
        "path": contract["tokenizer"]["path"],
    }:
        raise ValueError("tokenizer changed after contract validation")

    build_specification = _contract_fingerprint_specification(contract)
    build_fingerprint = _fingerprint(build_specification)
    manifest_path = output / INDEX_MANIFEST_FILENAME
    database_path = output / INDEX_DATABASE_FILENAME
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"source-index manifest already exists: {manifest_path}")
        cached = validate_scaling_source_index(manifest_path)
        if cached["build_fingerprint"] != build_fingerprint:
            raise ValueError("cached source index does not match the requested contract")
        return cached
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output.iterdir() if path.name != INDEX_DATABASE_FILENAME]
    if unexpected and not resume:
        raise FileExistsError(f"source-index output is not empty: {output}")
    if database_path.exists() and not resume:
        raise FileExistsError(f"source-index state exists; pass --resume: {database_path}")

    heldout_rows, heldout_indexes = _load_heldout_rows(ledger_path)
    heldout_by_id = {row["document_id"]: row for row in heldout_rows}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if len(tokenizer) != contract["tokenizer"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary changed before indexing")
    strata = contract["selection"]["strata"]
    strata_by_id = {item["id"]: item for item in strata}
    connection = _connect_index(database_path)
    try:
        stored_fingerprint = _read_metadata(connection, "build_fingerprint")
        state = _read_metadata(connection, "state")
        if stored_fingerprint is None:
            if state is not None:
                raise ValueError("source-index state has no build fingerprint")
            state = _initial_index_state(strata)
            connection.execute("BEGIN IMMEDIATE")
            _write_metadata(connection, "build_fingerprint", build_fingerprint)
            _write_metadata(connection, "state", state)
            connection.commit()
        elif stored_fingerprint != build_fingerprint:
            raise ValueError("source-index checkpoint does not match the contract")
        elif not isinstance(state, dict):
            raise ValueError("source-index checkpoint has no durable state")
        assert isinstance(state, dict)
        if state.get("status") not in {"scanning", "complete"}:
            raise ValueError("source-index checkpoint has an invalid status")

        next_progress = (
            (state["documents_seen"] // progress_documents) + 1
        ) * progress_documents
        if state["status"] == "scanning":
            with input_path.open("rb") as input_handle:
                input_handle.seek(state["input_offset"])
                reached_eof = False
                while not reached_eof:
                    batch = []
                    batch_text_bytes = 0
                    while len(batch) < checkpoint_documents:
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
                        document_id, content_sha256, near_duplicate = _record_identity(record)
                        category, component, stratum_id = _record_stratum(record)
                        if stratum_id not in strata_by_id:
                            raise ValueError(f"source record has unknown stratum {stratum_id}")
                        text_bytes = text.encode("utf-8")
                        if hashlib.sha256(text_bytes).hexdigest() != content_sha256:
                            raise ValueError(f"source content changed for {document_id}")
                        heldout, heldout_exact = _heldout_match(
                            document_id=document_id,
                            content_sha256=content_sha256,
                            near_duplicate_sha256=near_duplicate,
                            stratum_id=stratum_id,
                            rows_by_id=heldout_by_id,
                            indexes=heldout_indexes,
                        )
                        batch.append(
                            {
                                "category": category,
                                "component": component,
                                "content_sha256": content_sha256,
                                "document_id": document_id,
                                "heldout": heldout,
                                "heldout_exact": heldout_exact,
                                "input_offset": line_offset,
                                "line_bytes": len(raw_line),
                                "near_duplicate_sha256": near_duplicate,
                                "next_offset": next_offset,
                                "stratum_id": stratum_id,
                                "text": text,
                            }
                        )
                        batch_text_bytes += len(text_bytes)
                        if batch_text_bytes >= checkpoint_text_bytes:
                            break
                        if (
                            _interrupt_after_documents is not None
                            and state["documents_seen"] + len(batch)
                            >= _interrupt_after_documents
                        ):
                            break
                    if not batch:
                        break
                    encodings = tokenizer.backend_tokenizer.encode_batch(
                        [item["text"] for item in batch], add_special_tokens=False
                    )
                    if len(encodings) != len(batch):
                        raise RuntimeError("tokenizer batch cardinality changed")
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        for item, encoding in zip(batch, encodings, strict=True):
                            token_count = 1 + len(encoding.ids)
                            if token_count < 1:
                                raise RuntimeError("source document produced no marker token")
                            if any(
                                token_id < 0 or token_id >= contract["tokenizer"]["vocab_size"]
                                for token_id in encoding.ids
                            ):
                                raise ValueError("tokenizer emitted an out-of-vocabulary ID")
                            rank = _selection_rank(
                                contract["selection"]["seed"],
                                item["document_id"],
                                item["content_sha256"],
                            )
                            try:
                                connection.execute(
                                    "INSERT INTO documents VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        item["input_offset"],
                                        item["line_bytes"],
                                        item["document_id"],
                                        item["content_sha256"],
                                        item["near_duplicate_sha256"],
                                        item["stratum_id"],
                                        rank,
                                        token_count,
                                        int(item["heldout"]),
                                        int(item["heldout_exact"]),
                                    ),
                                )
                            except sqlite3.IntegrityError as error:
                                raise ValueError(
                                    "sealed source contains duplicate identities"
                                ) from error
                            values = state["strata"][item["stratum_id"]]
                            values["documents"] += 1
                            values["text_tokens"] += token_count - 1
                            values["tokens"] += token_count
                            if item["heldout"]:
                                values["heldout_documents"] += 1
                                values["heldout_tokens"] += token_count
                                if item["heldout_exact"]:
                                    values["heldout_exact_documents"] += 1
                            else:
                                values["eligible_documents"] += 1
                                values["eligible_tokens"] += token_count
                            state["documents_seen"] += 1
                            state["source_text_tokens"] += token_count - 1
                            state["input_offset"] = item["next_offset"]
                        _write_metadata(connection, "state", state)
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
                    if state["documents_seen"] >= next_progress:
                        eligible = sum(
                            item["eligible_tokens"] for item in state["strata"].values()
                        )
                        print(
                            f"indexed_documents={state['documents_seen']:,} "
                            f"eligible_tokens={eligible:,}",
                            flush=True,
                        )
                        next_progress += progress_documents
                    if (
                        _interrupt_after_documents is not None
                        and state["documents_seen"] >= _interrupt_after_documents
                    ):
                        raise RuntimeError("simulated scaling source-index interruption")

            statistics = _index_statistics(state, strata)
            _validate_index_totals(contract, statistics)
            state["completed_at"] = _utc_now()
            state["status"] = "complete"
            connection.execute("BEGIN IMMEDIATE")
            _write_metadata(connection, "state", state)
            connection.commit()
        else:
            statistics = _index_statistics(state, strata)
            _validate_index_totals(contract, statistics)
        _finalize_sqlite(connection)
    finally:
        connection.close()

    database_artifact = _file_artifact(database_path)
    database_artifact["name"] = database_path.name
    database_artifact.pop("path")
    manifest = {
        "build_fingerprint": build_fingerprint,
        "completed_at": state["completed_at"],
        "contract": build_specification,
        "database": database_artifact,
        "schema_version": SCALING_DATA_INDEX_SCHEMA_VERSION,
        "statistics": statistics,
        "status": "complete",
    }
    _write_json_atomic(manifest_path, manifest)
    return validate_scaling_source_index(manifest_path)


def _allocate_scaling_strata(
    index_manifest: dict[str, Any], *, target_tokens: int
) -> list[dict[str, Any]]:
    capacities = index_manifest["statistics"]["strata"]
    total_capacity = sum(item["eligible_tokens"] for item in capacities)
    if target_tokens > total_capacity:
        raise ValueError("packed target exceeds eligible indexed capacity")
    allocated = 0
    strata = []
    for item in capacities:
        numerator = target_tokens * item["eligible_tokens"]
        quota = numerator // total_capacity
        strata.append(
            {
                "allocation_remainder": numerator % total_capacity,
                "category": item["category"],
                "component": item["component"],
                "eligible_documents": item["eligible_documents"],
                "eligible_tokens": item["eligible_tokens"],
                "id": item["id"],
                "target_tokens": quota,
            }
        )
        allocated += quota
    leftovers = target_tokens - allocated
    order = sorted(
        strata,
        key=lambda item: (
            -item["allocation_remainder"],
            item["category"],
            item["component"],
        ),
    )
    for item in order[:leftovers]:
        item["target_tokens"] += 1
    for item in strata:
        item.pop("allocation_remainder")
        if item["target_tokens"] < 1:
            raise ValueError("packed target is too small to preserve every source stratum")
        if item["target_tokens"] > item["eligible_tokens"]:
            raise RuntimeError(f"allocated target exceeds capacity for {item['id']}")
    if sum(item["target_tokens"] for item in strata) != target_tokens:
        raise RuntimeError("source-stratum allocation does not add up to the target")
    return strata


def _connect_pack_state(path: Path) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS selected_documents (
            input_offset INTEGER PRIMARY KEY,
            tokens_used INTEGER NOT NULL,
            truncated INTEGER NOT NULL CHECK(truncated IN (0, 1))
        );
        """
    )
    return connection


class _ScalingShardWriter:
    def __init__(
        self,
        output: Path,
        *,
        sequence_length: int,
        sequences_per_shard: int,
        tokens_written: int,
    ) -> None:
        self.output = output
        self.capacity = sequence_length * sequences_per_shard
        self.tokens_written = tokens_written
        self._handle = None
        self._shard_index = -1

    def _ensure_handle(self):
        shard_index = self.tokens_written // self.capacity
        if self._handle is not None and shard_index == self._shard_index:
            return self._handle
        self.flush_close()
        self._shard_index = shard_index
        self._handle = (self.output / f"shard-{shard_index:05d}.bin").open("ab")
        return self._handle

    def append(self, token_ids: list[int]) -> None:
        written = 0
        while written < len(token_ids):
            offset = self.tokens_written % self.capacity
            take = min(len(token_ids) - written, self.capacity - offset)
            values = array(SCALING_TOKEN_ARRAY_CODE, token_ids[written : written + take])
            if values.itemsize != SCALING_TOKEN_BYTES:
                raise RuntimeError("platform unsigned-short width changed")
            if sys.byteorder != TOKEN_ENDIANNESS:
                values.byteswap()
            self._ensure_handle().write(values.tobytes())
            self.tokens_written += take
            written += take

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def flush_close(self) -> None:
        if self._handle is not None:
            self.flush()
            self._handle.close()
            self._handle = None


def _reconcile_scaling_shards(
    output: Path, *, committed_tokens: int, shard_capacity: int
) -> None:
    expected_shards = (committed_tokens + shard_capacity - 1) // shard_capacity
    existing = {path.name: path for path in output.glob("shard-*.bin")}
    for shard_index in range(expected_shards):
        path = output / f"shard-{shard_index:05d}.bin"
        expected_tokens = min(shard_capacity, committed_tokens - shard_index * shard_capacity)
        expected_bytes = expected_tokens * SCALING_TOKEN_BYTES
        if not path.is_file():
            raise RuntimeError(f"packing state references missing shard: {path}")
        if path.stat().st_size < expected_bytes:
            raise RuntimeError(f"packed shard is shorter than committed state: {path}")
        if path.stat().st_size > expected_bytes:
            with path.open("r+b") as handle:
                handle.truncate(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        existing.pop(path.name, None)
    for path in existing.values():
        path.unlink()


def _pack_fingerprint_specification(specification: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(specification))
    if isinstance(value.get("input"), dict):
        value["input"].pop("path", None)
    if isinstance(value.get("tokenizer"), dict):
        value["tokenizer"].pop("path", None)
    source_index = value.get("selection", {}).get("source_index")
    if isinstance(source_index, dict):
        source_index.pop("path", None)
    return value


def _artifact_from_output(path: Path) -> dict[str, Any]:
    artifact = _file_artifact(path)
    artifact["name"] = path.name
    artifact.pop("path")
    return artifact


def _read_source_record(input_handle, row: sqlite3.Row) -> dict[str, Any]:
    input_handle.seek(row["input_offset"])
    raw_line = input_handle.read(row["line_bytes"])
    if len(raw_line) != row["line_bytes"]:
        raise RuntimeError("indexed source line changed length")
    record = json.loads(raw_line)
    document_id, content_sha256, near_duplicate = _record_identity(record)
    _, _, stratum_id = _record_stratum(record)
    if (
        document_id != row["document_id"]
        or content_sha256 != row["content_sha256"]
        or near_duplicate != row["near_duplicate_sha256"]
        or stratum_id != row["stratum_id"]
    ):
        raise RuntimeError("indexed source identity changed during packing")
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != content_sha256:
        raise RuntimeError(f"indexed source content changed for {document_id}")
    return record


def _seal_scaling_shards(
    output: Path, *, sequence_length: int
) -> tuple[list[dict[str, Any]], int]:
    shards = []
    sequences = 0
    for path in sorted(output.glob("shard-*.bin")):
        size = path.stat().st_size
        tokens = size // SCALING_TOKEN_BYTES
        if size == 0 or size % SCALING_TOKEN_BYTES or tokens % sequence_length:
            raise RuntimeError(f"scaling packed shard is not sequence aligned: {path}")
        shard_sequences = tokens // sequence_length
        shards.append(
            {
                "bytes": size,
                "name": path.name,
                "sequences": shard_sequences,
                "sha256": _sha256_file(path),
                "tokens": tokens,
            }
        )
        sequences += shard_sequences
    if not shards:
        raise RuntimeError("scaling packed output has no shards")
    return shards, sequences


def validate_scaling_packed_manifest(
    manifest_path: str | Path,
    *,
    index_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate packed shards, the compact selection state, and hash-prefix proof."""

    manifest_path = Path(manifest_path)
    manifest = validate_packed_manifest(manifest_path)
    if manifest.get("selection", {}).get("mode") != SCALING_SELECTION_MODE:
        raise ValueError("manifest is not a held-out-safe Lumi scaling artifact")
    proof_artifact = manifest.get("selection_proof")
    state_artifact = manifest.get("selection_state")
    for artifact, label in (
        (proof_artifact, "selection proof"),
        (state_artifact, "selection state"),
    ):
        if not isinstance(artifact, dict):
            raise ValueError(f"scaling {label} artifact is missing")
        name = artifact.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"scaling {label} artifact name is invalid")
        path = manifest_path.parent / name
        if not path.is_file() or path.stat().st_size != artifact.get("bytes"):
            raise ValueError(f"scaling {label} artifact size changed")
        if _sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"scaling {label} artifact hash changed")

    proof = _read_json(manifest_path.parent / proof_artifact["name"])
    if proof.get("schema_version") != SCALING_DATA_PROOF_SCHEMA_VERSION:
        raise ValueError("unsupported scaling selection-proof schema")
    if proof.get("build_fingerprint") != manifest.get("build_fingerprint"):
        raise ValueError("scaling selection-proof fingerprint changed")
    if proof.get("selected_tokens") != manifest["statistics"]["tokens"]:
        raise ValueError("scaling selection-proof token total changed")
    if proof.get("selected_documents") != manifest["statistics"]["documents_selected"]:
        raise ValueError("scaling selection-proof document count changed")

    state_path = manifest_path.parent / state_artifact["name"]
    uri = f"file:{state_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if _read_metadata(connection, "build_fingerprint") != manifest.get(
            "build_fingerprint"
        ):
            raise ValueError("scaling selection-state fingerprint changed")
        state = _read_metadata(connection, "state")
        if not isinstance(state, dict) or state.get("status") != "complete":
            raise ValueError("scaling selection-state is not complete")
        aggregate = connection.execute(
            "SELECT COUNT(*) AS documents, COALESCE(SUM(tokens_used), 0) AS tokens, "
            "COALESCE(SUM(truncated), 0) AS truncated FROM selected_documents"
        ).fetchone()
        if aggregate["documents"] != manifest["statistics"]["documents_selected"]:
            raise ValueError("selected-document count changed in scaling state")
        if aggregate["tokens"] != manifest["statistics"]["tokens"]:
            raise ValueError("selected-token count changed in scaling state")

        if index_manifest_path is not None:
            index_manifest_path = Path(index_manifest_path)
            index_manifest = validate_scaling_source_index(index_manifest_path)
            index_artifact = _file_artifact(index_manifest_path)
            expected_index = manifest["selection"]["source_index"]
            if index_artifact["sha256"] != expected_index["sha256"]:
                raise ValueError("packed artifact references a different source index")
            if index_manifest["build_fingerprint"] != expected_index["build_fingerprint"]:
                raise ValueError("packed artifact source-index fingerprint changed")
            index_database = index_manifest_path.parent / index_manifest["database"]["name"]
            connection.execute("ATTACH DATABASE ? AS source_index", (str(index_database),))
            overlap = connection.execute(
                "SELECT COUNT(*) AS count FROM selected_documents s "
                "JOIN source_index.documents d ON d.input_offset = s.input_offset "
                "WHERE d.heldout = 1"
            ).fetchone()["count"]
            if overlap:
                raise ValueError("scaling selection includes held-out documents")
            for item in manifest["selection"]["strata"]:
                threshold_rank = item["rank_threshold"]
                threshold_document = item["threshold_document_id"]
                prefix = connection.execute(
                    "SELECT COUNT(*) AS documents, COALESCE(SUM(token_count), 0) AS tokens "
                    "FROM source_index.documents WHERE heldout = 0 AND stratum_id = ? "
                    "AND (rank < ? OR (rank = ? AND document_id <= ?))",
                    (item["id"], threshold_rank, threshold_rank, threshold_document),
                ).fetchone()
                before = connection.execute(
                    "SELECT COALESCE(SUM(token_count), 0) AS tokens "
                    "FROM source_index.documents WHERE heldout = 0 AND stratum_id = ? "
                    "AND (rank < ? OR (rank = ? AND document_id < ?))",
                    (item["id"], threshold_rank, threshold_rank, threshold_document),
                ).fetchone()["tokens"]
                selected = connection.execute(
                    "SELECT COUNT(*) AS documents, COALESCE(SUM(s.tokens_used), 0) AS tokens, "
                    "COALESCE(SUM(s.truncated), 0) AS truncated "
                    "FROM selected_documents s JOIN source_index.documents d "
                    "ON d.input_offset = s.input_offset WHERE d.stratum_id = ?",
                    (item["id"],),
                ).fetchone()
                outside_prefix = connection.execute(
                    "SELECT COUNT(*) AS count FROM selected_documents s "
                    "JOIN source_index.documents d ON d.input_offset = s.input_offset "
                    "WHERE d.stratum_id = ? AND NOT "
                    "(d.rank < ? OR (d.rank = ? AND d.document_id <= ?))",
                    (item["id"], threshold_rank, threshold_rank, threshold_document),
                ).fetchone()["count"]
                if outside_prefix:
                    raise ValueError(f"scaling selection escapes hash prefix for {item['id']}")
                invalid_full_rows = connection.execute(
                    "SELECT COUNT(*) AS count FROM selected_documents s "
                    "JOIN source_index.documents d ON d.input_offset = s.input_offset "
                    "WHERE d.stratum_id = ? AND NOT "
                    "(d.rank = ? AND d.document_id = ?) "
                    "AND (s.tokens_used != d.token_count OR s.truncated != 0)",
                    (item["id"], threshold_rank, threshold_document),
                ).fetchone()["count"]
                if invalid_full_rows:
                    raise ValueError(f"scaling hash prefix has partial rows for {item['id']}")
                if prefix["documents"] != selected["documents"]:
                    raise ValueError(f"scaling selection is not a hash prefix for {item['id']}")
                expected_threshold_take = item["target_tokens"] - before
                threshold = connection.execute(
                    "SELECT s.tokens_used, s.truncated, d.token_count "
                    "FROM selected_documents s JOIN source_index.documents d "
                    "ON d.input_offset = s.input_offset "
                    "WHERE d.stratum_id = ? AND d.rank = ? AND d.document_id = ?",
                    (item["id"], threshold_rank, threshold_document),
                ).fetchone()
                if threshold is None or threshold["tokens_used"] != expected_threshold_take:
                    raise ValueError(f"scaling threshold token count changed for {item['id']}")
                if expected_threshold_take > threshold["token_count"]:
                    raise ValueError(f"scaling threshold exceeds source tokens for {item['id']}")
                if selected["tokens"] != item["target_tokens"]:
                    raise ValueError(f"scaling source quota changed for {item['id']}")
                expected_truncated = int(expected_threshold_take < threshold["token_count"])
                if threshold["truncated"] != expected_truncated:
                    raise ValueError(f"scaling threshold truncation changed for {item['id']}")
                if selected["truncated"] != expected_truncated:
                    raise ValueError(f"scaling selection has extra truncation for {item['id']}")
    finally:
        connection.close()
    return manifest


def build_scaling_packed_dataset(
    *,
    index_manifest_path: str | Path,
    input_path: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
    target_tokens: int,
    sequence_length: int,
    sequences_per_shard: int = 256,
    checkpoint_documents: int = 256,
    checkpoint_text_bytes: int = 16 * 1024 * 1024,
    progress_documents: int = 10_000,
    purpose: str = "canary",
    resume: bool = False,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Select per-source hash prefixes and pack them with exact durable resume."""

    index_manifest_path = Path(index_manifest_path)
    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    output = Path(output)
    if purpose not in {"canary", "final"}:
        raise ValueError("scaling packed-data purpose must be canary or final")
    for value, name in (
        (target_tokens, "target-tokens"),
        (sequence_length, "sequence-length"),
        (sequences_per_shard, "sequences-per-shard"),
        (checkpoint_documents, "checkpoint-documents"),
        (checkpoint_text_bytes, "checkpoint-text-bytes"),
        (progress_documents, "progress-documents"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if sequence_length < 2 or target_tokens < sequence_length:
        raise ValueError("scaling packed target must contain at least one full sequence")
    effective_target = target_tokens // sequence_length * sequence_length
    index_manifest = validate_scaling_source_index(index_manifest_path)
    contract = index_manifest["contract"]
    if sequence_length != contract["selection"]["sequence_length"]:
        raise ValueError("packed sequence length differs from the source-index contract")
    if purpose == "final" and target_tokens != contract["selection"]["final_target_tokens"]:
        raise ValueError("final packed target differs from the source-index contract")
    if effective_target > contract["selection"]["final_target_tokens"]:
        raise ValueError("packed target exceeds the planned final target")

    input_artifact = _file_artifact(input_path)
    if _artifact_without_path(input_artifact) != contract["input"]:
        raise ValueError("packing input differs from the indexed source")
    tokenizer_artifact = _tokenizer_artifact(tokenizer_path)
    if tokenizer_artifact["files"] != contract["tokenizer"]["files"]:
        raise ValueError("packing tokenizer differs from the source index")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if len(tokenizer) != contract["tokenizer"]["vocab_size"]:
        raise ValueError("packing tokenizer vocabulary changed")
    if tokenizer.get_vocab().get("<|doc|>") != contract["tokenizer"]["doc_token_id"]:
        raise ValueError("packing tokenizer document token changed")

    allocations = _allocate_scaling_strata(index_manifest, target_tokens=effective_target)
    index_artifact = _file_artifact(index_manifest_path)
    index_reference = {
        "build_fingerprint": index_manifest["build_fingerprint"],
        **index_artifact,
    }
    build_specification = {
        "format": {
                "dtype": SCALING_TOKEN_DTYPE,
            "endianness": TOKEN_ENDIANNESS,
            "layout": "flat_fixed_sequences",
                "token_bytes": SCALING_TOKEN_BYTES,
        },
        "input": input_artifact,
        "packing": {
            "checkpoint_documents": checkpoint_documents,
            "max_documents": None,
            "purpose": purpose,
            "sequence_length": sequence_length,
            "sequences_per_shard": sequences_per_shard,
            "target_tokens_effective": effective_target,
            "target_tokens_requested": target_tokens,
            "text_key": "text",
        },
        "selection": {
            "allocation": ALLOCATION_MODE,
            "mode": SCALING_SELECTION_MODE,
            "seed": contract["selection"]["seed"],
            "source_index": index_reference,
            "strata": allocations,
        },
        "tokenizer": {
            **tokenizer_artifact,
            "doc_token": "<|doc|>",
            "doc_token_id": contract["tokenizer"]["doc_token_id"],
            "vocab_size": contract["tokenizer"]["vocab_size"],
        },
    }
    build_fingerprint = _fingerprint(
        _pack_fingerprint_specification(build_specification)
    )
    manifest_path = output / MANIFEST_FILENAME
    state_path = output / PACK_STATE_FILENAME
    proof_path = output / SELECTION_PROOF_FILENAME
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"scaling packed manifest already exists: {manifest_path}")
        cached = validate_scaling_packed_manifest(
            manifest_path, index_manifest_path=index_manifest_path
        )
        if cached["build_fingerprint"] != build_fingerprint:
            raise ValueError("cached scaling packed artifact does not match the request")
        return cached
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists() and not resume:
        raise FileExistsError(f"scaling packing state exists; pass --resume: {state_path}")
    if not resume:
        unexpected = list(output.iterdir())
        if unexpected:
            raise FileExistsError(f"scaling packed output is not empty: {output}")

    index_database = index_manifest_path.parent / index_manifest["database"]["name"]
    connection = _connect_pack_state(state_path)
    connection.execute("ATTACH DATABASE ? AS source_index", (str(index_database),))
    try:
        stored_fingerprint = _read_metadata(connection, "build_fingerprint")
        state = _read_metadata(connection, "state")
        if stored_fingerprint is None:
            state = {
                "completed_at": None,
                "documents_packed": 0,
                "pack_input_offset": -1,
                "selection_statistics": {},
                "status": "selecting",
                "stratum_documents_packed": {item["id"]: 0 for item in allocations},
                "stratum_tokens_packed": {item["id"]: 0 for item in allocations},
                "tokens_written": 0,
            }
            connection.execute("BEGIN IMMEDIATE")
            _write_metadata(connection, "build_fingerprint", build_fingerprint)
            _write_metadata(connection, "state", state)
            connection.commit()
        elif stored_fingerprint != build_fingerprint:
            raise ValueError("scaling packing checkpoint does not match the request")
        elif not isinstance(state, dict):
            raise ValueError("scaling packing checkpoint has no durable state")
        assert isinstance(state, dict)
        if state.get("status") not in {"selecting", "packing", "complete"}:
            raise ValueError("scaling packing checkpoint has an invalid status")

        if state["status"] == "selecting":
            for allocation in allocations:
                if allocation["id"] in state["selection_statistics"]:
                    continue
                remaining = allocation["target_tokens"]
                selected_documents = 0
                threshold = None
                connection.execute("BEGIN IMMEDIATE")
                try:
                    candidates = connection.execute(
                        "SELECT input_offset, document_id, rank, token_count "
                        "FROM source_index.documents WHERE heldout = 0 AND stratum_id = ? "
                        "ORDER BY rank, document_id",
                        (allocation["id"],),
                    )
                    try:
                        for candidate in candidates:
                            if remaining == 0:
                                break
                            tokens_used = min(candidate["token_count"], remaining)
                            truncated = tokens_used < candidate["token_count"]
                            connection.execute(
                                "INSERT INTO selected_documents VALUES(?, ?, ?)",
                                (
                                    candidate["input_offset"],
                                    tokens_used,
                                    int(truncated),
                                ),
                            )
                            remaining -= tokens_used
                            selected_documents += 1
                            threshold = {
                                "rank_threshold": candidate["rank"],
                                "threshold_document_id": candidate["document_id"],
                                "truncated_final_document": truncated,
                            }
                    finally:
                        candidates.close()
                    if remaining or threshold is None:
                        raise RuntimeError(
                            f"source index cannot satisfy target for {allocation['id']}: "
                            f"missing {remaining}"
                        )
                    state["selection_statistics"][allocation["id"]] = {
                        **allocation,
                        **threshold,
                        "documents_selected": selected_documents,
                        "tokens": allocation["target_tokens"],
                    }
                    _write_metadata(connection, "state", state)
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
            state["status"] = "packing"
            connection.execute("BEGIN IMMEDIATE")
            _write_metadata(connection, "state", state)
            connection.commit()

        shard_capacity = sequence_length * sequences_per_shard
        _reconcile_scaling_shards(
            output,
            committed_tokens=state["tokens_written"],
            shard_capacity=shard_capacity,
        )
        writer = _ScalingShardWriter(
            output,
            sequence_length=sequence_length,
            sequences_per_shard=sequences_per_shard,
            tokens_written=state["tokens_written"],
        )
        try:
            next_progress = (
                (state["documents_packed"] // progress_documents) + 1
            ) * progress_documents
            if state["status"] == "packing":
                with input_path.open("rb") as input_handle:
                    while True:
                        rows = connection.execute(
                            "SELECT d.input_offset, d.line_bytes, d.document_id, "
                            "d.content_sha256, d.near_duplicate_sha256, d.stratum_id, "
                            "d.token_count, s.tokens_used, s.truncated "
                            "FROM selected_documents s JOIN source_index.documents d "
                            "ON d.input_offset = s.input_offset "
                            "WHERE s.input_offset > ? ORDER BY s.input_offset LIMIT ?",
                            (state["pack_input_offset"], checkpoint_documents),
                        ).fetchall()
                        if not rows:
                            break
                        records = []
                        text_bytes = 0
                        for row in rows:
                            record = _read_source_record(input_handle, row)
                            records.append((row, record))
                            text_bytes += len(record["text"].encode("utf-8"))
                            if text_bytes >= checkpoint_text_bytes:
                                break
                            if (
                                _interrupt_after_documents is not None
                                and state["documents_packed"] + len(records)
                                >= _interrupt_after_documents
                            ):
                                break
                        encodings = tokenizer.backend_tokenizer.encode_batch(
                            [record["text"] for _, record in records],
                            add_special_tokens=False,
                        )
                        for (row, _), encoding in zip(records, encodings, strict=True):
                            token_ids = [contract["tokenizer"]["doc_token_id"], *encoding.ids]
                            if len(token_ids) != row["token_count"]:
                                raise RuntimeError("indexed token count changed during packing")
                            if row["tokens_used"] > len(token_ids):
                                raise RuntimeError(
                                    "selected token count exceeds the source document"
                                )
                            writer.append(token_ids[: row["tokens_used"]])
                            state["documents_packed"] += 1
                            state["tokens_written"] += row["tokens_used"]
                            state["pack_input_offset"] = row["input_offset"]
                            state["stratum_documents_packed"][row["stratum_id"]] += 1
                            state["stratum_tokens_packed"][row["stratum_id"]] += row[
                                "tokens_used"
                            ]
                        if writer.tokens_written != state["tokens_written"]:
                            raise RuntimeError("scaling shard writer and durable state diverged")
                        writer.flush()
                        connection.execute("BEGIN IMMEDIATE")
                        _write_metadata(connection, "state", state)
                        connection.commit()
                        if state["documents_packed"] >= next_progress:
                            print(
                                f"packed_documents={state['documents_packed']:,} "
                                f"tokens={state['tokens_written']:,}",
                                flush=True,
                            )
                            next_progress += progress_documents
                        if (
                            _interrupt_after_documents is not None
                            and state["documents_packed"] >= _interrupt_after_documents
                        ):
                            raise RuntimeError("simulated scaling packing interruption")
                if state["tokens_written"] != effective_target:
                    raise RuntimeError("scaling packed token total does not match the target")
                selected_total = connection.execute(
                    "SELECT COUNT(*) AS documents FROM selected_documents"
                ).fetchone()["documents"]
                if state["documents_packed"] != selected_total:
                    raise RuntimeError("not every selected scaling document was packed")
                for allocation in allocations:
                    identifier = allocation["id"]
                    if state["stratum_tokens_packed"][identifier] != allocation[
                        "target_tokens"
                    ]:
                        raise RuntimeError(f"packed source quota changed for {identifier}")
                state["completed_at"] = _utc_now()
                state["status"] = "complete"
                connection.execute("BEGIN IMMEDIATE")
                _write_metadata(connection, "state", state)
                connection.commit()
        finally:
            writer.flush_close()
        _finalize_sqlite(connection)
    finally:
        connection.close()

    ordered_selection = [
        state["selection_statistics"][item["id"]] for item in allocations
    ]
    proof = {
        "build_fingerprint": build_fingerprint,
        "heldout": {
            "document_id_overlap": 0,
            "exact_documents_excluded": index_manifest["statistics"][
                "heldout_exact_documents"
            ],
            "family_documents_excluded": index_manifest["statistics"][
                "heldout_documents"
            ],
            "near_duplicate_sha256_overlap": 0,
            "content_sha256_overlap": 0,
        },
        "schema_version": SCALING_DATA_PROOF_SCHEMA_VERSION,
        "selected_documents": state["documents_packed"],
        "selected_tokens": state["tokens_written"],
        "source_index_build_fingerprint": index_manifest["build_fingerprint"],
        "strata": ordered_selection,
    }
    _write_json_atomic(proof_path, proof)
    shards, sequences = _seal_scaling_shards(output, sequence_length=sequence_length)
    selection_state_artifact = _artifact_from_output(state_path)
    proof_artifact = _artifact_from_output(proof_path)
    statistics_strata = [
        {
            **item,
            "documents_packed": state["stratum_documents_packed"][item["id"]],
            "tokens": state["stratum_tokens_packed"][item["id"]],
        }
        for item in ordered_selection
    ]
    selection = {
        **build_specification["selection"],
        "strata": ordered_selection,
    }
    manifest = {
        **build_specification,
        "build_fingerprint": build_fingerprint,
        "completed_at": state["completed_at"],
        "end_reason": f"scaling_{purpose}_target",
        "leakage": proof["heldout"],
        "schema_version": PACKED_DATA_SCHEMA_VERSION,
        "selection": selection,
        "selection_proof": proof_artifact,
        "selection_state": selection_state_artifact,
        "shards": shards,
        "statistics": {
            "documents_packed": state["documents_packed"],
            "documents_selected": state["documents_packed"],
            "documents_seen": index_manifest["statistics"]["documents"],
            "sequences": sequences,
            "strata": statistics_strata,
            "tokens": state["tokens_written"],
        },
        "status": "complete",
    }
    _write_json_atomic(manifest_path, manifest)
    return validate_scaling_packed_manifest(
        manifest_path, index_manifest_path=index_manifest_path
    )
