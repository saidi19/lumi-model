"""Verify and export the immutable prefix for distributed corpus resume."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .distributed_corpus import (
    DEDUP_ANCHOR_SCHEMA_VERSION,
    MinHashCodec,
    validate_anchor,
)
from .production_corpus import _canonical_json_bytes, _read_json, _sha256_file

ANCHOR_FILE = "resume-anchors-v0.1.jsonl.gz"
REPORT_FILE = "report.json"
MANIFEST_FILE = "manifest.sha256"


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_bytes_atomic(path, rendered.encode("utf-8"))


def _safe_evidence_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("resume evidence manifest contains an invalid path")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("resume evidence manifest path escapes its directory")
    return path


def verify_evidence_directory(
    baseline_contract_path: Path,
    evidence_directory: Path,
) -> dict[str, Any]:
    """Verify every file from the already-hashed manual-stop evidence archive."""

    baseline = _read_json(baseline_contract_path)
    if baseline.get("gate") != "verified_single_writer_resume_baseline_v0.1":
        raise ValueError("unexpected distributed resume baseline gate")
    evidence_root = evidence_directory.resolve()
    manifest_path = evidence_root / "manifest.json"
    expected_manifest = baseline.get("archive", {}).get("internal_manifest_sha256")
    if _sha256_file(manifest_path) != expected_manifest:
        raise ValueError("resume evidence internal manifest changed")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, list) or len(files) != 8:
        raise ValueError("resume evidence internal manifest is invalid")
    verified = {}
    for record in files:
        path = _safe_evidence_path(evidence_root, record.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"missing resume evidence file: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"resume evidence byte size changed: {path.name}")
        digest = _sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"resume evidence SHA-256 changed: {path.name}")
        verified[str(record["path"])] = {
            "bytes": path.stat().st_size,
            "path": str(path),
            "sha256": digest,
        }
    manual = _read_json(evidence_root / "manual-stop-evidence.json")
    totals = baseline["totals"]
    total_accepted_tokens = manual.get("total_accepted_tokens")
    if total_accepted_tokens is None:
        total_accepted_tokens = sum(
            int(row["accepted_tokens"]) for row in manual.get("categories", [])
        )
    sealed_tokens = manual.get("sealed_tokens")
    if sealed_tokens is None:
        sealed_tokens = sum(int(row["tokens"]) for row in manual.get("sealed_shards", []))
    partial_tokens = manual.get("partial_tokens")
    if partial_tokens is None:
        partial_tokens = manual.get("current_shard", {}).get("tokens")
    if (
        total_accepted_tokens != totals["accepted_tokens"]
        or sealed_tokens != totals["sealed_tokens"]
        or partial_tokens != baseline["partial_prefix"]["accepted_tokens"]
    ):
        raise ValueError("resume evidence token totals changed")
    partial_clean = evidence_root / baseline["partial_prefix"]["evidence_member_clean"]
    partial_tokens = evidence_root / baseline["partial_prefix"]["evidence_member_tokens"]
    for path, size_name, digest_name in (
        (partial_clean, "clean_bytes", "clean_sha256"),
        (partial_tokens, "token_bytes", "token_sha256"),
    ):
        if path.stat().st_size != baseline["partial_prefix"][size_name]:
            raise ValueError(f"resume partial artifact byte size changed: {path.name}")
        if _sha256_file(path) != baseline["partial_prefix"][digest_name]:
            raise ValueError(f"resume partial artifact SHA-256 changed: {path.name}")
    return {
        "files": verified,
        "manual_stop": manual,
        "partial_clean": str(partial_clean),
        "partial_tokens": str(partial_tokens),
        "status": "verified",
    }


def _quality_fingerprint(quality: Mapping[str, Any], codec: MinHashCodec) -> str:
    near = quality["near_duplicate"]
    material = {
        "hashranges": codec.hashranges,
        "num_perm": codec.num_perm,
        "package_version": near["package_version"],
        "scheme": codec.scheme,
        "seed": near["seed"],
        "shingle_tokens": near["shingle_tokens"],
        "threshold": codec.threshold,
    }
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _anchor_json(anchor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "content_sha256": anchor["content_sha256"],
        "document_id": anchor["document_id"],
        "lsh_buckets": [
            {"band": row["band"], "bucket_hex": row["bucket"].hex()}
            for row in anchor["lsh_buckets"]
        ],
        "minhash_signature_hex": anchor["minhash_signature"].hex(),
        "schema_version": anchor["schema_version"],
        "segment_index": anchor["segment_index"],
        "source_line": anchor["source_line"],
        "source_ordinal": anchor["source_ordinal"],
    }


def anchor_from_json(row: Mapping[str, Any], codec: MinHashCodec) -> dict[str, Any]:
    try:
        anchor = {
            "content_sha256": str(row["content_sha256"]),
            "document_id": str(row["document_id"]),
            "lsh_buckets": [
                {"band": int(bucket["band"]), "bucket": bytes.fromhex(bucket["bucket_hex"])}
                for bucket in row["lsh_buckets"]
            ],
            "minhash_signature": bytes.fromhex(row["minhash_signature_hex"]),
            "schema_version": int(row["schema_version"]),
            "segment_index": int(row["segment_index"]),
            "source_line": int(row["source_line"]),
            "source_ordinal": int(row["source_ordinal"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid distributed resume anchor JSON") from error
    validate_anchor(anchor, codec)
    return anchor


def _verify_partial_shard(
    clean_path: Path,
    token_path: Path,
    expected: Mapping[str, Any],
    accepted_document_ids: set[str],
) -> dict[str, Any]:
    documents = 0
    tokens = 0
    next_offset = 0
    with clean_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                document_id = str(row["id"])
                text = str(row["text"])
                content_sha256 = str(row["content_sha256"])
                token_count = int(row["token_count"])
                token_offset = int(row["token_offset"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid partial clean record at line {line_number}") from error
            if document_id not in accepted_document_ids:
                raise ValueError("partial clean record is absent from accepted dedup state")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_sha256:
                raise ValueError("partial clean record content SHA-256 changed")
            if token_count < 0 or token_offset != next_offset:
                raise ValueError("partial clean token offsets are not contiguous")
            documents += 1
            tokens += token_count
            next_offset += token_count
    if documents != expected["accepted_documents"] or tokens != expected["accepted_tokens"]:
        raise ValueError("partial clean shard totals changed")
    if token_path.stat().st_size != tokens * 2:
        raise ValueError("partial uint16 token payload byte count changed")
    return {"documents": documents, "token_bytes": token_path.stat().st_size, "tokens": tokens}


def export_resume_anchors(
    *,
    baseline_contract_path: Path,
    quality_contract_path: Path,
    evidence_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Export a deterministic, compressed resume-anchor ledger from SQLite."""

    evidence = verify_evidence_directory(baseline_contract_path, evidence_directory)
    baseline = _read_json(baseline_contract_path)
    quality = _read_json(quality_contract_path)
    codec = MinHashCodec.from_config(quality["near_duplicate"])
    database = Path(evidence["files"]["acquisition-state.sqlite3"]["path"])
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("resume SQLite integrity check failed")
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM production_near_duplicate_metadata"
            ).fetchall()
        )
        if metadata.get("schema_version") != "1":
            raise ValueError("resume near-duplicate state schema changed")
        if metadata.get("fingerprint") != _quality_fingerprint(quality, codec):
            raise ValueError("resume near-duplicate state configuration changed")
        totals = baseline["totals"]
        exact_count = int(connection.execute("SELECT COUNT(*) FROM exact_content").fetchone()[0])
        signature_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM production_near_duplicate_signatures"
            ).fetchone()[0]
        )
        bucket_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM production_near_duplicate_buckets"
            ).fetchone()[0]
        )
        if exact_count != totals["accepted_documents"] or signature_count != exact_count:
            raise ValueError("resume accepted-document index counts changed")
        if bucket_count != exact_count * len(codec.hashranges):
            raise ValueError("resume LSH bucket count changed")
        shared_bucket_pairs = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT a.document_id, b.document_id "
                "FROM production_near_duplicate_buckets a "
                "JOIN production_near_duplicate_buckets b "
                "ON a.band = b.band AND a.bucket = b.bucket "
                "AND a.document_id < b.document_id)"
            ).fetchone()[0]
        )
        if shared_bucket_pairs != 0:
            raise ValueError("resume anchors unexpectedly contain near-duplicate candidates")

        stored_buckets: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
        for row in connection.execute(
            "SELECT document_id, band, bucket FROM production_near_duplicate_buckets "
            "ORDER BY document_id, band"
        ):
            stored_buckets[str(row["document_id"])].append(
                (int(row["band"]), bytes(row["bucket"]))
            )
        rows = connection.execute(
            "SELECT e.document_id, e.content_sha256, e.source_ordinal, e.source_line, "
            "e.segment_index, s.signature FROM exact_content e "
            "JOIN production_near_duplicate_signatures s ON s.document_id = e.document_id "
            "ORDER BY e.source_ordinal, e.source_line, e.segment_index, e.document_id"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        anchor_path = output_directory / ANCHOR_FILE
        temporary = anchor_path.with_name(f".{anchor_path.name}.{os.getpid()}.tmp")
        canonical_digest = hashlib.sha256()
        first_order = None
        last_order = None
        exported = 0
        accepted_ids = set()
        with temporary.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                for row in rows:
                    signature = bytes(row["signature"])
                    buckets = [
                        {
                            "band": band,
                            "bucket": signature[start * codec.width : end * codec.width],
                        }
                        for band, (start, end) in enumerate(codec.hashranges)
                    ]
                    expected_buckets = [
                        (bucket["band"], bucket["bucket"]) for bucket in buckets
                    ]
                    if stored_buckets[str(row["document_id"])] != expected_buckets:
                        raise ValueError("resume anchor LSH bucket payload changed")
                    anchor = {
                        "content_sha256": str(row["content_sha256"]),
                        "document_id": str(row["document_id"]),
                        "lsh_buckets": buckets,
                        "minhash_signature": signature,
                        "schema_version": DEDUP_ANCHOR_SCHEMA_VERSION,
                        "segment_index": int(row["segment_index"]),
                        "source_line": int(row["source_line"]),
                        "source_ordinal": int(row["source_ordinal"]),
                    }
                    validate_anchor(anchor, codec)
                    rendered = _canonical_json_bytes(_anchor_json(anchor)) + b"\n"
                    canonical_digest.update(rendered)
                    compressed.write(rendered)
                    order = (
                        anchor["source_ordinal"],
                        anchor["source_line"],
                        anchor["segment_index"],
                        anchor["document_id"],
                    )
                    first_order = order if first_order is None else first_order
                    last_order = order
                    accepted_ids.add(anchor["document_id"])
                    exported += 1
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        temporary.replace(anchor_path)
        if exported != totals["accepted_documents"]:
            raise ValueError("resume anchor export row count changed")
        partial = _verify_partial_shard(
            Path(evidence["partial_clean"]),
            Path(evidence["partial_tokens"]),
            baseline["partial_prefix"],
            accepted_ids,
        )
    finally:
        connection.close()

    report = {
        "anchor_file": {
            "bytes": anchor_path.stat().st_size,
            "canonical_rows_sha256": canonical_digest.hexdigest(),
            "path": ANCHOR_FILE,
            "sha256": _sha256_file(anchor_path),
        },
        "anchors": {
            "first_order": list(first_order) if first_order is not None else None,
            "last_order": list(last_order) if last_order is not None else None,
            "rows": exported,
            "shared_lsh_bucket_pairs": shared_bucket_pairs,
        },
        "baseline_contract_sha256": _sha256_file(baseline_contract_path),
        "codec": {
            "bands": len(codec.hashranges),
            "num_perm": codec.num_perm,
            "scheme": codec.scheme,
            "threshold": codec.threshold,
        },
        "gate": "distributed_resume_anchor_export_v0.1",
        "partial_prefix": partial,
        "schema_version": 1,
        "sqlite": {
            "bytes": database.stat().st_size,
            "integrity_check": "ok",
            "sha256": _sha256_file(database),
        },
        "status": "verified",
    }
    report_path = output_directory / REPORT_FILE
    _write_json_atomic(report_path, report)
    manifest = "".join(
        f"{_sha256_file(path)}  {path.name}\n"
        for path in (anchor_path, report_path)
    )
    _write_bytes_atomic(output_directory / MANIFEST_FILE, manifest.encode("utf-8"))
    return report


def verify_resume_anchor_export(
    output_directory: Path,
    quality_contract_path: Path,
) -> dict[str, Any]:
    """Independently verify the deterministic anchor export and its manifest."""

    report = _read_json(output_directory / REPORT_FILE)
    if report.get("gate") != "distributed_resume_anchor_export_v0.1":
        raise ValueError("unexpected resume anchor export gate")
    manifest_rows = (output_directory / MANIFEST_FILE).read_text(encoding="utf-8").splitlines()
    expected_names = {ANCHOR_FILE, REPORT_FILE}
    seen = set()
    for line in manifest_rows:
        digest, name = line.split("  ", 1)
        if name not in expected_names or name in seen:
            raise ValueError("resume anchor export manifest changed")
        if _sha256_file(output_directory / name) != digest:
            raise ValueError(f"resume anchor export hash mismatch: {name}")
        seen.add(name)
    if seen != expected_names:
        raise ValueError("resume anchor export manifest is incomplete")
    quality = _read_json(quality_contract_path)
    codec = MinHashCodec.from_config(quality["near_duplicate"])
    count = 0
    canonical_digest = hashlib.sha256()
    document_ids = set()
    content_hashes = set()
    previous_order = None
    with gzip.open(output_directory / ANCHOR_FILE, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid resume anchor line {line_number}") from error
            anchor = anchor_from_json(raw, codec)
            canonical_digest.update(_canonical_json_bytes(_anchor_json(anchor)) + b"\n")
            order = (
                anchor["source_ordinal"],
                anchor["source_line"],
                anchor["segment_index"],
                anchor["document_id"],
            )
            if previous_order is not None and order <= previous_order:
                raise ValueError("resume anchors are not strictly source ordered")
            if anchor["document_id"] in document_ids:
                raise ValueError("resume anchor document ID is duplicated")
            if anchor["content_sha256"] in content_hashes:
                raise ValueError("resume anchor exact content is duplicated")
            previous_order = order
            document_ids.add(anchor["document_id"])
            content_hashes.add(anchor["content_sha256"])
            count += 1
    if count != report["anchors"]["rows"]:
        raise ValueError("resume anchor export row count differs from its report")
    if canonical_digest.hexdigest() != report["anchor_file"]["canonical_rows_sha256"]:
        raise ValueError("resume anchor canonical row digest changed")
    if _sha256_file(output_directory / ANCHOR_FILE) != report["anchor_file"]["sha256"]:
        raise ValueError("resume anchor compressed digest changed")
    return {
        "anchor_file_sha256": report["anchor_file"]["sha256"],
        "anchors": count,
        "canonical_rows_sha256": canonical_digest.hexdigest(),
        "status": "verified",
    }
