"""Deterministic transient source splitting for Lumi production EMR jobs."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    if value.lower() != value:
        raise ValueError(f"{label} must be lowercase")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} is outside its allowed range")
    return value


def _aws_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return None if code is None else str(code)


def _verify_existing(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    expected_bytes: int,
    expected_sha256: str,
) -> str | None:
    head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if int(head.get("ContentLength", -1)) != expected_bytes:
        raise ValueError("existing staged object byte size changed")
    if head.get("Metadata", {}).get("sha256") != expected_sha256:
        raise ValueError("existing staged object SHA-256 metadata changed")
    version = head.get("VersionId")
    return None if version is None else str(version)


def _compress_zstd(source: Path, destination: Path) -> None:
    import zstandard

    compressor = zstandard.ZstdCompressor(level=3, threads=0)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        compressor.copy_stream(input_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def split_gzip_source(
    source_path: Path,
    output_directory: Path,
    *,
    source_file: str,
    source_sha256: str,
    start_after_line: int,
    target_uncompressed_bytes: int = 134_217_728,
    hard_transient_bytes: int = 100_000_000_000,
    hard_compressed_bytes: int | None = None,
    maximum_chunks: int | None = None,
) -> dict[str, Any]:
    """Split one pinned gzip JSONL file into independently readable zstd chunks.

    Chunk payloads keep the original JSONL bytes.  The manifest carries each
    chunk's inclusive source-line interval, so executors recover stable line
    numbers without reserializing records.  ``start_after_line`` is the last
    line already committed by the preceding immutable checkpoint.
    """

    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if Path(source_file).name != source_file or not source_file.endswith(".json.gz"):
        raise ValueError("production source filename is unsafe")
    expected_digest = _valid_sha256(source_sha256, "source SHA-256")
    if _sha256_file(source_path) != expected_digest:
        raise ValueError("production source SHA-256 changed")
    start_after_line = _positive_int(
        start_after_line, "start-after line", allow_zero=True
    )
    target_uncompressed_bytes = _positive_int(
        target_uncompressed_bytes, "target chunk bytes"
    )
    hard_transient_bytes = _positive_int(hard_transient_bytes, "transient byte cap")
    if target_uncompressed_bytes > hard_transient_bytes:
        raise ValueError("target chunk size exceeds the transient byte cap")
    if maximum_chunks is not None:
        maximum_chunks = _positive_int(maximum_chunks, "maximum chunks")
    if hard_compressed_bytes is not None:
        hard_compressed_bytes = _positive_int(
            hard_compressed_bytes, "compressed staging byte cap"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise FileExistsError("production staging output directory is not empty")
    chunks: list[dict[str, Any]] = []
    raw_path = output_directory / ".chunk.jsonl.part"
    raw_handle = None
    chunk_start = 0
    chunk_end = 0
    chunk_rows = 0
    chunk_bytes = 0
    total_uncompressed = 0
    source_complete = True

    def seal_chunk() -> None:
        nonlocal raw_handle, chunk_start, chunk_end, chunk_rows, chunk_bytes
        if raw_handle is None:
            return
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
        raw_handle.close()
        sequence = len(chunks)
        compressed = output_directory / (
            f"chunk-{sequence:06d}-{chunk_start:012d}-{chunk_end:012d}.jsonl.zst"
        )
        _compress_zstd(raw_path, compressed)
        if hard_compressed_bytes is not None and (
            sum(int(row["bytes"]) for row in chunks) + compressed.stat().st_size
            > hard_compressed_bytes
        ):
            compressed.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
            raise RuntimeError("production staged source exceeds the local compressed-byte cap")
        chunks.append(
            {
                "bytes": compressed.stat().st_size,
                "end_line": chunk_end,
                "file": compressed.name,
                "rows": chunk_rows,
                "sequence": sequence,
                "sha256": _sha256_file(compressed),
                "start_line": chunk_start,
                "uncompressed_bytes": chunk_bytes,
            }
        )
        raw_path.unlink()
        raw_handle = None
        chunk_start = 0
        chunk_end = 0
        chunk_rows = 0
        chunk_bytes = 0

    try:
        with gzip.open(source_path, "rb") as source:
            for line_number, line in enumerate(source, start=1):
                if line_number <= start_after_line:
                    continue
                if (
                    maximum_chunks is not None
                    and raw_handle is None
                    and len(chunks) >= maximum_chunks
                ):
                    source_complete = False
                    break
                if raw_handle is None:
                    raw_handle = raw_path.open("wb")
                    chunk_start = line_number
                raw_handle.write(line)
                chunk_end = line_number
                chunk_rows += 1
                chunk_bytes += len(line)
                total_uncompressed += len(line)
                if total_uncompressed > hard_transient_bytes:
                    raise RuntimeError("production staged source exceeds the transient byte cap")
                if chunk_bytes >= target_uncompressed_bytes:
                    seal_chunk()
        seal_chunk()
    except BaseException:
        if raw_handle is not None and not raw_handle.closed:
            raw_handle.close()
        raw_path.unlink(missing_ok=True)
        for chunk in chunks:
            (output_directory / chunk["file"]).unlink(missing_ok=True)
        raise
    if not chunks:
        raise ValueError("production source contains no uncommitted records")
    manifest = {
        "chunks": chunks,
        "gate": "lumi_emr_transient_source_chunks_v0.1",
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "source_complete": source_complete,
        "source_sha256": expected_digest,
        "start_after_line": start_after_line,
        "total_compressed_bytes": sum(int(row["bytes"]) for row in chunks),
        "total_rows": sum(int(row["rows"]) for row in chunks),
        "total_uncompressed_bytes": total_uncompressed,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    validate_staged_source_manifest(manifest, hard_transient_bytes=hard_transient_bytes)
    return manifest


def validate_staged_source_manifest(
    manifest: Mapping[str, Any], *, hard_transient_bytes: int = 100_000_000_000
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported staged-source manifest schema")
    if manifest.get("gate") != "lumi_emr_transient_source_chunks_v0.1":
        raise ValueError("unexpected staged-source manifest gate")
    _valid_sha256(manifest.get("source_sha256"), "source SHA-256")
    if Path(str(manifest.get("source_file"))).name != manifest.get("source_file"):
        raise ValueError("staged-source filename is unsafe")
    if not isinstance(manifest.get("source_complete"), bool):
        raise ValueError("staged-source completion state is invalid")
    start_after = _positive_int(
        manifest.get("start_after_line"), "start-after line", allow_zero=True
    )
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("staged-source manifest has no chunks")
    expected_line = start_after + 1
    total_rows = 0
    total_uncompressed = 0
    total_compressed = 0
    seen_digests: set[str] = set()
    for sequence, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping) or chunk.get("sequence") != sequence:
            raise ValueError("staged-source chunk sequence changed")
        start = _positive_int(chunk.get("start_line"), "chunk start line")
        end = _positive_int(chunk.get("end_line"), "chunk end line")
        rows = _positive_int(chunk.get("rows"), "chunk rows")
        if start != expected_line or end < start or end - start + 1 != rows:
            raise ValueError("staged-source chunk line interval is not contiguous")
        expected_line = end + 1
        digest = _valid_sha256(chunk.get("sha256"), "chunk SHA-256")
        if digest in seen_digests:
            raise ValueError("staged-source manifest repeats a chunk")
        seen_digests.add(digest)
        if Path(str(chunk.get("file"))).name != chunk.get("file"):
            raise ValueError("staged-source chunk filename is unsafe")
        total_rows += rows
        total_uncompressed += _positive_int(
            chunk.get("uncompressed_bytes"), "chunk uncompressed bytes"
        )
        total_compressed += _positive_int(chunk.get("bytes"), "chunk compressed bytes")
    if total_rows != manifest.get("total_rows"):
        raise ValueError("staged-source row total changed")
    if total_uncompressed != manifest.get("total_uncompressed_bytes"):
        raise ValueError("staged-source uncompressed-byte total changed")
    if total_compressed != manifest.get("total_compressed_bytes"):
        raise ValueError("staged-source compressed-byte total changed")
    if total_uncompressed > hard_transient_bytes:
        raise ValueError("staged-source manifest exceeds the transient byte cap")
    material = dict(manifest)
    observed = material.pop("manifest_sha256", None)
    if _valid_sha256(observed, "manifest SHA-256") != _sha256_bytes(
        _canonical_json_bytes(material)
    ):
        raise ValueError("staged-source manifest SHA-256 changed")
    return dict(manifest)


def publish_staged_source(
    manifest: Mapping[str, Any],
    *,
    directory: Path,
    bucket: str,
    key_prefix: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Conditionally publish chunks and then their content-addressed manifest."""

    validated = validate_staged_source_manifest(manifest)
    published = []
    for chunk in validated["chunks"]:
        path = directory / chunk["file"]
        if not path.is_file() or path.stat().st_size != chunk["bytes"]:
            raise FileNotFoundError(path)
        if _sha256_file(path) != chunk["sha256"]:
            raise ValueError("staged-source chunk changed before upload")
        key = f"{key_prefix}/chunks/{validated['source_sha256']}/{chunk['sha256']}.jsonl.zst"
        checksum = base64.b64encode(bytes.fromhex(chunk["sha256"])).decode()
        version_id = None
        try:
            with path.open("rb") as body:
                response = s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=body,
                    ChecksumSHA256=checksum,
                    ContentType="application/zstd",
                    IfNoneMatch="*",
                    Metadata={
                        "end-line": str(chunk["end_line"]),
                        "sha256": chunk["sha256"],
                        "source-sha256": validated["source_sha256"],
                        "start-line": str(chunk["start_line"]),
                    },
                    ServerSideEncryption="AES256",
                )
                if response.get("VersionId") is not None:
                    version_id = str(response["VersionId"])
        except BaseException as error:
            if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
                raise
            version_id = _verify_existing(
                s3_client,
                bucket=bucket,
                key=key,
                expected_bytes=path.stat().st_size,
                expected_sha256=chunk["sha256"],
            )
        published.append({**chunk, "s3_key": key, "version_id": version_id})
    remote_manifest = {**validated, "chunks": published}
    remote_manifest.pop("manifest_sha256", None)
    body = _canonical_json_bytes(remote_manifest)
    digest = _sha256_bytes(body)
    key = f"{key_prefix}/manifests/{validated['source_sha256']}/{digest}.json"
    manifest_version_id = None
    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={"sha256": digest, "source-sha256": validated["source_sha256"]},
            ServerSideEncryption="AES256",
        )
        if response.get("VersionId") is not None:
            manifest_version_id = str(response["VersionId"])
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        manifest_version_id = _verify_existing(
            s3_client,
            bucket=bucket,
            key=key,
            expected_bytes=len(body),
            expected_sha256=digest,
        )
    return {
        **remote_manifest,
        "manifest_sha256": digest,
        "s3_key": key,
        "version_id": manifest_version_id,
    }


def validate_published_staged_source(
    manifest: Mapping[str, Any],
    *,
    bucket: str,
    key_prefix: str,
    hard_transient_bytes: int = 100_000_000_000,
) -> dict[str, Any]:
    """Validate a remotely published chunk manifest and every bounded S3 key."""

    remote = dict(manifest)
    observed = _valid_sha256(
        remote.pop("manifest_sha256", None), "published manifest SHA-256"
    )
    manifest_key = remote.pop("s3_key", None)
    remote.pop("version_id", None)
    if observed != _sha256_bytes(_canonical_json_bytes(remote)):
        raise ValueError("published staged-source manifest SHA-256 changed")
    if (
        not isinstance(manifest_key, str)
        or not manifest_key.startswith(f"{key_prefix}/manifests/")
        or ".." in manifest_key
    ):
        raise ValueError("published staged-source manifest key is unsafe")
    chunks = remote.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("published staged-source chunks are missing")
    local_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError("published staged-source chunk is invalid")
        key = chunk.get("s3_key")
        if (
            not isinstance(key, str)
            or not key.startswith(f"{key_prefix}/chunks/")
            or ".." in key
        ):
            raise ValueError("published staged-source chunk key is unsafe")
        value = dict(chunk)
        value.pop("s3_key", None)
        value.pop("version_id", None)
        local_chunks.append(value)
    local = {**remote, "chunks": local_chunks}
    local["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(local))
    validate_staged_source_manifest(
        local, hard_transient_bytes=hard_transient_bytes
    )
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("published staged-source bucket is invalid")
    return dict(manifest)


def clean_local_staging(directory: Path) -> None:
    """Remove only a dedicated local transient staging directory."""

    if directory.is_dir():
        shutil.rmtree(directory)
