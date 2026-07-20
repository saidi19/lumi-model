"""Bounded local shard assembly for accepted Lumi production EMR rows."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import tarfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .production_corpus import _compress_zstd, _write_provenance_parquet

SCHEMA_VERSION = 1


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


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"accepted row {label} is missing")
    return dict(value)


def _record(row: Mapping[str, Any], token_offset: int) -> dict[str, Any]:
    token_count = _positive_int(row.get("token_count"), "token count")
    token_bytes = row.get("token_bytes")
    if isinstance(token_bytes, bytearray):
        token_bytes = bytes(token_bytes)
    if not isinstance(token_bytes, bytes) or len(token_bytes) != token_count * 2:
        raise ValueError("accepted row token bytes are invalid")
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("accepted row text is missing")
    content_sha256 = _valid_sha256(row.get("content_sha256"), "content SHA-256")
    if hashlib.sha256(text.encode()).hexdigest() != content_sha256:
        raise ValueError("accepted row content SHA-256 changed")
    provenance = _json_mapping(row.get("provenance_json", row.get("provenance")), "provenance")
    quality = _json_mapping(row.get("quality_json", row.get("quality")), "quality")
    transformations = _json_mapping(
        row.get("transformations_json", row.get("transformations")), "transformations"
    )
    return {
        "content_sha256": content_sha256,
        "id": str(row["document_id"]),
        "provenance": {
            "category": str(row["category"]),
            "component": str(row["component"]),
            "dataset_id": str(row["dataset_id"]),
            "dataset_revision": str(row["dataset_revision"]),
            "retained": provenance,
            "segment_index": int(row["segment_index"]),
            "source_file": str(row["source_file"]),
            "source_line": int(row["source_line"]),
            "source_ordinal": int(row.get("source_ordinal", 0)),
            "source_sha256": _valid_sha256(row["source_sha256"], "source SHA-256"),
        },
        "quality": quality,
        "text": text,
        "token_count": token_count,
        "token_offset": token_offset,
        "transformations": transformations,
    }


@dataclass
class _OpenShard:
    category: str
    clean_path: Path
    documents: int
    sequence: int
    text_bytes: int
    token_path: Path
    tokens: int


def _new_open_shard(directory: Path, category: str, sequence: int) -> _OpenShard:
    clean = directory / f"{category}-{sequence:06d}.jsonl"
    tokens = directory / f"{category}-{sequence:06d}.uint16.bin"
    clean.touch()
    tokens.touch()
    return _OpenShard(category, clean, 0, sequence, 0, tokens, 0)


def _load_partial(directory: Path, partial: Mapping[str, Any]) -> _OpenShard:
    category = str(partial["category"])
    sequence = _positive_int(partial["sequence"], "partial sequence", allow_zero=True)
    clean_source = Path(str(partial["clean_path"]))
    token_source = Path(str(partial["token_path"]))
    for source in (clean_source, token_source):
        if not source.is_file():
            raise FileNotFoundError(source)
    clean = directory / f"{category}-{sequence:06d}.jsonl"
    tokens = directory / f"{category}-{sequence:06d}.uint16.bin"
    shutil.copyfile(clean_source, clean)
    shutil.copyfile(token_source, tokens)
    if _sha256_file(clean) != _valid_sha256(partial["clean_sha256"], "partial clean"):
        raise ValueError("partial clean shard SHA-256 changed")
    if _sha256_file(tokens) != _valid_sha256(partial["token_sha256"], "partial tokens"):
        raise ValueError("partial token shard SHA-256 changed")
    token_count = _positive_int(partial["tokens"], "partial tokens")
    if tokens.stat().st_size != token_count * 2:
        raise ValueError("partial token shard byte count changed")
    return _OpenShard(
        category=category,
        clean_path=clean,
        documents=_positive_int(partial["documents"], "partial documents"),
        sequence=sequence,
        text_bytes=_positive_int(partial["text_bytes"], "partial text bytes"),
        token_path=tokens,
        tokens=token_count,
    )


def _seal_local(
    shard: _OpenShard,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    clean_zst = shard.clean_path.with_suffix(".jsonl.zst")
    provenance = shard.clean_path.with_suffix(".provenance.parquet")
    _compress_zstd(shard.clean_path, clean_zst)
    source_files = _write_provenance_parquet(shard.clean_path, provenance)
    clean_sha = _sha256_file(clean_zst)
    token_sha = _sha256_file(shard.token_path)
    provenance_sha = _sha256_file(provenance)
    manifest = {
        "accepted_text_utf8_bytes": shard.text_bytes,
        "category": shard.category,
        "documents": shard.documents,
        "object_bytes": clean_zst.stat().st_size,
        "object_sha256": clean_sha,
        "provenance_index_bytes": provenance.stat().st_size,
        "provenance_index_sha256": provenance_sha,
        "removal_ledger_generation": 0,
        "schema_version": SCHEMA_VERSION,
        "sequence": shard.sequence,
        "shard_id": f"{shard.category}-{shard.sequence:06d}",
        "source_files": sorted(source_files),
        "source_manifest_sha256": _valid_sha256(
            source_manifest_sha256, "source manifest SHA-256"
        ),
        "token_dtype": "uint16_le",
        "token_object_bytes": shard.token_path.stat().st_size,
        "token_object_sha256": token_sha,
        "tokens": shard.tokens,
    }
    manifest_path = shard.clean_path.with_suffix(".manifest.json")
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    shard.clean_path.unlink()
    return {
        **manifest,
        "clean_path": str(clean_zst),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "provenance_path": str(provenance),
        "token_path": str(shard.token_path),
    }


def assemble_candidate_shards(
    rows: Iterable[Mapping[str, Any]],
    *,
    directory: Path,
    category: str,
    next_sequence: int,
    source_manifest_sha256: str,
    target_tokens: int = 100_000_000,
    partial: Mapping[str, Any] | None = None,
    seal_final_partial: bool = False,
    sealed_callback: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stream ordered accepted rows into full shards plus one resumable partial."""

    target_tokens = _positive_int(target_tokens, "target shard tokens")
    next_sequence = _positive_int(next_sequence, "next sequence", allow_zero=True)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("candidate shard directory is not empty")
    if partial is None:
        open_shard = _new_open_shard(directory, category, next_sequence)
    else:
        open_shard = _load_partial(directory, partial)
        if open_shard.category != category or open_shard.sequence != next_sequence:
            raise ValueError("partial shard does not match the requested category/sequence")
    sealed = []
    accepted_documents = 0
    accepted_tokens = 0
    previous_order: tuple[int, int, int, str] | None = None

    def record_sealed(shard: _OpenShard) -> None:
        local = _seal_local(
            shard,
            source_manifest_sha256=source_manifest_sha256,
        )
        if sealed_callback is None:
            sealed.append(local)
            return
        published = dict(sealed_callback(local))
        for name in ("clean_path", "manifest_path", "provenance_path", "token_path"):
            Path(str(local[name])).unlink(missing_ok=True)
        sealed.append(published)

    for row in rows:
        if str(row.get("category")) != category:
            raise ValueError("candidate shard crosses categories")
        order = (
            int(row.get("source_ordinal", 0)),
            int(row["source_line"]),
            int(row["segment_index"]),
            str(row["document_id"]),
        )
        if previous_order is not None and order <= previous_order:
            raise ValueError("accepted rows are not in strict source order")
        previous_order = order
        record = _record(row, open_shard.tokens)
        token_bytes = row["token_bytes"]
        if isinstance(token_bytes, bytearray):
            token_bytes = bytes(token_bytes)
        encoded = _canonical_json_bytes(record) + b"\n"
        with open_shard.clean_path.open("ab") as clean_handle:
            clean_handle.write(encoded)
        with open_shard.token_path.open("ab") as token_handle:
            token_handle.write(token_bytes)
        token_count = int(record["token_count"])
        open_shard.documents += 1
        open_shard.text_bytes += len(str(record["text"]).encode())
        open_shard.tokens += token_count
        accepted_documents += 1
        accepted_tokens += token_count
        if open_shard.tokens >= target_tokens:
            record_sealed(open_shard)
            next_sequence += 1
            open_shard = _new_open_shard(directory, category, next_sequence)
    for path in (open_shard.clean_path, open_shard.token_path):
        with path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    partial_result = None
    if open_shard.tokens and seal_final_partial:
        record_sealed(open_shard)
        next_sequence += 1
        open_shard = _new_open_shard(directory, category, next_sequence)
    if open_shard.tokens:
        partial_result = {
            "category": category,
            "clean_bytes": open_shard.clean_path.stat().st_size,
            "clean_path": str(open_shard.clean_path),
            "clean_sha256": _sha256_file(open_shard.clean_path),
            "documents": open_shard.documents,
            "sequence": open_shard.sequence,
            "text_bytes": open_shard.text_bytes,
            "token_bytes": open_shard.token_path.stat().st_size,
            "token_path": str(open_shard.token_path),
            "token_sha256": _sha256_file(open_shard.token_path),
            "tokens": open_shard.tokens,
        }
    else:
        open_shard.clean_path.unlink(missing_ok=True)
        open_shard.token_path.unlink(missing_ok=True)
    return {
        "accepted_documents": accepted_documents,
        "accepted_tokens": accepted_tokens,
        "next_sequence": next_sequence,
        "partial": partial_result,
        "sealed_shards": sealed,
    }


def _put_file(
    path: Path,
    *,
    bucket: str,
    key: str,
    content_type: str,
    metadata: Mapping[str, str],
    s3_client: Any,
) -> str | None:
    digest = _sha256_file(path)
    version_id = None
    try:
        with path.open("rb") as body:
            response = s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
                ContentType=content_type,
                IfNoneMatch="*",
                Metadata={**metadata, "sha256": digest},
                ServerSideEncryption="AES256",
            )
            if response.get("VersionId") is not None:
                version_id = str(response["VersionId"])
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != path.stat().st_size:
            raise ValueError("existing production object byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing production object SHA-256 metadata changed") from error
        if head.get("VersionId") is not None:
            version_id = str(head["VersionId"])
    return version_id


def _put_bytes(
    body: bytes,
    *,
    bucket: str,
    key: str,
    content_type: str,
    s3_client: Any,
) -> str:
    digest = _sha256_bytes(body)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            ContentType=content_type,
            IfNoneMatch="*",
            Metadata={"sha256": digest},
            ServerSideEncryption="AES256",
        )
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != len(body):
            raise ValueError("existing production manifest byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing production manifest SHA-256 changed") from error
    return digest


def publish_candidate_shard(
    shard: Mapping[str, Any],
    *,
    bucket: str,
    namespace: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Publish one locally sealed shard using immutable content-addressed keys."""

    category = str(shard["category"])
    clean_sha = _valid_sha256(shard["object_sha256"], "clean object")
    token_sha = _valid_sha256(shard["token_object_sha256"], "token object")
    provenance_sha = _valid_sha256(
        shard["provenance_index_sha256"], "provenance object"
    )
    manifest_sha = _valid_sha256(shard["manifest_sha256"], "manifest")
    keys = {
        "object_key": f"{namespace}/staging/clean/{category}/{clean_sha}.jsonl.zst",
        "token_object_key": f"{namespace}/staging/tokenized/{category}/{token_sha}.bin",
        "provenance_index_key": (
            f"{namespace}/control/provenance/shard-{clean_sha[:3]}/"
            f"{provenance_sha}.parquet"
        ),
        "manifest_object_key": f"{namespace}/control/manifests/{manifest_sha}.json",
    }
    _put_file(
        Path(str(shard["clean_path"])),
        bucket=bucket,
        key=keys["object_key"],
        content_type="application/zstd",
        metadata={"category": category},
        s3_client=s3_client,
    )
    _put_file(
        Path(str(shard["token_path"])),
        bucket=bucket,
        key=keys["token_object_key"],
        content_type="application/octet-stream",
        metadata={"category": category, "dtype": "uint16_le"},
        s3_client=s3_client,
    )
    _put_file(
        Path(str(shard["provenance_path"])),
        bucket=bucket,
        key=keys["provenance_index_key"],
        content_type="application/vnd.apache.parquet",
        metadata={"category": category},
        s3_client=s3_client,
    )
    local_manifest = {
        key: value
        for key, value in shard.items()
        if key not in {"clean_path", "manifest_path", "provenance_path", "token_path"}
    }
    publication = {
        "local_manifest_sha256": manifest_sha,
        "objects": {key: value for key, value in keys.items() if key != "manifest_object_key"},
        "schema_version": SCHEMA_VERSION,
        "shard": local_manifest,
    }
    body = _canonical_json_bytes(publication)
    publication_sha = _sha256_bytes(body)
    keys["manifest_object_key"] = (
        f"{namespace}/control/manifests/{publication_sha}.json"
    )
    _put_bytes(
        body,
        bucket=bucket,
        key=keys["manifest_object_key"],
        content_type="application/json",
        s3_client=s3_client,
    )
    return {**local_manifest, **keys, "manifest_sha256": publication_sha}


def publish_partial_shard(
    partial: Mapping[str, Any],
    *,
    bucket: str,
    namespace: str,
    checkpoint_parent_sha256: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Publish a resumable under-target shard snapshot between source jobs."""

    parent = _valid_sha256(checkpoint_parent_sha256, "parent checkpoint")
    clean = Path(str(partial["clean_path"]))
    tokens = Path(str(partial["token_path"]))
    clean_sha = _sha256_file(clean)
    token_sha = _sha256_file(tokens)
    base = f"{namespace}/control/distributed/production/partials/{parent}"
    clean_key = f"{base}/{clean_sha}.jsonl"
    token_key = f"{base}/{token_sha}.uint16.bin"
    clean_version_id = _put_file(
        clean,
        bucket=bucket,
        key=clean_key,
        content_type="application/x-ndjson",
        metadata={"category": str(partial["category"])},
        s3_client=s3_client,
    )
    token_version_id = _put_file(
        tokens,
        bucket=bucket,
        key=token_key,
        content_type="application/octet-stream",
        metadata={"category": str(partial["category"]), "dtype": "uint16_le"},
        s3_client=s3_client,
    )
    return {
        key: value
        for key, value in partial.items()
        if key not in {"clean_path", "token_path"}
    } | {
        "clean_s3_key": clean_key,
        "clean_version_id": clean_version_id,
        "token_s3_key": token_key,
        "token_version_id": token_version_id,
    }


def materialize_partial_shard(
    partial: Mapping[str, Any],
    *,
    directory: Path,
    bucket: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Download and verify either the legacy archive or an S3 partial snapshot."""

    directory.mkdir(parents=True, exist_ok=True)
    clean = directory / "partial.jsonl"
    tokens = directory / "partial.uint16.bin"
    state = partial.get("state")
    if state == "s3_snapshot":
        for key, path in (
            (partial.get("clean_s3_key"), clean),
            (partial.get("token_s3_key"), tokens),
        ):
            if not isinstance(key, str) or not key:
                raise ValueError("partial S3 snapshot key is missing")
            with path.open("wb") as handle:
                s3_client.download_fileobj(bucket, key, handle)
    elif state == "baseline_archive":
        archive_key = partial.get("archive_key")
        if not isinstance(archive_key, str) or not archive_key:
            raise ValueError("baseline partial archive key is missing")
        archive = directory / "baseline.tar.zst"
        with archive.open("wb") as handle:
            s3_client.download_fileobj(bucket, archive_key, handle)
        if archive.stat().st_size != int(partial["archive_bytes"]):
            raise ValueError("baseline partial archive byte size changed")
        if _sha256_file(archive) != _valid_sha256(
            partial["archive_sha256"], "partial archive"
        ):
            raise ValueError("baseline partial archive SHA-256 changed")
        import zstandard

        wanted = {
            f"./{partial['clean_member']}": clean,
            f"./{partial['token_member']}": tokens,
        }
        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                    for member in tar:
                        destination = wanted.get(member.name)
                        if destination is None:
                            continue
                        source = tar.extractfile(member)
                        if source is None:
                            raise ValueError("baseline partial archive member is not a file")
                        with destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
        archive.unlink()
    else:
        raise ValueError("unsupported partial-shard state")
    if clean.stat().st_size != int(partial["clean_bytes"]):
        raise ValueError("partial clean byte size changed")
    if tokens.stat().st_size != int(partial["token_bytes"]):
        raise ValueError("partial token byte size changed")
    if _sha256_file(clean) != _valid_sha256(partial["clean_sha256"], "partial clean"):
        raise ValueError("partial clean SHA-256 changed")
    if _sha256_file(tokens) != _valid_sha256(partial["token_sha256"], "partial tokens"):
        raise ValueError("partial token SHA-256 changed")
    return {
        **partial,
        "clean_path": str(clean),
        "token_path": str(tokens),
    }


def write_anchor_segment(
    rows: Iterable[Mapping[str, Any]],
    *,
    destination: Path,
) -> dict[str, Any]:
    """Write accepted rows as one strict-order gzip anchor delta."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    previous: tuple[int, int, int, str] | None = None
    documents = 0
    with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            order = (
                int(row["source_ordinal"]),
                int(row["source_line"]),
                int(row["segment_index"]),
                str(row["document_id"]),
            )
            if previous is not None and order <= previous:
                raise ValueError("anchor segment is not in strict source order")
            previous = order
            buckets = row["lsh_buckets"]
            payload = {
                "content_sha256": _valid_sha256(row["content_sha256"], "anchor content"),
                "document_id": str(row["document_id"]),
                "lsh_buckets": [
                    {
                        "band": int(bucket["band"]),
                        "bucket_hex": bytes(bucket["bucket"]).hex(),
                    }
                    for bucket in buckets
                ],
                "minhash_signature_hex": bytes(row["minhash_signature"]).hex(),
                "schema_version": 1,
                "segment_index": order[2],
                "source_line": order[1],
                "source_ordinal": order[0],
            }
            handle.write(_canonical_json_bytes(payload).decode() + "\n")
            documents += 1
    if documents == 0:
        raise ValueError("anchor segment cannot be empty")
    return {
        "bytes": destination.stat().st_size,
        "documents": documents,
        "object_sha256": _sha256_file(destination),
        "path": str(destination),
    }


def publish_anchor_segment(
    segment: Mapping[str, Any],
    *,
    bucket: str,
    namespace: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Publish an immutable anchor delta and its content-addressed manifest."""

    path = Path(str(segment["path"]))
    object_sha = _valid_sha256(segment["object_sha256"], "anchor object")
    object_key = (
        f"{namespace}/control/distributed/production/anchor-segments/"
        f"{object_sha}.jsonl.gz"
    )
    _put_file(
        path,
        bucket=bucket,
        key=object_key,
        content_type="application/gzip",
        metadata={"documents": str(segment["documents"])},
        s3_client=s3_client,
    )
    manifest = {
        "documents": int(segment["documents"]),
        "object_bytes": int(segment["bytes"]),
        "object_key": object_key,
        "object_sha256": object_sha,
        "schema_version": SCHEMA_VERSION,
    }
    body = _canonical_json_bytes(manifest)
    digest = _sha256_bytes(body)
    key = (
        f"{namespace}/control/distributed/production/anchor-manifests/{digest}.json"
    )
    _put_bytes(
        body,
        bucket=bucket,
        key=key,
        content_type="application/json",
        s3_client=s3_client,
    )
    return {**manifest, "manifest_key": key, "manifest_sha256": digest}
