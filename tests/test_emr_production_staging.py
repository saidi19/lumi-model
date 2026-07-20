from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from lumi.emr_production_staging import (
    publish_staged_source,
    split_gzip_source,
    validate_published_staged_source,
    validate_staged_source_manifest,
)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **request: object) -> dict[str, str]:
        key = (str(request["Bucket"]), str(request["Key"]))
        if request.get("IfNoneMatch") == "*" and key in self.objects:
            raise RuntimeError("precondition failed")
        body = request["Body"]
        if hasattr(body, "read"):
            payload = body.read()
        else:
            payload = bytes(body)
        self.objects[key] = payload
        return {"VersionId": f"version-{len(self.objects)}"}


def _source(tmp_path: Path, rows: int = 12) -> tuple[Path, list[bytes]]:
    payloads = [
        json.dumps({"text": f"document {index}", "id": index}).encode() + b"\n"
        for index in range(1, rows + 1)
    ]
    path = tmp_path / "fixture.json.gz"
    with gzip.open(path, "wb") as handle:
        for row in payloads:
            handle.write(row)
    return path, payloads


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_is_contiguous_resumable_and_deterministic(tmp_path: Path) -> None:
    source, payloads = _source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = split_gzip_source(
        source,
        first,
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=3,
        target_uncompressed_bytes=len(payloads[3]) + 1,
    )
    repeated = split_gzip_source(
        source,
        second,
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=3,
        target_uncompressed_bytes=len(payloads[3]) + 1,
    )
    assert manifest == repeated
    assert manifest["total_rows"] == 9
    assert manifest["chunks"][0]["start_line"] == 4
    assert manifest["chunks"][-1]["end_line"] == 12
    assert validate_staged_source_manifest(manifest) == manifest


def test_split_can_checkpoint_a_bounded_source_slice(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    first = split_gzip_source(
        source,
        tmp_path / "first-slice",
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=0,
        target_uncompressed_bytes=1,
        maximum_chunks=2,
    )
    assert first["source_complete"] is False
    assert len(first["chunks"]) == 2
    assert first["chunks"][-1]["end_line"] == 2
    resumed = split_gzip_source(
        source,
        tmp_path / "second-slice",
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=2,
        target_uncompressed_bytes=1,
        maximum_chunks=20,
    )
    assert resumed["source_complete"] is True
    assert resumed["chunks"][0]["start_line"] == 3
    assert resumed["chunks"][-1]["end_line"] == 12


def test_split_rejects_wrong_source_hash_and_transient_cap(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 changed"):
        split_gzip_source(
            source,
            tmp_path / "wrong",
            source_file=source.name,
            source_sha256="0" * 64,
            start_after_line=0,
        )
    with pytest.raises(RuntimeError, match="transient byte cap"):
        split_gzip_source(
            source,
            tmp_path / "cap",
            source_file=source.name,
            source_sha256=_sha256(source),
            start_after_line=0,
            target_uncompressed_bytes=16,
            hard_transient_bytes=32,
        )
    with pytest.raises(RuntimeError, match="compressed-byte cap"):
        split_gzip_source(
            source,
            tmp_path / "compressed-cap",
            source_file=source.name,
            source_sha256=_sha256(source),
            start_after_line=0,
            target_uncompressed_bytes=16,
            hard_compressed_bytes=1,
        )


def test_manifest_rejects_noncontiguous_lines_and_hash_drift(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    manifest = split_gzip_source(
        source,
        tmp_path / "chunks",
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=0,
        target_uncompressed_bytes=64,
    )
    broken = json.loads(json.dumps(manifest))
    broken["chunks"][0]["start_line"] = 2
    with pytest.raises(ValueError, match="not contiguous"):
        validate_staged_source_manifest(broken)
    broken = json.loads(json.dumps(manifest))
    broken["total_rows"] += 1
    with pytest.raises(ValueError, match="row total"):
        validate_staged_source_manifest(broken)


def test_publish_uses_content_addressed_chunk_and_manifest_keys(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    directory = tmp_path / "chunks"
    manifest = split_gzip_source(
        source,
        directory,
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=2,
        target_uncompressed_bytes=64,
    )
    client = FakeS3()
    remote = publish_staged_source(
        manifest,
        directory=directory,
        bucket="fixture-bucket",
        key_prefix="lumi-corpus/v0.1/raw/transient",
        s3_client=client,
    )
    assert len(client.objects) == len(manifest["chunks"]) + 1
    assert remote["s3_key"].endswith(f"/{remote['manifest_sha256']}.json")
    assert remote["version_id"]
    for chunk in remote["chunks"]:
        assert chunk["s3_key"].endswith(f"/{chunk['sha256']}.jsonl.zst")
        assert chunk["version_id"]
    assert validate_published_staged_source(
        remote,
        bucket="fixture-bucket",
        key_prefix="lumi-corpus/v0.1/raw/transient",
    ) == remote
    changed = json.loads(json.dumps(remote))
    changed["chunks"][0]["s3_key"] = "outside/chunk.zst"
    material = dict(changed)
    material.pop("manifest_sha256")
    material.pop("s3_key")
    material.pop("version_id")
    changed["manifest_sha256"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="chunk key"):
        validate_published_staged_source(
            changed,
            bucket="fixture-bucket",
            key_prefix="lumi-corpus/v0.1/raw/transient",
        )


def test_publish_refuses_changed_chunk(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    directory = tmp_path / "chunks"
    manifest = split_gzip_source(
        source,
        directory,
        source_file=source.name,
        source_sha256=_sha256(source),
        start_after_line=0,
        target_uncompressed_bytes=64,
    )
    (directory / manifest["chunks"][0]["file"]).write_bytes(b"changed")
    with pytest.raises((FileNotFoundError, ValueError)):
        publish_staged_source(
            manifest,
            directory=directory,
            bucket="fixture-bucket",
            key_prefix="lumi-corpus/v0.1/raw/transient",
            s3_client=FakeS3(),
        )
