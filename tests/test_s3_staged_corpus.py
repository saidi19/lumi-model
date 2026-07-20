from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lumi.production_corpus import _iter_gzip_records
from lumi.s3_staged_corpus import (
    bind_staged_sources,
    build_s3_staging_plan,
    build_staged_object,
    download_staged_source,
    stage_s3_staging_plan,
    stage_source_file,
    validate_s3_staging_plan,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "s3-staged-corpus-v0.1.json"


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeBody:
    def __init__(self, payload: bytes, *, fail_after: int | None = None) -> None:
        self.payload = payload
        self.fail_after = fail_after
        self.offset = 0
        self.closed = False

    def iter_chunks(self, *, chunk_size: int) -> Any:
        while self.offset < len(self.payload):
            if self.fail_after is not None and self.offset >= self.fail_after:
                raise OSError("synthetic interrupted S3 body")
            end = min(len(self.payload), self.offset + chunk_size)
            if self.fail_after is not None:
                end = min(end, self.fail_after)
            chunk = self.payload[self.offset : end]
            self.offset = end
            yield chunk

    def close(self) -> None:
        self.closed = True


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.get_ranges: list[str | None] = []
        self.fail_next_get_after: int | None = None
        self.put_calls = 0

    def put_object(self, **request: Any) -> dict[str, Any]:
        key = (request["Bucket"], request["Key"])
        if request.get("IfNoneMatch") == "*" and key in self.objects:
            raise FakeClientError("PreconditionFailed")
        payload = request["Body"].read()
        digest = hashlib.sha256(payload).digest()
        checksum = base64.b64encode(digest).decode("ascii")
        if checksum != request["ChecksumSHA256"]:
            raise FakeClientError("BadDigest")
        if len(payload) != request["ContentLength"]:
            raise FakeClientError("InvalidRequest")
        self.put_calls += 1
        self.objects[key] = {
            "body": payload,
            "head": {
                "ChecksumSHA256": checksum,
                "ContentLength": len(payload),
                "ContentType": request["ContentType"],
                "ETag": f'"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}"',
                "Metadata": dict(request["Metadata"]),
                "ServerSideEncryption": request["ServerSideEncryption"],
                "VersionId": "fixture-version-1",
            },
        }
        return {"ChecksumSHA256": checksum, "VersionId": "fixture-version-1"}

    def head_object(self, **request: Any) -> dict[str, Any]:
        key = (request["Bucket"], request["Key"])
        if key not in self.objects:
            raise FakeClientError("404")
        assert request["ChecksumMode"] == "ENABLED"
        return dict(self.objects[key]["head"])

    def get_object(self, **request: Any) -> dict[str, Any]:
        key = (request["Bucket"], request["Key"])
        if key not in self.objects:
            raise FakeClientError("NoSuchKey")
        assert request["ChecksumMode"] == "ENABLED"
        payload = self.objects[key]["body"]
        raw_range = request.get("Range")
        self.get_ranges.append(raw_range)
        response: dict[str, Any] = {}
        if raw_range is not None:
            start = int(raw_range.removeprefix("bytes=").removesuffix("-"))
            response["ContentRange"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            payload = payload[start:]
        fail_after = self.fail_next_get_after
        self.fail_next_get_after = None
        response["Body"] = FakeBody(payload, fail_after=fail_after)
        return response


def _payload() -> tuple[bytes, list[dict[str, Any]]]:
    records = [
        {"id": "doc-1", "text": "alpha"},
        {"id": "doc-2", "text": "beta"},
        {"id": "doc-3", "text": "gamma"},
    ]
    raw = b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in records
    )
    return gzip.compress(raw, compresslevel=9, mtime=0), records


def _source(payload: bytes) -> dict[str, Any]:
    return {
        "category": "science_and_education",
        "component": "arxiv_papers",
        "dataset_id": "common-pile/arxiv_papers_filtered",
        "ordinal": 7,
        "revision": "a" * 40,
        "source_bytes": len(payload),
        "source_file": "fixture.json.gz",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _entry(payload: bytes) -> dict[str, Any]:
    source = _source(payload)
    return {
        "object": build_staged_object(
            source,
            bucket="lumi-corpus-fixture",
            key_prefix="lumi-corpus/v0.1/raw/source-objects/sha256",
        ),
        "source": source,
    }


def test_local_plan_is_bounded_s3_only_and_non_authorizing() -> None:
    plan = build_s3_staging_plan(CONTRACT, repository_root=ROOT)
    verified = validate_s3_staging_plan(plan)
    assert verified == {
        "plan_fingerprint": plan["plan_fingerprint"],
        "source_bytes": 1_456_514_440,
        "source_files": 8,
        "status": "verified_local_no_cloud_mutation",
    }
    assert plan["consumer"] == {
        "direct_external_http_fallback": False,
        "network_path": "same_region_s3_only",
        "service": "aws_emr_serverless",
    }
    assert not any(plan["authorization"].values())
    assert plan["producer"] == {
        "compute_type": "BUILD_GENERAL1_SMALL",
        "disk_gb": 64,
        "direct_source_network_required": True,
        "launch_authorized": False,
        "maximum_build_minutes": 30,
        "maximum_parallel_downloads": 2,
        "memory_gib": 4,
        "service": "aws_codebuild_same_image_build",
        "source_files": 8,
        "vcpu": 2,
    }
    assert {entry["source"]["category"] for entry in plan["sources"]} == {
        "books",
        "code",
        "government_legal_and_patents",
        "open_web",
        "questions_and_discussion",
        "reference",
        "science_and_education",
    }
    serialized_consumer = json.dumps(
        [
            {
                **entry["source"],
                "staged_object": entry["object"],
                "start_line": entry["start_line"],
            }
            for entry in plan["sources"]
        ]
    )
    assert "https://" not in serialized_consumer
    assert "huggingface.co" not in serialized_consumer


def test_stage_reuse_and_s3_only_round_trip_preserve_exact_records(tmp_path: Path) -> None:
    payload, records = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()

    first = stage_source_file(entry, source_path, s3_client=client)
    assert first["status"] == "staged_new"
    assert first["sha256"] == entry["source"]["source_sha256"]
    second = stage_source_file(entry, source_path, s3_client=client)
    assert second["status"] == "reused_verified"
    assert client.put_calls == 1

    consumer_source = {**entry["source"], "staged_object": entry["object"]}
    downloaded = download_staged_source(
        consumer_source,
        tmp_path / "consumer",
        scratch_cap=10_000_000,
        s3_client=client,
        chunk_size=17,
    )
    assert downloaded.read_bytes() == payload
    observed = [row for _, row in _iter_gzip_records(downloaded)]
    assert observed == records


def test_bounded_codebuild_plan_reuses_verified_objects_without_source_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_s3_staging_plan(CONTRACT, repository_root=ROOT)
    client = FakeS3()
    for entry in plan["sources"]:
        source = entry["source"]
        staged = entry["object"]
        client.objects[(staged["bucket"], staged["key"])] = {
            "body": b"",
            "head": {
                "ChecksumSHA256": staged["checksum_sha256_base64"],
                "ContentLength": staged["bytes"],
                "ContentType": staged["content_type"],
                "ETag": f'"{staged["sha256"][:32]}"',
                "Metadata": {
                    "lumi-schema-version": "1",
                    "lumi-source-bytes": str(source["source_bytes"]),
                    "lumi-source-identity": staged["identity_fingerprint"],
                    "lumi-source-sha256": source["source_sha256"],
                },
                "ServerSideEncryption": staged["server_side_encryption"],
                "VersionId": f"fixture-{source['ordinal']}",
            },
        }

    def unexpected_download(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("verified staged objects must skip the upstream download")

    monkeypatch.setattr("lumi.s3_staged_corpus._download_source", unexpected_download)
    result = stage_s3_staging_plan(
        plan,
        tmp_path / "staging",
        scratch_cap=plan["summary"]["source_bytes"] + 128 * 1024 * 1024,
        s3_client=client,
        max_workers=2,
    )

    assert result["status"] == "verified_all_source_objects_staged"
    assert result["plan_fingerprint"] == plan["plan_fingerprint"]
    assert result["summary"] == {
        "reused_objects": 8,
        "source_bytes": 1_456_514_440,
        "source_files": 8,
        "staged_new_objects": 0,
    }
    assert [row["ordinal"] for row in result["results"]] == sorted(
        row["source"]["ordinal"] for row in plan["sources"]
    )


def test_s3_download_resumes_from_preserved_partial_range(tmp_path: Path) -> None:
    payload, _ = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()
    stage_source_file(entry, source_path, s3_client=client)
    consumer_source = {**entry["source"], "staged_object": entry["object"]}
    scratch = tmp_path / "consumer"
    client.fail_next_get_after = 32

    with pytest.raises(OSError, match="interrupted S3 body"):
        download_staged_source(
            consumer_source,
            scratch,
            scratch_cap=10_000_000,
            s3_client=client,
            chunk_size=32,
        )
    partial = scratch / f"source-{entry['source']['source_sha256']}.json.gz.part"
    assert partial.stat().st_size == 32

    downloaded = download_staged_source(
        consumer_source,
        scratch,
        scratch_cap=10_000_000,
        s3_client=client,
        chunk_size=32,
    )
    assert downloaded.read_bytes() == payload
    assert client.get_ranges == [None, "bytes=32-"]


def test_s3_download_seals_a_complete_verified_partial_without_an_empty_range(
    tmp_path: Path,
) -> None:
    payload, _ = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()
    stage_source_file(entry, source_path, s3_client=client)
    consumer_source = {**entry["source"], "staged_object": entry["object"]}
    scratch = tmp_path / "consumer"
    scratch.mkdir()
    partial = scratch / f"source-{entry['source']['source_sha256']}.json.gz.part"
    partial.write_bytes(payload)

    downloaded = download_staged_source(
        consumer_source,
        scratch,
        scratch_cap=10_000_000,
        s3_client=client,
        chunk_size=32,
    )

    assert downloaded.read_bytes() == payload
    assert not partial.exists()
    assert client.get_ranges == []


def test_s3_download_rejects_and_removes_a_corrupt_complete_partial(
    tmp_path: Path,
) -> None:
    payload, _ = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()
    stage_source_file(entry, source_path, s3_client=client)
    consumer_source = {**entry["source"], "staged_object": entry["object"]}
    scratch = tmp_path / "consumer"
    scratch.mkdir()
    partial = scratch / f"source-{entry['source']['source_sha256']}.json.gz.part"
    partial.write_bytes(b"x" * len(payload))

    with pytest.raises(ValueError, match="complete staged-source partial failed SHA-256"):
        download_staged_source(
            consumer_source,
            scratch,
            scratch_cap=10_000_000,
            s3_client=client,
            chunk_size=32,
        )

    assert not partial.exists()
    assert client.get_ranges == []


def test_consumer_fails_closed_without_s3_binding(tmp_path: Path) -> None:
    payload, _ = _payload()
    with pytest.raises(ValueError, match="missing its staged object binding"):
        download_staged_source(
            _source(payload),
            tmp_path,
            scratch_cap=10_000_000,
            s3_client=FakeS3(),
        )


def test_existing_wrong_object_is_never_overwritten(tmp_path: Path) -> None:
    payload, _ = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()
    staged = entry["object"]
    wrong = b"wrong-object"
    client.objects[(staged["bucket"], staged["key"])] = {
        "body": wrong,
        "head": {
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(wrong).digest()).decode(),
            "ContentLength": len(wrong),
            "ContentType": "application/gzip",
            "ETag": '"wrong"',
            "Metadata": {},
            "ServerSideEncryption": "AES256",
            "VersionId": "wrong-version",
        },
    }
    with pytest.raises(ValueError, match="byte size changed"):
        stage_source_file(entry, source_path, s3_client=client)
    assert client.put_calls == 0
    assert client.objects[(staged["bucket"], staged["key"])]["body"] == wrong


def test_existing_unversioned_object_is_rejected(tmp_path: Path) -> None:
    payload, _ = _payload()
    entry = _entry(payload)
    source_path = tmp_path / "fixture.json.gz"
    source_path.write_bytes(payload)
    client = FakeS3()
    stage_source_file(entry, source_path, s3_client=client)
    staged = entry["object"]
    client.objects[(staged["bucket"], staged["key"])]["head"]["VersionId"] = "null"

    with pytest.raises(ValueError, match="version ID is missing"):
        stage_source_file(entry, source_path, s3_client=client)

    assert client.put_calls == 1


def test_bind_consumer_sources_rejects_identity_or_plan_tampering() -> None:
    plan = build_s3_staging_plan(CONTRACT, repository_root=ROOT)
    sources = [
        {
            **entry["source"],
            "maximum_records": entry["maximum_records"],
            "source_provenance": entry["source_provenance"],
            "start_line": entry["start_line"],
        }
        for entry in plan["sources"]
    ]
    bound = bind_staged_sources(sources, plan)
    assert len(bound) == 8
    assert all(str(row["staged_object"]["s3_uri"]).startswith("s3://") for row in bound)

    tampered_sources = json.loads(json.dumps(sources))
    tampered_sources[0]["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        bind_staged_sources(tampered_sources, plan)

    tampered_plan = json.loads(json.dumps(plan))
    tampered_plan["sources"][0]["object"]["bytes"] += 1
    with pytest.raises(ValueError, match="fingerprint changed"):
        bind_staged_sources(sources, tampered_plan)
