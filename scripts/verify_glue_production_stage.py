#!/usr/bin/env python3
"""Independently verify every object published by Lumi Glue staging."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.emr_production_staging import validate_published_staged_source
from lumi.glue_production import (
    validate_glue_production_stage_authorization,
    validate_glue_production_stage_release,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
        raise ValueError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    if value.lower() != value:
        raise ValueError(f"{label} is not lowercase")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_stage_result(
    result: Mapping[str, Any],
    release: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sealed driver result against its release and authorization."""

    validate_glue_production_stage_release(release)
    issued = datetime.fromisoformat(str(authorization["issued_at"])).astimezone(UTC)
    validate_glue_production_stage_authorization(
        authorization, release, now=issued
    )
    if result.get("schema_version") != 1:
        raise ValueError("unsupported Glue production stage result schema")
    if result.get("gate") != "lumi_glue_production_stage_result_v0.1":
        raise ValueError("unexpected Glue production stage result gate")
    if result.get("status") != "pass" or result.get("training_started") is not False:
        raise ValueError("Glue production stage did not pass cleanly")
    if result.get("release_fingerprint") != release["release_fingerprint"]:
        raise ValueError("Glue production stage result targets another release")
    if result.get("authorization_sha256") != authorization["authorization_sha256"]:
        raise ValueError("Glue production stage result targets another authorization")
    if result.get("source_commit") != release["source_commit"]:
        raise ValueError("Glue production stage source commit changed")
    job_run_id = result.get("job_run_id")
    if not isinstance(job_run_id, str) or not job_run_id.startswith("jr_"):
        raise ValueError("Glue production stage job-run id is invalid")
    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
        raise ValueError("Glue production stage elapsed time is invalid")
    if elapsed > int(release["job"]["timeout_minutes"]) * 60:
        raise ValueError("Glue production stage exceeded its runtime cap")

    rows = result.get("sources")
    if not isinstance(rows, list) or len(rows) != len(release["sources"]):
        raise ValueError("Glue production stage source result count changed")
    planned = {int(row["ordinal"]): row for row in release["sources"]}
    if len(planned) != len(release["sources"]):
        raise ValueError("Glue production release repeats a source ordinal")
    prefix = f"{release['output']['staged_prefix']}/{release['release_fingerprint']}"
    manifest_keys: set[str] = set()
    totals = {"chunks": 0, "compressed": 0, "rows": 0, "uncompressed": 0}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Glue production stage source result is invalid")
        ordinal = row.get("source_ordinal")
        if not isinstance(ordinal, int) or ordinal not in planned:
            raise ValueError("Glue production stage source ordinal changed")
        source = planned.pop(ordinal)
        for result_key, source_key in (
            ("category", "category"),
            ("component", "component"),
            ("source_bytes", "source_bytes"),
            ("source_file", "source_file"),
            ("source_sha256", "source_sha256"),
            ("start_after_line", "start_after_line"),
        ):
            if row.get(result_key) != source[source_key]:
                raise ValueError(f"Glue production stage {result_key} changed")
        if row.get("source_complete") is not True or not isinstance(
            row.get("reused"), bool
        ):
            raise ValueError("Glue production stage source completion changed")
        manifest_sha = _valid_sha256(
            row.get("manifest_sha256"), "staged manifest SHA-256"
        )
        manifest_key = row.get("manifest_s3_key")
        expected_manifest_prefix = f"{prefix}/manifests/{source['source_sha256']}/"
        if (
            not isinstance(manifest_key, str)
            or not manifest_key.startswith(expected_manifest_prefix)
            or not manifest_key.endswith(f"/{manifest_sha}.json")
            or manifest_key in manifest_keys
        ):
            raise ValueError("Glue production stage manifest key changed")
        manifest_keys.add(manifest_key)
        if not isinstance(row.get("manifest_version_id"), str):
            raise ValueError("Glue production stage manifest version is missing")
        totals["chunks"] += _positive_int(row.get("chunk_count"), "chunk count")
        totals["compressed"] += _positive_int(
            row.get("staged_compressed_bytes"), "staged compressed bytes"
        )
        totals["rows"] += _positive_int(row.get("staged_rows"), "staged rows")
        totals["uncompressed"] += _positive_int(
            row.get("staged_uncompressed_bytes"), "staged uncompressed bytes"
        )
    if planned:
        raise ValueError("Glue production stage omitted planned sources")
    expected = {
        "chunks": result.get("staged_chunks"),
        "compressed": result.get("staged_compressed_bytes"),
        "rows": result.get("staged_rows"),
        "uncompressed": result.get("staged_uncompressed_bytes"),
    }
    if totals != expected:
        raise ValueError("Glue production stage aggregate totals changed")
    if result.get("source_files") != len(rows):
        raise ValueError("Glue production stage source-file total changed")
    if totals["uncompressed"] > int(release["limits"]["hard_logical_storage_bytes"]):
        raise ValueError("Glue production stage exceeded its logical storage cap")
    return {
        "elapsed_seconds": elapsed,
        "job_run_id": job_run_id,
        "source_files": len(rows),
        "staged_chunks": totals["chunks"],
        "staged_compressed_bytes": totals["compressed"],
        "staged_rows": totals["rows"],
        "staged_uncompressed_bytes": totals["uncompressed"],
        "status": "driver_result_verified",
    }


def _verify_response_integrity(
    response: Mapping[str, Any],
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_version_id: str,
    label: str,
) -> None:
    if int(response.get("ContentLength", -1)) != expected_bytes:
        raise ValueError(f"{label} byte size changed")
    if response.get("VersionId") != expected_version_id:
        raise ValueError(f"{label} version changed")
    if response.get("Metadata", {}).get("sha256") != expected_sha256:
        raise ValueError(f"{label} metadata SHA-256 changed")
    expected_checksum = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
    if response.get("ChecksumSHA256") != expected_checksum:
        raise ValueError(f"{label} S3 checksum changed")
    if response.get("ServerSideEncryption") != "AES256":
        raise ValueError(f"{label} encryption changed")


def _verify_source(
    pair: tuple[Mapping[str, Any], Mapping[str, Any]],
    *,
    bucket: str,
    key_prefix: str,
    s3_client: Any,
) -> dict[str, int]:
    row, source = pair
    manifest_response = s3_client.get_object(
        Bucket=bucket,
        Key=row["manifest_s3_key"],
        VersionId=row["manifest_version_id"],
        ChecksumMode="ENABLED",
    )
    body = manifest_response["Body"].read()
    digest = _sha256_bytes(body)
    if digest != row["manifest_sha256"]:
        raise ValueError("staged manifest body SHA-256 changed")
    _verify_response_integrity(
        manifest_response,
        expected_bytes=len(body),
        expected_sha256=digest,
        expected_version_id=row["manifest_version_id"],
        label="staged manifest",
    )
    manifest = json.loads(body)
    if not isinstance(manifest, dict):
        raise ValueError("staged manifest is not a JSON object")
    published = {
        **manifest,
        "manifest_sha256": digest,
        "s3_key": row["manifest_s3_key"],
        "version_id": row["manifest_version_id"],
    }
    validate_published_staged_source(
        published, bucket=bucket, key_prefix=key_prefix
    )
    for manifest_key, result_key in (
        ("source_file", "source_file"),
        ("source_sha256", "source_sha256"),
        ("source_complete", "source_complete"),
        ("start_after_line", "start_after_line"),
        ("total_compressed_bytes", "staged_compressed_bytes"),
        ("total_rows", "staged_rows"),
        ("total_uncompressed_bytes", "staged_uncompressed_bytes"),
    ):
        if manifest.get(manifest_key) != row[result_key]:
            raise ValueError(f"staged manifest {manifest_key} changed")
    if manifest.get("source_sha256") != source["source_sha256"]:
        raise ValueError("staged manifest targets another planned source")
    if len(manifest["chunks"]) != row["chunk_count"]:
        raise ValueError("staged manifest chunk count changed")
    for chunk in manifest["chunks"]:
        head = s3_client.head_object(
            Bucket=bucket,
            Key=chunk["s3_key"],
            VersionId=chunk["version_id"],
            ChecksumMode="ENABLED",
        )
        _verify_response_integrity(
            head,
            expected_bytes=int(chunk["bytes"]),
            expected_sha256=chunk["sha256"],
            expected_version_id=chunk["version_id"],
            label="staged chunk",
        )
        metadata = head.get("Metadata", {})
        if metadata.get("source-sha256") != source["source_sha256"]:
            raise ValueError("staged chunk source metadata changed")
        if metadata.get("start-line") != str(chunk["start_line"]):
            raise ValueError("staged chunk start-line metadata changed")
        if metadata.get("end-line") != str(chunk["end_line"]):
            raise ValueError("staged chunk end-line metadata changed")
    return {
        "chunk_objects": len(manifest["chunks"]),
        "compressed_bytes": int(manifest["total_compressed_bytes"]),
        "manifest_objects": 1,
        "rows": int(manifest["total_rows"]),
        "uncompressed_bytes": int(manifest["total_uncompressed_bytes"]),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-result-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    import boto3
    from botocore.config import Config

    args = _args()
    if args.workers < 1 or args.workers > 128:
        raise ValueError("verification workers must be between 1 and 128")
    expected_result_sha = _valid_sha256(
        args.expected_stage_result_sha256, "expected stage-result SHA-256"
    )
    if _sha256_file(args.stage_result) != expected_result_sha:
        raise ValueError("local stage-result SHA-256 changed")
    release = _read_json(args.release)
    authorization = _read_json(args.authorization)
    result = _read_json(args.stage_result)
    driver = validate_stage_result(result, release, authorization)
    sources = sorted(result["sources"], key=lambda row: int(row["source_ordinal"]))
    planned = sorted(release["sources"], key=lambda row: int(row["ordinal"]))
    client = boto3.client(
        "s3",
        region_name=str(release["region"]),
        config=Config(
            connect_timeout=10,
            max_pool_connections=args.workers,
            read_timeout=60,
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )
    bucket = str(release["storage"]["bucket"])
    key_prefix = (
        f"{release['output']['staged_prefix']}/{release['release_fingerprint']}"
    )
    pairs = list(zip(sources, planned, strict=True))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        verified = list(
            executor.map(
                lambda pair: _verify_source(
                    pair,
                    bucket=bucket,
                    key_prefix=key_prefix,
                    s3_client=client,
                ),
                pairs,
            )
        )
    remote = {
        key: sum(row[key] for row in verified)
        for key in (
            "chunk_objects",
            "compressed_bytes",
            "manifest_objects",
            "rows",
            "uncompressed_bytes",
        )
    }
    if remote != {
        "chunk_objects": driver["staged_chunks"],
        "compressed_bytes": driver["staged_compressed_bytes"],
        "manifest_objects": driver["source_files"],
        "rows": driver["staged_rows"],
        "uncompressed_bytes": driver["staged_uncompressed_bytes"],
    }:
        raise ValueError("remote staging aggregate totals changed")
    report = {
        "authorization_sha256": authorization["authorization_sha256"],
        "driver_result": driver,
        "gate": "lumi_glue_production_stage_verification_v0.1",
        "release_fingerprint": release["release_fingerprint"],
        "remote_objects": remote,
        "schema_version": 1,
        "source_commit": release["source_commit"],
        "stage_result_sha256": expected_result_sha,
        "status": "pass",
        "training_started": False,
        "verified_at": datetime.now(UTC).isoformat(),
        "workers": args.workers,
    }
    report_path = args.output_directory / "stage-verification-v0.1.json"
    _write_json_atomic(report_path, report)
    manifest_path = args.output_directory / "manifest.sha256"
    manifest_path.write_text(
        f"{_sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
