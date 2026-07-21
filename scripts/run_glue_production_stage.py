#!/usr/bin/env python3
"""Stage all pinned Lumi sources into deterministic private-S3 chunks."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GLUE_REQUESTED_OPTIONS = (
    "JOB_NAME",
    "LUMI_AUTHORIZATION_FILE",
    "LUMI_RELEASE_FILE",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _aws_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return None if code is None else str(code)


def _put_immutable(
    value: Mapping[str, Any], *, bucket: str, key: str, s3_client: Any
) -> dict[str, Any]:
    body = _canonical_json_bytes(value)
    digest = _sha256_bytes(body)
    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={"sha256": digest},
            ServerSideEncryption="AES256",
        )
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != len(body):
            raise ValueError("existing production-stage report byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing production-stage report SHA-256 changed") from error
        version_id = head.get("VersionId")
    else:
        version_id = response.get("VersionId")
    return {
        "bytes": len(body),
        "key": key,
        "sha256": digest,
        "version_id": None if version_id is None else str(version_id),
    }


def _existing_manifest(
    *, source: Mapping[str, Any], bucket: str, key_prefix: str, s3_client: Any
) -> dict[str, Any] | None:
    from lumi.emr_production_staging import validate_published_staged_source

    prefix = f"{key_prefix}/manifests/{source['source_sha256']}/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=3)
    keys = sorted(
        str(row["Key"])
        for row in response.get("Contents", [])
        if str(row["Key"]).endswith(".json")
    )
    if not keys:
        return None
    if len(keys) != 1 or response.get("IsTruncated"):
        raise RuntimeError("production staging found multiple manifests for one source")
    body = s3_client.get_object(Bucket=bucket, Key=keys[0])["Body"].read()
    manifest = json.loads(body)
    validate_published_staged_source(
        manifest,
        bucket=bucket,
        key_prefix=key_prefix,
        hard_transient_bytes=100_000_000_000,
    )
    if manifest.get("source_file") != source["source_file"]:
        raise ValueError("existing staged manifest source filename changed")
    if manifest.get("source_sha256") != source["source_sha256"]:
        raise ValueError("existing staged manifest source digest changed")
    if int(manifest.get("start_after_line", -1)) != int(source["start_after_line"]):
        raise ValueError("existing staged manifest resume line changed")
    return manifest


def _stage_partition(
    source_iterator: Any,
    *,
    bucket: str,
    key_prefix: str,
    target_chunk_bytes: int,
    maximum_source_bytes: int,
    region: str,
) -> Any:
    import boto3
    from pyspark import TaskContext

    from lumi.emr_production_staging import (
        publish_staged_source,
        split_gzip_source,
    )
    from lumi.production_corpus import _download_source

    s3 = boto3.client("s3", region_name=region)
    context = TaskContext.get()
    partition = -1 if context is None else int(context.partitionId())
    for source in source_iterator:
        if int(source["source_bytes"]) > maximum_source_bytes:
            raise ValueError("production staging source exceeds its runtime cap")
        existing = _existing_manifest(
            source=source,
            bucket=bucket,
            key_prefix=key_prefix,
            s3_client=s3,
        )
        reused = existing is not None
        if existing is None:
            scratch = Path(tempfile.mkdtemp(prefix=f"lumi-stage-{partition}-"))
            try:
                source_path = _download_source(
                    source,
                    scratch,
                    int(source["source_bytes"]) + 16 * 1024 * 1024,
                )
                chunks = scratch / "chunks"
                manifest = split_gzip_source(
                    source_path,
                    chunks,
                    source_file=str(source["source_file"]),
                    source_sha256=str(source["source_sha256"]),
                    start_after_line=int(source["start_after_line"]),
                    target_uncompressed_bytes=target_chunk_bytes,
                    hard_transient_bytes=100_000_000_000,
                    hard_compressed_bytes=maximum_source_bytes * 4,
                )
                existing = publish_staged_source(
                    manifest,
                    directory=chunks,
                    bucket=bucket,
                    key_prefix=key_prefix,
                    s3_client=s3,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        assert existing is not None
        result = {
            "category": source["category"],
            "chunk_count": len(existing["chunks"]),
            "component": source["component"],
            "manifest_s3_key": existing["s3_key"],
            "manifest_sha256": existing["manifest_sha256"],
            "manifest_version_id": existing.get("version_id"),
            "reused": reused,
            "source_bytes": int(source["source_bytes"]),
            "source_complete": bool(existing["source_complete"]),
            "source_file": source["source_file"],
            "source_ordinal": int(source["ordinal"]),
            "source_sha256": source["source_sha256"],
            "staged_compressed_bytes": int(existing["total_compressed_bytes"]),
            "staged_rows": int(existing["total_rows"]),
            "staged_uncompressed_bytes": int(existing["total_uncompressed_bytes"]),
            "start_after_line": int(source["start_after_line"]),
        }
        print(
            "lumi_stage_source "
            + json.dumps(
                {
                    "chunks": result["chunk_count"],
                    "ordinal": result["source_ordinal"],
                    "reused": reused,
                    "rows": result["staged_rows"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        yield json.dumps(result, separators=(",", ":"), sort_keys=True)


def main() -> None:
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles

    from lumi.glue_production import (
        validate_glue_production_stage_authorization,
        validate_glue_production_stage_release,
    )

    args = getResolvedOptions(sys.argv, list(GLUE_REQUESTED_OPTIONS))
    if not args.get("JOB_RUN_ID"):
        raise RuntimeError("AWS Glue did not inject its reserved JOB_RUN_ID")
    release = json.loads(Path(SparkFiles.get(args["LUMI_RELEASE_FILE"])).read_text())
    authorization = json.loads(
        Path(SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"])).read_text()
    )
    validate_glue_production_stage_release(release)
    validate_glue_production_stage_authorization(authorization, release)
    context = SparkContext.getOrCreate()
    glue_context = GlueContext(context)
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    started = time.monotonic()
    bucket = release["storage"]["bucket"]
    key_prefix = (
        f"{release['output']['staged_prefix']}/{release['release_fingerprint']}"
    )
    rows = context.parallelize(
        release["sources"], len(release["sources"])
    ).mapPartitions(
        lambda iterator: _stage_partition(
            iterator,
            bucket=bucket,
            key_prefix=key_prefix,
            target_chunk_bytes=int(release["limits"]["target_chunk_uncompressed_bytes"]),
            maximum_source_bytes=int(
                release["limits"]["maximum_source_compressed_bytes"]
            ),
            region=str(release["region"]),
        )
    ).collect()
    sources = sorted((json.loads(row) for row in rows), key=lambda row: row["source_ordinal"])
    expected = release["sources"]
    if len(sources) != len(expected):
        raise RuntimeError("production staging did not return every source")
    if [row["source_ordinal"] for row in sources] != list(range(len(expected))):
        raise RuntimeError("production staging result source order changed")
    if any(not row["source_complete"] for row in sources):
        raise RuntimeError("production staging left an incomplete source")
    if len({row["manifest_sha256"] for row in sources}) != len(sources):
        raise RuntimeError("production staging repeated a manifest")
    staged_bytes = sum(row["staged_compressed_bytes"] for row in sources)
    if staged_bytes > int(release["limits"]["hard_logical_storage_bytes"]):
        raise RuntimeError("production staging exceeded the logical-storage cap")
    report = {
        "authorization_sha256": authorization["authorization_sha256"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gate": "lumi_glue_production_stage_result_v0.1",
        "job_name": args["JOB_NAME"],
        "job_run_id": args["JOB_RUN_ID"],
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "source_commit": release["source_commit"],
        "source_files": len(sources),
        "sources": sources,
        "staged_chunks": sum(row["chunk_count"] for row in sources),
        "staged_compressed_bytes": staged_bytes,
        "staged_rows": sum(row["staged_rows"] for row in sources),
        "staged_uncompressed_bytes": sum(
            row["staged_uncompressed_bytes"] for row in sources
        ),
        "status": "pass",
        "training_started": False,
    }
    s3 = boto3.client("s3", region_name=str(release["region"]))
    report_key = (
        f"{release['output']['result_prefix']}/{release['release_fingerprint']}/"
        f"{args['JOB_RUN_ID']}/stage-result.json"
    )
    report_object = _put_immutable(report, bucket=bucket, key=report_key, s3_client=s3)
    print(json.dumps({**report, "report_object": report_object}, indent=2, sort_keys=True))
    job.commit()


if __name__ == "__main__":
    main()
