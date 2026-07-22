#!/usr/bin/env python3
"""Run one exactly authorized Lumi final-80B selection/sharding Glue canary."""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

GLUE_REQUESTED_OPTIONS = (
    "JOB_NAME",
    "LUMI_AUTHORIZATION_FILE",
    "LUMI_FINAL_PLAN_FILE",
    "LUMI_RELEASE_FILE",
)

ORDER_COLUMNS = (
    "_lumi_selection_rank",
    "content_sha256",
    "document_id",
    "source_ordinal",
    "source_line",
    "segment_index",
)

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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _put_immutable_json(
    value: Mapping[str, Any], *, bucket: str, key: str, s3_client: Any
) -> None:
    body = _canonical_json_bytes(value)
    digest = _sha256_bytes(body)
    try:
        s3_client.put_object(
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
            raise ValueError("existing final-corpus result byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing final-corpus result SHA-256 changed") from error


def _put_immutable_file(
    path: Path, *, bucket: str, key: str, digest: str, s3_client: Any
) -> None:
    try:
        with path.open("rb") as body:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
                ContentType="application/octet-stream",
                IfNoneMatch="*",
                Metadata={"dtype": "uint16_le", "sha256": digest},
                ServerSideEncryption="AES256",
            )
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != path.stat().st_size:
            raise ValueError("existing final-corpus token bytes changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing final-corpus token SHA-256 changed") from error


def _inventory(s3_client: Any, *, bucket: str, prefix: str) -> dict[str, Any]:
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for row in page.get("Contents", []):
            key = str(row["Key"])
            if key.endswith("/_SUCCESS"):
                continue
            objects.append({"bytes": int(row["Size"]), "key": key})
    if not objects:
        raise RuntimeError("Glue final-corpus S3 inventory is empty")
    objects.sort(key=lambda row: row["key"])
    return {
        "object_bytes": sum(int(row["bytes"]) for row in objects),
        "object_count": len(objects),
        "objects": objects,
        "objects_sha256": _sha256_bytes(_canonical_json_bytes(objects)),
    }


def _verify_input_inventory(s3_client: Any, release: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = release["input"]["accepted_candidates"]
    parsed = urlparse(str(expected["s3_uri"]))
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    observed = _inventory(s3_client, bucket=parsed.netloc, prefix=prefix)
    for name in ("object_bytes", "object_count", "objects_sha256"):
        if observed[name] != expected[name]:
            raise ValueError(f"Glue final-corpus accepted inventory changed: {name}")
    return list(observed["objects"])


def _category_object_uris(
    objects: list[dict[str, Any]], release: Mapping[str, Any]
) -> dict[str, list[str]]:
    bucket = str(release["storage"]["bucket"])
    maximum = int(release["limits"]["maximum_input_objects_per_category"])
    output: dict[str, list[str]] = {}
    seen_keys: set[str] = set()
    for category in [str(row["category"]) for row in release["categories"]]:
        marker = f"/category={category}/"
        matching = [str(row["key"]) for row in objects if marker in f"/{row['key']}" ]
        if not matching:
            raise RuntimeError(f"Glue final-corpus canary has no input for {category}")
        selected = sorted(matching)[:maximum]
        if len(selected) != maximum:
            raise RuntimeError(f"Glue final-corpus canary input-object cap is unmet: {category}")
        if set(selected) & seen_keys:
            raise RuntimeError("Glue final-corpus canary repeats an input object")
        seen_keys.update(selected)
        output[category] = [f"s3://{bucket}/{key}" for key in selected]
    return output


def _selection_digest(frame: Any) -> str:
    digest = hashlib.sha256()
    for row in frame.select(*ORDER_COLUMNS, "token_count").orderBy(
        *ORDER_COLUMNS
    ).toLocalIterator():
        digest.update(
            _canonical_json_bytes(
                {name: row[name] for name in (*ORDER_COLUMNS, "token_count")}
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _partition_by_shard(key: tuple[Any, ...]) -> int:
    return int(key[0])


def _seal_token_partitions(rows: Any, config: Mapping[str, Any]) -> Any:
    import boto3

    s3 = boto3.client("s3", region_name=str(config["region"]))
    scratch = Path(tempfile.mkdtemp(prefix="lumi-final-canary-shard-"))
    try:
        sequence = None
        documents = 0
        tokens = 0
        previous = None
        path = scratch / "tokens.uint16.bin"
        with path.open("wb") as handle:
            for key, row in rows:
                current_sequence = int(key[0])
                if sequence is None:
                    sequence = current_sequence
                if current_sequence != sequence:
                    raise RuntimeError("final-corpus partition crosses shard sequences")
                order = tuple(key[1:])
                if previous is not None and order <= previous:
                    raise RuntimeError("final-corpus packed rows are not strictly ordered")
                previous = order
                token_count = int(row["token_count"])
                token_bytes = bytes(row["token_bytes"])
                if token_count <= 0 or len(token_bytes) != token_count * 2:
                    raise ValueError("final-corpus candidate token payload changed")
                handle.write(token_bytes)
                documents += 1
                tokens += token_count
            handle.flush()
        if sequence is None or documents == 0 or tokens == 0:
            return
        if path.stat().st_size != tokens * 2:
            raise RuntimeError("final-corpus packed token-byte conservation failed")
        digest = _sha256_file(path)
        key = (
            f"{config['prefix']}/{config['category']}/{digest}.uint16.bin"
        )
        _put_immutable_file(
            path,
            bucket=str(config["bucket"]),
            key=key,
            digest=digest,
            s3_client=s3,
        )
        yield {
            "documents": documents,
            "object_bytes": path.stat().st_size,
            "object_sha256": digest,
            "s3_uri": f"s3://{config['bucket']}/{key}",
            "sequence": sequence,
            "tokens": tokens,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _verify_head(
    s3_client: Any, *, uri: str, expected_bytes: int, expected_sha256: str
) -> None:
    parsed = urlparse(uri)
    head = s3_client.head_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        ChecksumMode="ENABLED",
    )
    if int(head.get("ContentLength", -1)) != expected_bytes:
        raise ValueError("Glue final-corpus baseline object byte size changed")
    expected_checksum = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
    if (
        head.get("Metadata", {}).get("sha256") != expected_sha256
        and head.get("ChecksumSHA256") != expected_checksum
    ):
        raise ValueError("Glue final-corpus baseline object SHA-256 changed")


def main() -> None:
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles, StorageLevel
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    from lumi.final_corpus import (
        assign_hash_ranked_token_shards,
        final_corpus_plan_fingerprint,
        select_hash_ranked_token_prefix,
    )
    from lumi.glue_corpus import candidate_spark_schema
    from lumi.glue_final_corpus import (
        glue_final_corpus_canary_job_token,
        validate_glue_final_corpus_canary_authorization,
        validate_glue_final_corpus_canary_release,
        validate_glue_final_corpus_canary_result,
    )
    from lumi.glue_runtime import (
        FINAL_CORPUS_SPARK_CONFIG,
        verify_spark_configuration,
    )

    args = getResolvedOptions(__import__("sys").argv, list(GLUE_REQUESTED_OPTIONS))
    context = SparkContext.getOrCreate()
    glue = GlueContext(context)
    spark = glue.spark_session
    parquet_runtime = verify_spark_configuration(spark, FINAL_CORPUS_SPARK_CONFIG)
    job = Job(glue)
    job.init(args["JOB_NAME"], args)
    release_path = Path(SparkFiles.get(args["LUMI_RELEASE_FILE"]))
    plan_path = Path(SparkFiles.get(args["LUMI_FINAL_PLAN_FILE"]))
    authorization_path = Path(SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"]))
    release = _read_json(release_path)
    plan = _read_json(plan_path)
    authorization = _read_json(authorization_path)
    validate_glue_final_corpus_canary_release(release)
    validate_glue_final_corpus_canary_authorization(authorization, release)
    if _sha256_file(plan_path) != release["artifacts"]["final_plan"]["sha256"]:
        raise ValueError("Glue final-corpus runtime plan file changed")
    if plan.get("plan_fingerprint") != final_corpus_plan_fingerprint(plan):
        raise ValueError("Glue final-corpus runtime plan fingerprint changed")
    if plan["plan_fingerprint"] != release["input"]["plan_fingerprint"]:
        raise ValueError("Glue final-corpus runtime plan targets another release")

    runtime = {
        "parquet_reader": parquet_runtime,
        "python": platform.python_version(),
        "spark": str(spark.version),
    }
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError("Glue final-corpus canary requires Python 3.11")
    if not str(spark.version).startswith(str(release["runtime"]["spark_version"])):
        raise RuntimeError("Glue final-corpus canary Spark runtime changed")

    bucket = str(release["storage"]["bucket"])
    s3 = boto3.client("s3", region_name=str(release["region"]))
    objects = _verify_input_inventory(s3, release)
    category_uris = _category_object_uris(objects, release)
    for row in release["baseline"]["sealed_shards"]:
        _verify_head(
            s3,
            uri=str(row["s3_uri"]),
            expected_bytes=int(row["bytes"]),
            expected_sha256=str(row["sha256"]),
        )
    archive = release["baseline"]["archive"]
    _verify_head(
        s3,
        uri=str(archive["s3_uri"]),
        expected_bytes=int(archive["bytes"]),
        expected_sha256=str(archive["sha256"]),
    )

    token = glue_final_corpus_canary_job_token(authorization)
    run_prefix = f"{release['output']['canary_prefix']}/{token}"
    category_reports = []
    total_documents = 0
    total_tokens = 0
    output_bytes = 0
    schema = candidate_spark_schema()
    physical_schema = T.StructType(
        [field for field in schema.fields if field.name != "category"]
    )
    target = int(release["limits"]["selected_target_tokens_per_category"])
    overshoot = int(release["limits"]["maximum_document_boundary_overshoot_tokens"])
    for category_row in release["categories"]:
        category = str(category_row["category"])
        frame = (
            spark.read.schema(physical_schema)
            .parquet(*category_uris[category])
            .withColumn("category", F.lit(category))
            .select(*[field.name for field in schema.fields])
            .persist(StorageLevel.DISK_ONLY)
        )
        totals = frame.agg(
            F.count(F.lit(1)).alias("documents"),
            F.sum("token_count").alias("tokens"),
        ).first()
        available_documents = int(totals["documents"] or 0)
        available_tokens = int(totals["tokens"] or 0)
        if available_documents == 0 or available_tokens < target:
            raise RuntimeError(f"Glue final-corpus canary input is too small: {category}")
        if available_documents > int(
            release["limits"]["maximum_candidate_documents_per_category"]
        ):
            raise RuntimeError(f"Glue final-corpus canary document cap exceeded: {category}")

        first = select_hash_ranked_token_prefix(
            frame,
            seed_sha256=str(release["input"]["selection_seed_sha256"]),
            target_tokens=target,
            maximum_overshoot_tokens=overshoot,
            partitions=16,
        )
        second = select_hash_ranked_token_prefix(
            frame,
            seed_sha256=str(release["input"]["selection_seed_sha256"]),
            target_tokens=target,
            maximum_overshoot_tokens=overshoot,
            partitions=23,
        )
        first_digest = _selection_digest(first["selected"])
        second_digest = _selection_digest(second["selected"])
        if (
            first_digest != second_digest
            or int(first["selected_documents"]) != int(second["selected_documents"])
            or int(first["selected_tokens"]) != int(second["selected_tokens"])
            or first["cutoff"] != second["cutoff"]
        ):
            raise RuntimeError(f"Glue final-corpus partition parity failed: {category}")
        second["selected"].unpersist()
        selected = first["selected"].persist(StorageLevel.DISK_ONLY)
        selected_key = f"{run_prefix}/selected/{category}"
        selected.write.mode("errorifexists").parquet(f"s3://{bucket}/{selected_key}")
        selected_inventory = _inventory(
            s3,
            bucket=bucket,
            prefix=f"{selected_key}/",
        )
        selected_inventory.pop("objects")

        assignment = assign_hash_ranked_token_shards(
            selected,
            target_tokens=int(release["limits"]["packed_shard_target_tokens"]),
            maximum_overshoot_tokens=overshoot,
            partitions=16,
        )
        keyed = assignment["assigned"].rdd.map(
            lambda row: (
                (
                    int(row["shard_sequence"]),
                    *[row[name] for name in ORDER_COLUMNS],
                ),
                row,
            )
        )
        pack_config = {
            "bucket": bucket,
            "category": category,
            "prefix": f"{run_prefix}/packed",
            "region": release["region"],
        }
        packed = (
            keyed.repartitionAndSortWithinPartitions(
                numPartitions=int(assignment["shard_count"]),
                partitionFunc=_partition_by_shard,
            )
            .mapPartitions(
                lambda rows, config=pack_config: _seal_token_partitions(rows, config)
            )
            .collect()
        )
        assignment["assigned"].unpersist()
        packed.sort(key=lambda row: int(row["sequence"]))
        if [int(row["sequence"]) for row in packed] != list(range(len(packed))):
            raise RuntimeError("Glue final-corpus packed sequence changed")
        if sum(int(row["tokens"]) for row in packed) != int(first["selected_tokens"]):
            raise RuntimeError("Glue final-corpus packed token conservation failed")
        if sum(int(row["documents"]) for row in packed) != int(
            first["selected_documents"]
        ):
            raise RuntimeError("Glue final-corpus packed document conservation failed")
        category_reports.append(
            {
                "available_documents": available_documents,
                "available_tokens": available_tokens,
                "category": category,
                "cutoff": first["cutoff"],
                "packed_shards": packed,
                "partition_parity": "pass",
                "selected_documents": int(first["selected_documents"]),
                "selected_inventory": {
                    **selected_inventory,
                    "s3_uri": f"s3://{bucket}/{selected_key}",
                },
                "selected_tokens": int(first["selected_tokens"]),
                "selection_sha256": first_digest,
            }
        )
        total_documents += int(first["selected_documents"])
        total_tokens += int(first["selected_tokens"])
        output_bytes += int(selected_inventory["object_bytes"]) + sum(
            int(row["object_bytes"]) for row in packed
        )
        selected.unpersist()
        frame.unpersist()

    report = {
        "accepted_input_inventory": dict(release["input"]["accepted_candidates"]),
        "authorization_sha256": authorization["authorization_sha256"],
        "baseline_verification": {
            "archive_head": "pass",
            "sealed_shard_heads": 2,
            "status": "pass",
        },
        "categories": category_reports,
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "gate": "lumi_glue_final_corpus_canary_result_v0.1",
        "job_token": token,
        "output_bytes": output_bytes,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "runtime": runtime,
        "schema_version": 1,
        "selected_documents": total_documents,
        "selected_tokens": total_tokens,
        "selection_seed_sha256": release["input"]["selection_seed_sha256"],
        "status": "pass_selection_sharding_canary",
    }
    validate_glue_final_corpus_canary_result(report, release, authorization)
    result_key = f"{release['output']['result_prefix']}/{token}/result.json"
    _put_immutable_json(report, bucket=bucket, key=result_key, s3_client=s3)
    print(json.dumps({**report, "result_key": result_key}, indent=2, sort_keys=True))
    job.commit()


if __name__ == "__main__":
    main()
