#!/usr/bin/env python3
"""Run one authorized production final-80B selection-and-sharding job."""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

GLUE_REQUESTED_OPTIONS = (
    "JOB_NAME",
    "LUMI_AUTHORIZATION_FILE",
    "LUMI_CANARY_REPORT_FILE",
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


def _normalized_tar_member_name(value: object) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Glue final-corpus baseline archive member is unsafe")
    return "/".join(part for part in path.parts if part not in ("", "."))


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
            raise ValueError("existing final-corpus JSON bytes changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing final-corpus JSON SHA-256 changed") from error


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


def _verify_input_inventory(
    s3_client: Any, release: Mapping[str, Any]
) -> list[dict[str, Any]]:
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
    output: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in release["categories"]:
        category = str(row["category"])
        marker = f"/category={category}/"
        matching = sorted(
            str(obj["key"]) for obj in objects if marker in f"/{obj['key']}"
        )
        if not matching:
            raise RuntimeError(f"Glue final-corpus production has no input: {category}")
        if set(matching) & seen:
            raise RuntimeError("Glue final-corpus production repeats an input object")
        seen.update(matching)
        output[category] = [f"s3://{bucket}/{key}" for key in matching]
    if seen != {str(row["key"]) for row in objects}:
        raise RuntimeError("Glue final-corpus production left input objects unassigned")
    return output


def _distributed_selection_digest(frame: Any) -> str:
    columns = (*ORDER_COLUMNS, "token_count")

    def summarize(index: int, rows: Any) -> Any:
        digest = hashlib.sha256()
        documents = 0
        tokens = 0
        first = None
        last = None
        for row in rows:
            material = {name: row[name] for name in columns}
            encoded = _canonical_json_bytes(material)
            digest.update(encoded)
            digest.update(b"\n")
            documents += 1
            tokens += int(row["token_count"])
            order = [row[name] for name in ORDER_COLUMNS]
            first = order if first is None else first
            last = order
        yield {
            "digest": digest.hexdigest(),
            "documents": documents,
            "first": first,
            "index": index,
            "last": last,
            "tokens": tokens,
        }

    summaries = sorted(
        frame.select(*columns).rdd.mapPartitionsWithIndex(summarize).collect(),
        key=lambda row: int(row["index"]),
    )
    return _sha256_bytes(_canonical_json_bytes(summaries))


def _partition_by_shard(key: tuple[Any, ...]) -> int:
    return int(key[0])


def _seal_token_partitions(rows: Any, config: Mapping[str, Any]) -> Any:
    import boto3

    s3 = boto3.client("s3", region_name=str(config["region"]))
    scratch = Path(tempfile.mkdtemp(prefix="lumi-final-production-shard-"))
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
        key = f"{config['prefix']}/{config['category']}/{digest}.uint16.bin"
        _put_immutable_file(
            path,
            bucket=str(config["bucket"]),
            key=key,
            digest=digest,
            s3_client=s3,
        )
        yield {
            "category": str(config["category"]),
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
        Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), ChecksumMode="ENABLED"
    )
    if int(head.get("ContentLength", -1)) != expected_bytes:
        raise ValueError("Glue final-corpus source object byte size changed")
    expected_checksum = base64.b64encode(bytes.fromhex(expected_sha256)).decode()
    if (
        head.get("Metadata", {}).get("sha256") != expected_sha256
        and head.get("ChecksumSHA256") != expected_checksum
    ):
        raise ValueError("Glue final-corpus source object SHA-256 changed")


def _copy_baseline_object(
    s3_client: Any,
    *,
    bucket: str,
    source_uri: str,
    output_prefix: str,
    sequence: int,
    category: str,
    documents: int,
    tokens: int,
    digest: str,
    expected_bytes: int,
) -> dict[str, Any]:
    _verify_head(
        s3_client,
        uri=source_uri,
        expected_bytes=expected_bytes,
        expected_sha256=digest,
    )
    parsed = urlparse(source_uri)
    key = f"{output_prefix}/{category}/{digest}.uint16.bin"
    scratch = Path(tempfile.mkdtemp(prefix="lumi-final-baseline-copy-"))
    try:
        local = scratch / "tokens.uint16.bin"
        s3_client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local))
        if local.stat().st_size != expected_bytes or _sha256_file(local) != digest:
            raise ValueError("Glue final-corpus baseline download changed")
        _put_immutable_file(
            local,
            bucket=bucket,
            key=key,
            digest=digest,
            s3_client=s3_client,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "category": category,
        "documents": documents,
        "object_bytes": expected_bytes,
        "object_sha256": digest,
        "s3_uri": f"s3://{bucket}/{key}",
        "sequence": sequence,
        "tokens": tokens,
    }


def _materialize_partial_baseline(
    s3_client: Any,
    *,
    bucket: str,
    archive: Mapping[str, Any],
    partial: Mapping[str, Any],
    output_prefix: str,
) -> dict[str, Any]:
    import zstandard

    _verify_head(
        s3_client,
        uri=str(archive["s3_uri"]),
        expected_bytes=int(archive["bytes"]),
        expected_sha256=str(archive["sha256"]),
    )
    scratch = Path(tempfile.mkdtemp(prefix="lumi-final-baseline-"))
    try:
        archive_path = scratch / "baseline.tar.zst"
        tar_path = scratch / "baseline.tar"
        token_path = scratch / "partial.uint16.bin"
        parsed = urlparse(str(archive["s3_uri"]))
        s3_client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(archive_path))
        if _sha256_file(archive_path) != archive["sha256"]:
            raise ValueError("Glue final-corpus baseline archive download changed")
        with archive_path.open("rb") as source, tar_path.open("wb") as destination:
            zstandard.ZstdDecompressor().copy_stream(source, destination)
        with tarfile.open(tar_path, mode="r:") as archive_handle:
            expected_member = _normalized_tar_member_name(partial["member"])
            if not expected_member:
                raise ValueError("Glue final-corpus baseline member is empty")
            matches = [
                candidate
                for candidate in archive_handle.getmembers()
                if _normalized_tar_member_name(candidate.name) == expected_member
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Glue final-corpus baseline archive member is missing or ambiguous"
                )
            member = matches[0]
            if not member.isfile() or member.size != int(partial["bytes"]):
                raise ValueError("Glue final-corpus baseline archive member changed")
            source = archive_handle.extractfile(member)
            if source is None:
                raise ValueError("Glue final-corpus baseline member cannot be read")
            with source, token_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        digest = _sha256_file(token_path)
        if digest != partial["sha256"]:
            raise ValueError("Glue final-corpus baseline partial SHA-256 changed")
        key = f"{output_prefix}/science_and_education/{digest}.uint16.bin"
        _put_immutable_file(
            token_path,
            bucket=bucket,
            key=key,
            digest=digest,
            s3_client=s3_client,
        )
        return {
            "category": "science_and_education",
            "documents": int(partial["documents"]),
            "object_bytes": int(partial["bytes"]),
            "object_sha256": digest,
            "s3_uri": f"s3://{bucket}/{key}",
            "sequence": 2,
            "tokens": int(partial["tokens"]),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles, StorageLevel
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    from lumi.final_corpus import (
        SELECTION_RESULT_GATE,
        SHARD_MANIFEST_GATE,
        assign_hash_ranked_token_shards,
        final_corpus_plan_fingerprint,
        final_selection_result_fingerprint,
        select_hash_ranked_token_prefix,
    )
    from lumi.glue_corpus import candidate_spark_schema
    from lumi.glue_final_corpus_production import (
        CATEGORY_CHECKPOINT_GATE,
        RESULT_GATE,
        category_checkpoint_fingerprint,
        glue_final_corpus_production_job_token,
        production_result_fingerprint,
        validate_glue_final_corpus_category_checkpoint,
        validate_glue_final_corpus_production_authorization,
        validate_glue_final_corpus_production_release,
        validate_glue_final_corpus_production_result,
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
    canary_path = Path(SparkFiles.get(args["LUMI_CANARY_REPORT_FILE"]))
    release = _read_json(release_path)
    plan = _read_json(plan_path)
    authorization = _read_json(authorization_path)
    validate_glue_final_corpus_production_release(release)
    validate_glue_final_corpus_production_authorization(authorization, release)
    if _sha256_file(plan_path) != release["artifacts"]["final_plan"]["sha256"]:
        raise ValueError("Glue final-corpus production plan file changed")
    if _sha256_file(canary_path) != release["canary"]["report_file_sha256"]:
        raise ValueError("Glue final-corpus production canary report changed")
    if plan.get("plan_fingerprint") != final_corpus_plan_fingerprint(plan):
        raise ValueError("Glue final-corpus production plan fingerprint changed")
    if plan["plan_fingerprint"] != release["input"]["plan_fingerprint"]:
        raise ValueError("Glue final-corpus production plan targets another release")
    runtime = {
        "parquet_reader": parquet_runtime,
        "python": platform.python_version(),
        "spark": str(spark.version),
    }
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError("Glue final-corpus production requires Python 3.11")
    if not str(spark.version).startswith("3.5.6"):
        raise RuntimeError("Glue final-corpus production Spark runtime changed")

    bucket = str(release["storage"]["bucket"])
    s3 = boto3.client("s3", region_name=str(release["region"]))
    objects = _verify_input_inventory(s3, release)
    category_uris = _category_object_uris(objects, release)
    token = glue_final_corpus_production_job_token(authorization)
    run_prefix = f"{release['output']['run_prefix']}/{token}"
    packed_prefix = f"{run_prefix}/packed"
    result_prefix = f"{release['output']['result_prefix']}/{token}"

    baseline_shards = []
    for sequence, row in enumerate(release["baseline"]["sealed_shards"]):
        baseline_shards.append(
            _copy_baseline_object(
                s3,
                bucket=bucket,
                source_uri=str(row["s3_uri"]),
                output_prefix=packed_prefix,
                sequence=sequence,
                category=str(row["category"]),
                documents=int(row["documents"]),
                tokens=int(row["tokens"]),
                digest=str(row["sha256"]),
                expected_bytes=int(row["bytes"]),
            )
        )
    baseline_shards.append(
        _materialize_partial_baseline(
            s3,
            bucket=bucket,
            archive=release["baseline"]["archive"],
            partial=release["baseline"]["partial"],
            output_prefix=packed_prefix,
        )
    )

    schema = candidate_spark_schema()
    physical_schema = T.StructType(
        [field for field in schema.fields if field.name != "category"]
    )
    category_checkpoints = []
    selected_categories = []
    new_shards = []
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
        if available_documents != int(category_row["available_candidate_documents"]):
            raise RuntimeError(f"final-corpus available documents changed: {category}")
        if available_tokens != int(category_row["available_candidate_tokens"]):
            raise RuntimeError(f"final-corpus available tokens changed: {category}")

        selection = select_hash_ranked_token_prefix(
            frame,
            seed_sha256=str(release["input"]["selection_seed_sha256"]),
            target_tokens=int(category_row["requested_candidate_tokens"]),
            maximum_overshoot_tokens=int(
                release["limits"]["maximum_document_boundary_overshoot_tokens"]
            ),
            partitions=int(release["limits"]["selection_partitions"]),
        )
        selected = selection["selected"].persist(StorageLevel.DISK_ONLY)
        selection_sha = _distributed_selection_digest(selected)
        selected_key = f"{run_prefix}/selected/{category}"
        selected.write.mode("errorifexists").parquet(f"s3://{bucket}/{selected_key}")
        selected_inventory = _inventory(
            s3, bucket=bucket, prefix=f"{selected_key}/"
        )
        selected_inventory.pop("objects")

        assignment = assign_hash_ranked_token_shards(
            selected,
            target_tokens=int(release["limits"]["packed_shard_target_tokens"]),
            maximum_overshoot_tokens=int(
                release["limits"]["maximum_document_boundary_overshoot_tokens"]
            ),
            start_sequence=int(category_row["first_new_shard_sequence"]),
            partitions=int(release["limits"]["selection_partitions"]),
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
            "prefix": packed_prefix,
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
        checkpoint = {
            "authorization_sha256": authorization["authorization_sha256"],
            "available_candidate_documents": available_documents,
            "available_candidate_tokens": available_tokens,
            "category": category,
            "cutoff": selection["cutoff"],
            "gate": CATEGORY_CHECKPOINT_GATE,
            "job_token": token,
            "output_bytes": int(selected_inventory["object_bytes"])
            + sum(int(row["object_bytes"]) for row in packed),
            "packed_shards": packed,
            "plan_fingerprint": release["input"]["plan_fingerprint"],
            "release_fingerprint": release["release_fingerprint"],
            "schema_version": 1,
            "selected_candidate_documents": int(selection["selected_documents"]),
            "selected_candidate_tokens": int(selection["selected_tokens"]),
            "selected_inventory": {
                **selected_inventory,
                "s3_uri": f"s3://{bucket}/{selected_key}",
            },
            "selection_sha256": selection_sha,
            "status": "complete",
        }
        checkpoint["checkpoint_sha256"] = category_checkpoint_fingerprint(checkpoint)
        validate_glue_final_corpus_category_checkpoint(
            checkpoint, release, authorization
        )
        checkpoint_key = f"{run_prefix}/checkpoints/{category}.json"
        _put_immutable_json(checkpoint, bucket=bucket, key=checkpoint_key, s3_client=s3)
        category_checkpoints.append(checkpoint)
        selected_categories.append(
            {
                "available_candidate_documents": available_documents,
                "available_candidate_tokens": available_tokens,
                "category": category,
                "cutoff": selection["cutoff"],
                "final_documents": int(category_row["baseline_documents"])
                + int(selection["selected_documents"]),
                "final_tokens": int(category_row["baseline_tokens"])
                + int(selection["selected_tokens"]),
                "selected_candidate_documents": int(selection["selected_documents"]),
                "selected_candidate_tokens": int(selection["selected_tokens"]),
                "selected_inventory": checkpoint["selected_inventory"],
                "selection_sha256": selection_sha,
            }
        )
        new_shards.extend(packed)
        selected.unpersist()
        frame.unpersist()

    selection_result = {
        "categories": selected_categories,
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "final_documents": sum(int(row["final_documents"]) for row in selected_categories),
        "final_tokens": sum(int(row["final_tokens"]) for row in selected_categories),
        "gate": SELECTION_RESULT_GATE,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "schema_version": 1,
        "selection_seed_sha256": release["input"]["selection_seed_sha256"],
        "status": "complete",
        "training_token_limit": release["limits"]["training_token_limit"],
    }
    selection_result["result_sha256"] = final_selection_result_fingerprint(
        selection_result
    )
    all_shards = sorted(
        [*baseline_shards, *new_shards], key=lambda row: int(row["sequence"])
    )
    shard_manifest = {
        "documents": sum(int(row["documents"]) for row in all_shards),
        "gate": SHARD_MANIFEST_GATE,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "schema_version": 1,
        "selection_result_sha256": selection_result["result_sha256"],
        "shards": all_shards,
        "status": "sealed",
        "tokens": sum(int(row["tokens"]) for row in all_shards),
        "training_token_limit": release["limits"]["training_token_limit"],
    }
    result = {
        "authorization_sha256": authorization["authorization_sha256"],
        "category_checkpoints": category_checkpoints,
        "data_output_bytes": sum(int(row["output_bytes"]) for row in category_checkpoints)
        + sum(int(row["object_bytes"]) for row in baseline_shards),
        "dedup_result_sha256": release["input"]["dedup_result_sha256"],
        "gate": RESULT_GATE,
        "job_token": token,
        "plan_fingerprint": release["input"]["plan_fingerprint"],
        "release_fingerprint": release["release_fingerprint"],
        "runtime": runtime,
        "schema_version": 1,
        "selection_result": selection_result,
        "shard_manifest": shard_manifest,
        "status": "sealed",
    }
    result["result_sha256"] = production_result_fingerprint(result)
    validate_glue_final_corpus_production_result(
        result, release, authorization, plan
    )
    _put_immutable_json(
        selection_result,
        bucket=bucket,
        key=f"{result_prefix}/selection-result.json",
        s3_client=s3,
    )
    _put_immutable_json(
        shard_manifest,
        bucket=bucket,
        key=f"{result_prefix}/shard-manifest.json",
        s3_client=s3,
    )
    _put_immutable_json(
        result,
        bucket=bucket,
        key=f"{result_prefix}/result.json",
        s3_client=s3,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    job.commit()


if __name__ == "__main__":
    main()
