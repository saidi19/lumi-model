#!/usr/bin/env python3
"""Run one exactly authorized Lumi Glue global-dedup canary or production job."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GLUE_REQUESTED_OPTIONS = (
    "JOB_NAME",
    "LUMI_AUTHORIZATION_FILE",
    "LUMI_INPUT_INDEX_FILE",
    "LUMI_MAP_CHECKPOINT_FILE",
    "LUMI_MAP_RELEASE_FILE",
    "LUMI_RELEASE_FILE",
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


def _put_immutable(
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
            raise ValueError("existing global-dedup result byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing global-dedup result SHA-256 changed") from error


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
        raise RuntimeError("Glue global-dedup Parquet output is empty")
    objects.sort(key=lambda row: row["key"])
    return {
        "object_bytes": sum(int(row["bytes"]) for row in objects),
        "object_count": len(objects),
        "objects_sha256": _sha256_bytes(_canonical_json_bytes(objects)),
    }


def _verified_spark_file(artifact: Mapping[str, Any], uri: str) -> Path:
    from pyspark import SparkFiles

    path = Path(SparkFiles.get(Path(uri).name))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"Glue global-dedup artifact byte size changed: {path.name}")
    if _sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Glue global-dedup artifact SHA-256 changed: {path.name}")
    return path


def _runtime(spark: Any, *, expected_spark_version: str) -> dict[str, Any]:
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError(
            f"Glue global dedup requires Python 3.11, got {platform.python_version()}"
        )
    versions = {"datasketch": importlib.metadata.version("datasketch")}
    if versions != {"datasketch": "2.0.0"}:
        raise RuntimeError(f"Glue global-dedup dependencies changed: {versions}")
    spark_version = str(spark.version)
    expected_pattern = rf"{re.escape(expected_spark_version)}(?:-amzn-[1-9][0-9]*)?"
    if re.fullmatch(expected_pattern, spark_version) is None:
        raise RuntimeError(
            "Glue global-dedup Spark changed: "
            f"expected {expected_spark_version} or its AWS build suffix, got "
            f"{spark_version}"
        )
    return {
        "datasketch": versions["datasketch"],
        "python": platform.python_version(),
        "spark": spark_version,
        "spark_base": expected_spark_version,
    }


def main() -> None:
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles
    from pyspark.sql import functions as F

    from lumi.distributed_baseline import anchor_from_json
    from lumi.distributed_corpus import MinHashCodec
    from lumi.glue_corpus import (
        candidate_spark_schema,
        dedup_anchor_spark_schema,
        select_ordered_token_prefix,
        spark_deduplicate_candidates,
    )
    from lumi.glue_production_dedup import (
        glue_global_dedup_job_token,
        validate_glue_global_dedup_authorization,
        validate_glue_global_dedup_input_index,
        validate_glue_global_dedup_result,
        validate_glue_production_global_dedup_release,
    )

    args = getResolvedOptions(sys.argv, list(GLUE_REQUESTED_OPTIONS))
    context = SparkContext.getOrCreate()
    glue = GlueContext(context)
    spark = glue.spark_session
    job = Job(glue)
    job.init(args["JOB_NAME"], args)
    release_path = Path(SparkFiles.get(args["LUMI_RELEASE_FILE"]))
    input_path = Path(SparkFiles.get(args["LUMI_INPUT_INDEX_FILE"]))
    authorization_path = Path(SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"]))
    map_release_path = Path(SparkFiles.get(args["LUMI_MAP_RELEASE_FILE"]))
    map_checkpoint_path = Path(SparkFiles.get(args["LUMI_MAP_CHECKPOINT_FILE"]))
    release = _read_json(release_path)
    input_index = _read_json(input_path)
    authorization = _read_json(authorization_path)
    map_release = _read_json(map_release_path)
    map_checkpoint = _read_json(map_checkpoint_path)
    validate_glue_production_global_dedup_release(release)
    validate_glue_global_dedup_input_index(input_index, map_release, map_checkpoint)
    validate_glue_global_dedup_authorization(authorization, release)
    for name, path in (
        ("input_index", input_path),
        ("map_release", map_release_path),
        ("map_checkpoint", map_checkpoint_path),
    ):
        if _sha256_file(path) != release["artifacts"][name]["sha256"]:
            raise ValueError(f"Glue global-dedup runtime artifact changed: {name}")
    if input_index["input_index_sha256"] != release["input_index_sha256"]:
        raise ValueError("Glue global-dedup input index differs from release")

    runtime = _runtime(
        spark,
        expected_spark_version=str(release["runtime"]["spark_version"]),
    )
    anchor_artifact = release["artifacts"]["initial_anchors"]
    anchor_path = _verified_spark_file(
        anchor_artifact, str(anchor_artifact["s3_uri"])
    )
    codec = MinHashCodec.from_config(release["data_quality"]["near_duplicate"])
    anchors = []
    with gzip.open(anchor_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            anchors.append(anchor_from_json(json.loads(line), codec))
    anchor_frame = spark.createDataFrame(
        anchors, schema=dedup_anchor_spark_schema()
    ).persist()
    if anchor_frame.count() != int(release["resume"]["accepted_documents"]):
        raise RuntimeError("Glue global-dedup resume-anchor count changed")

    mode = str(authorization["mode"])
    rows = list(input_index["inputs"])
    if mode == "canary":
        rows = rows[: int(release["jobs"]["canary"]["maximum_input_uris"])]
    candidate_uris = [str(row["parquet_s3_uri"]) for row in rows]
    if not candidate_uris:
        raise RuntimeError("Glue global dedup has no Parquet inputs")
    candidates = spark.read.schema(candidate_spark_schema()).parquet(*candidate_uris)
    if mode == "canary":
        order = ["source_ordinal", "source_line", "segment_index", "document_id"]
        candidates = candidates.orderBy(*[F.col(name).asc() for name in order]).limit(
            int(release["jobs"]["canary"]["maximum_candidate_documents"])
        )
    candidates = candidates.persist()
    totals = candidates.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    candidate_documents = int(totals["documents"] or 0)
    candidate_tokens = int(totals["tokens"] or 0)
    if candidate_documents == 0 or candidate_tokens == 0:
        raise RuntimeError("Glue global-dedup candidate input is empty")
    if mode == "production" and (
        candidate_documents,
        candidate_tokens,
    ) != (
        int(input_index["mapped_documents"]),
        int(input_index["mapped_tokens"]),
    ):
        raise RuntimeError("Glue global-dedup candidate totals changed")

    bucket = str(release["storage"]["bucket"])
    token = glue_global_dedup_job_token(authorization)
    checkpoint_uri = (
        f"s3://{bucket}/{release['output']['checkpoint_prefix']}/"
        f"{release['input_index_sha256']}/{token}"
    )
    caps = release["deduplication"]
    dedup = spark_deduplicate_candidates(
        candidates,
        codec=codec,
        checkpoint_uri=checkpoint_uri,
        anchors=anchor_frame,
        maximum_rounds=int(caps["maximum_rounds"]),
        maximum_lsh_bucket_documents=int(caps["maximum_lsh_bucket_documents"]),
        maximum_candidate_pairs=int(caps["maximum_candidate_pairs"]),
    )
    accepted = dedup["accepted"].persist()
    s3 = boto3.client("s3", region_name=str(release["region"]))

    if mode == "canary":
        prefix = f"{release['output']['canary_prefix']}/{token}"
        accepted_key = f"{prefix}/accepted-ids"
        rejected_key = f"{prefix}/rejected-ids"
        accepted.select("document_id", "dedup_round").write.mode(
            "errorifexists"
        ).parquet(f"s3://{bucket}/{accepted_key}")
        dedup["rejected_ids"].write.mode("errorifexists").parquet(
            f"s3://{bucket}/{rejected_key}"
        )
        accepted_inventory = _inventory(
            s3, bucket=bucket, prefix=f"{accepted_key}/"
        )
        rejected_inventory = _inventory(
            s3, bucket=bucket, prefix=f"{rejected_key}/"
        )
        report = {
            "accepted_ids": {
                **accepted_inventory,
                "s3_uri": f"s3://{bucket}/{accepted_key}",
            },
            "authorization_sha256": authorization["authorization_sha256"],
            "deduplication": dedup["metrics"],
            "gate": "lumi_glue_production_global_dedup_canary_result_v0.1",
            "input_documents": candidate_documents,
            "input_index_sha256": input_index["input_index_sha256"],
            "input_tokens": candidate_tokens,
            "job_token": token,
            "map_checkpoint_sha256": input_index["map_checkpoint_sha256"],
            "mode": mode,
            "rejected_ids": {
                **rejected_inventory,
                "s3_uri": f"s3://{bucket}/{rejected_key}",
            },
            "release_fingerprint": release["release_fingerprint"],
            "runtime": runtime,
            "schema_version": 1,
            "status": "pass_structural_canary",
        }
    else:
        selected_frames = []
        category_results = []
        partitions = max(1, int(spark.conf.get("spark.sql.shuffle.partitions")))
        for category in release["categories"]:
            name = str(category["category"])
            remaining = int(category["floor_tokens"]) - int(
                category["accepted_tokens"]
            )
            available = accepted.where(F.col("category") == F.lit(name)).persist()
            available_totals = available.agg(
                F.count(F.lit(1)).alias("documents"),
                F.sum("token_count").alias("tokens"),
            ).first()
            available_documents = int(available_totals["documents"] or 0)
            available_tokens = int(available_totals["tokens"] or 0)
            if available_tokens < remaining:
                raise RuntimeError(f"global dedup leaves {name} below its clean floor")
            prefix = select_ordered_token_prefix(
                available,
                target_tokens=remaining,
                maximum_overshoot_tokens=int(
                    release["limits"]["maximum_category_overshoot_tokens"]
                ),
                partitions=partitions,
            )
            selected = prefix["selected"]
            selected_frames.append((selected, prefix))
            category_results.append(
                {
                    "available_documents": available_documents,
                    "available_tokens": available_tokens,
                    "category": name,
                    "deficit_tokens": 0,
                    "final_documents": int(category["accepted_documents"])
                    + int(prefix["selected_documents"]),
                    "final_tokens": int(category["accepted_tokens"])
                    + int(prefix["selected_tokens"]),
                    "floor_tokens": int(category["floor_tokens"]),
                    "selected_documents": int(prefix["selected_documents"]),
                    "selected_tokens": int(prefix["selected_tokens"]),
                }
            )
            available.unpersist()
        selected_all = selected_frames[0][0]
        for frame, _ in selected_frames[1:]:
            selected_all = selected_all.unionByName(frame)
        selected_all = selected_all.persist()
        selected_totals = selected_all.agg(
            F.count(F.lit(1)).alias("documents"),
            F.sum("token_count").alias("tokens"),
        ).first()
        prefix = f"{release['output']['production_prefix']}/{token}"
        accepted_key = f"{prefix}/accepted-candidates"
        rejected_key = f"{prefix}/rejected-ids"
        selected_all.write.mode("errorifexists").partitionBy("category").parquet(
            f"s3://{bucket}/{accepted_key}"
        )
        dedup["rejected_ids"].write.mode("errorifexists").parquet(
            f"s3://{bucket}/{rejected_key}"
        )
        accepted_inventory = _inventory(
            s3, bucket=bucket, prefix=f"{accepted_key}/"
        )
        rejected_inventory = _inventory(
            s3, bucket=bucket, prefix=f"{rejected_key}/"
        )
        output_bytes = int(accepted_inventory["object_bytes"]) + int(
            rejected_inventory["object_bytes"]
        )
        if output_bytes > int(release["limits"]["maximum_new_output_bytes"]):
            raise RuntimeError("Glue global-dedup exceeded its output storage cap")
        report = {
            "accepted_candidates": {
                **accepted_inventory,
                "documents": int(selected_totals["documents"] or 0),
                "s3_uri": f"s3://{bucket}/{accepted_key}",
                "tokens": int(selected_totals["tokens"] or 0),
            },
            "authorization_sha256": authorization["authorization_sha256"],
            "categories": category_results,
            "deduplication": dedup["metrics"],
            "gate": "lumi_glue_production_global_dedup_result_v0.1",
            "input_index_sha256": input_index["input_index_sha256"],
            "job_token": token,
            "map_checkpoint_sha256": input_index["map_checkpoint_sha256"],
            "mode": mode,
            "rejected_ids": {
                **rejected_inventory,
                "s3_uri": f"s3://{bucket}/{rejected_key}",
            },
            "release_fingerprint": release["release_fingerprint"],
            "runtime": runtime,
            "schema_version": 1,
            "status": "pass_category_floors",
            "total_deficit_tokens": 0,
        }
        selected_all.unpersist()
        for selected, _ in selected_frames:
            selected.unpersist()

    result_key = f"{release['output']['result_prefix']}/{token}/result.json"
    validate_glue_global_dedup_result(report, release, authorization)
    _put_immutable(report, bucket=bucket, key=result_key, s3_client=s3)
    print(json.dumps({**report, "result_key": result_key}, indent=2, sort_keys=True))
    accepted.unpersist()
    candidates.unpersist()
    anchor_frame.unpersist()
    job.commit()


if __name__ == "__main__":
    main()
