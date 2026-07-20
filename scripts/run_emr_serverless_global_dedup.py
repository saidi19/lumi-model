#!/usr/bin/env python3
"""Globally deduplicate sealed Lumi candidate-map outputs on EMR Serverless."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise RuntimeError("global-dedup Parquet output is empty")
    return {
        "object_bytes": sum(row["bytes"] for row in objects),
        "object_count": len(objects),
        "objects_sha256": _sha256_bytes(_canonical_json_bytes(objects)),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumi-authorization-file", required=True)
    parser.add_argument("--lumi-job-token", required=True)
    parser.add_argument("--lumi-map-checkpoint-file", required=True)
    parser.add_argument("--lumi-plan-file", required=True)
    parser.add_argument("--lumi-run-token", required=True)
    return parser.parse_args()


def main() -> None:
    import boto3
    from pyspark import SparkFiles
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from lumi.distributed_baseline import anchor_from_json
    from lumi.distributed_corpus import MinHashCodec
    from lumi.emr_candidate_mapping import verified_spark_file, verify_runtime
    from lumi.emr_production_dedup import dedup_job_token
    from lumi.emr_production_map import validate_map_checkpoint
    from lumi.glue_corpus import (
        candidate_spark_schema,
        dedup_anchor_spark_schema,
        select_ordered_token_prefix,
        spark_deduplicate_candidates,
    )

    args = _args()
    spark = SparkSession.builder.getOrCreate()
    plan = _read_json(Path(SparkFiles.get(args.lumi_plan_file)))
    map_checkpoint = _read_json(Path(SparkFiles.get(args.lumi_map_checkpoint_file)))
    authorization = _read_json(Path(SparkFiles.get(args.lumi_authorization_file)))
    validate_map_checkpoint(map_checkpoint, plan, authorization)
    if map_checkpoint["status"] != "ready_for_global_dedup":
        raise ValueError("production map checkpoint is not ready for global dedup")
    if authorization.get("run_token") != args.lumi_run_token:
        raise ValueError("global-dedup authorization token changed")
    if dedup_job_token(authorization, map_checkpoint) != args.lumi_job_token:
        raise ValueError("global-dedup job token changed")
    runtime = verify_runtime()
    anchor_key = plan["storage"]["artifacts"]["initial_anchors"]
    anchor_path = verified_spark_file(plan["artifacts"]["initial_anchors"], anchor_key)
    codec = MinHashCodec.from_config(plan["data_quality"]["near_duplicate"])
    anchors = []
    with gzip.open(anchor_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            anchors.append(anchor_from_json(json.loads(line), codec))
    anchor_frame = spark.createDataFrame(
        anchors, schema=dedup_anchor_spark_schema()
    ).persist()
    if anchor_frame.count() != int(plan["resume"]["accepted_documents"]):
        raise RuntimeError("global-dedup resume-anchor count changed")
    candidate_uris = [
        str(row["parquet_s3_uri"])
        for row in map_checkpoint["outputs"]
        if int(row["candidate_tokens"]) > 0
    ]
    if not candidate_uris:
        raise RuntimeError("global dedup has no non-empty map outputs")
    candidates = spark.read.schema(candidate_spark_schema()).parquet(*candidate_uris).persist()
    candidate_totals = candidates.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    if int(candidate_totals["documents"] or 0) != int(
        map_checkpoint["total_mapped_documents"]
    ):
        raise RuntimeError("global-dedup candidate document total changed")
    if int(candidate_totals["tokens"] or 0) != int(map_checkpoint["total_mapped_tokens"]):
        raise RuntimeError("global-dedup candidate token total changed")
    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    job_token = args.lumi_job_token
    checkpoint_uri = (
        f"s3://{bucket}/{namespace}/control/distributed/production/dedup-checkpoints/"
        f"{map_checkpoint['checkpoint_sha256']}/{job_token}"
    )
    dedup = spark_deduplicate_candidates(
        candidates,
        codec=codec,
        checkpoint_uri=checkpoint_uri,
        anchors=anchor_frame,
        maximum_rounds=256,
        maximum_lsh_bucket_documents=10_000,
        maximum_candidate_pairs=50_000_000,
    )
    accepted = dedup["accepted"].persist()
    selected_frames = []
    category_results = []
    partitions = max(1, int(spark.conf.get("spark.sql.shuffle.partitions")))
    for category in plan["categories"]:
        name = str(category["category"])
        remaining = int(category["floor_tokens"]) - int(category["accepted_tokens"])
        available = accepted.where(F.col("category") == F.lit(name)).persist()
        totals = available.agg(
            F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
        ).first()
        available_documents = int(totals["documents"] or 0)
        available_tokens = int(totals["tokens"] or 0)
        if available_documents:
            prefix = select_ordered_token_prefix(
                available,
                target_tokens=remaining,
                maximum_overshoot_tokens=int(
                    plan["limits"]["maximum_category_overshoot_tokens"]
                ),
                partitions=partitions,
            )
            selected = prefix["selected"]
            selected_documents = int(prefix["selected_documents"])
            selected_tokens = int(prefix["selected_tokens"])
        else:
            prefix = None
            selected = available.limit(0)
            selected_documents = 0
            selected_tokens = 0
        selected_frames.append((selected, prefix))
        category_results.append(
            {
                "available_documents": available_documents,
                "available_tokens": available_tokens,
                "category": name,
                "deficit_tokens": max(0, remaining - selected_tokens),
                "final_documents": int(category["accepted_documents"])
                + selected_documents,
                "final_tokens": int(category["accepted_tokens"]) + selected_tokens,
                "floor_tokens": int(category["floor_tokens"]),
                "selected_documents": selected_documents,
                "selected_tokens": selected_tokens,
            }
        )
        available.unpersist()
    selected_all = selected_frames[0][0]
    for frame, _ in selected_frames[1:]:
        selected_all = selected_all.unionByName(frame)
    selected_all = selected_all.persist()
    selected_totals = selected_all.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    output_prefix = (
        f"{namespace}/deduplicated/{map_checkpoint['checkpoint_sha256']}/{job_token}"
    )
    accepted_prefix = f"{output_prefix}/accepted-candidates"
    rejected_prefix = f"{output_prefix}/rejected-ids"
    selected_all.write.mode("errorifexists").partitionBy("category").parquet(
        f"s3://{bucket}/{accepted_prefix}"
    )
    dedup["rejected_ids"].write.mode("errorifexists").parquet(
        f"s3://{bucket}/{rejected_prefix}"
    )
    s3 = boto3.client("s3", region_name=str(plan["region"]))
    accepted_inventory = _inventory(s3, bucket=bucket, prefix=f"{accepted_prefix}/")
    rejected_inventory = _inventory(s3, bucket=bucket, prefix=f"{rejected_prefix}/")
    deficits = sum(int(row["deficit_tokens"]) for row in category_results)
    report = {
        "accepted_candidates": {
            **accepted_inventory,
            "documents": int(selected_totals["documents"] or 0),
            "s3_uri": f"s3://{bucket}/{accepted_prefix}",
            "tokens": int(selected_totals["tokens"] or 0),
        },
        "authorization_run_token": authorization["run_token"],
        "categories": category_results,
        "deduplication": dedup["metrics"],
        "gate": "lumi_emr_production_global_dedup_result_v0.1",
        "job_token": job_token,
        "map_checkpoint_sha256": map_checkpoint["checkpoint_sha256"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "rejected_ids": {
            **rejected_inventory,
            "s3_uri": f"s3://{bucket}/{rejected_prefix}",
        },
        "runtime": runtime,
        "schema_version": 1,
        "status": "pass_category_floors" if deficits == 0 else "needs_more_mapping",
        "total_deficit_tokens": deficits,
    }
    result_key = (
        f"{namespace}/control/distributed/production/dedup-runs/{job_token}/result.json"
    )
    _put_immutable(report, bucket=bucket, key=result_key, s3_client=s3)
    print(json.dumps({**report, "result_key": result_key}, indent=2, sort_keys=True))
    selected_all.unpersist()
    for selected, prefix in selected_frames:
        if prefix is not None:
            selected.unpersist()
    accepted.unpersist()
    candidates.unpersist()
    anchor_frame.unpersist()


if __name__ == "__main__":
    main()
