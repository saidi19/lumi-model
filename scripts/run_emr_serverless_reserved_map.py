#!/usr/bin/env python3
"""Map one externally staged, reservation-bound Lumi source from private S3."""

from __future__ import annotations

import argparse
import base64
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
            raise ValueError("existing reserved-map result byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing reserved-map result SHA-256 changed") from error


def _list_output(s3_client: Any, *, bucket: str, prefix: str) -> dict[str, Any]:
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for row in page.get("Contents", []):
            key = str(row["Key"])
            if not key.endswith("/_SUCCESS"):
                objects.append({"bytes": int(row["Size"]), "key": key})
    if not objects:
        raise RuntimeError("reserved candidate-map Parquet output is empty")
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
    parser.add_argument("--lumi-staged-source-file", required=True)
    return parser.parse_args()


def main() -> None:
    import boto3
    from pyspark import SparkFiles
    from pyspark.sql import SparkSession

    from lumi.distributed_corpus import MinHashCodec
    from lumi.emr_candidate_mapping import (
        map_chunk_partition,
        verified_spark_file,
        verify_runtime,
    )
    from lumi.emr_production_map import (
        validate_map_checkpoint,
        validate_reserved_staged_source,
    )
    from lumi.glue_corpus import candidate_spark_schema, select_ordered_token_prefix

    args = _args()
    spark = SparkSession.builder.getOrCreate()
    plan = _read_json(Path(SparkFiles.get(args.lumi_plan_file)))
    checkpoint = _read_json(Path(SparkFiles.get(args.lumi_map_checkpoint_file)))
    authorization = _read_json(Path(SparkFiles.get(args.lumi_authorization_file)))
    staged = _read_json(Path(SparkFiles.get(args.lumi_staged_source_file)))
    validate_map_checkpoint(checkpoint, plan, authorization)
    validate_reserved_staged_source(
        staged,
        checkpoint,
        plan,
        authorization,
        reservation_token=args.lumi_job_token,
    )
    if authorization.get("run_token") != args.lumi_run_token:
        raise ValueError("reserved-map authorization token changed")
    reservation = next(
        row
        for row in checkpoint["active_reservations"]
        if row["reservation_token"] == args.lumi_job_token
    )
    cursor = reservation["source_cursor"]
    source = plan["source_schedule"][int(cursor["source_ordinal"])]
    manifest = staged["manifest"]
    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    for chunk in manifest["chunks"]:
        chunk["bucket"] = bucket

    runtime_report = verify_runtime()
    artifact_keys = plan["storage"]["artifacts"]
    tokenizer = verified_spark_file(
        plan["artifacts"]["frozen_tokenizer"], artifact_keys["frozen_tokenizer"]
    )
    exclusions = verified_spark_file(
        plan["artifacts"]["training_exclusion_ledger"],
        artifact_keys["training_exclusion_ledger"],
    )
    benchmark = verified_spark_file(
        plan["artifacts"]["benchmark_registry"], artifact_keys["benchmark_registry"]
    )
    quality = plan["data_quality"]
    MinHashCodec.from_config(quality["near_duplicate"])
    runtime = {
        "benchmark_file": benchmark.name,
        "data_quality": quality,
        "exclusion_file": exclusions.name,
        "filters": plan["filters"],
        "tokenizer_file": tokenizer.name,
    }
    candidate_rdd = spark.sparkContext.parallelize(
        manifest["chunks"], len(manifest["chunks"])
    ).mapPartitions(lambda rows: map_chunk_partition(rows, runtime, source))
    candidates = spark.createDataFrame(
        candidate_rdd, schema=candidate_spark_schema()
    ).persist()
    candidate_documents = candidates.count()
    remaining = int(reservation["remaining_target_tokens"])
    prefix = None
    if candidate_documents:
        prefix = select_ordered_token_prefix(
            candidates,
            target_tokens=remaining,
            maximum_overshoot_tokens=int(
                plan["limits"]["maximum_category_overshoot_tokens"]
            ),
            partitions=max(1, int(spark.conf.get("spark.sql.shuffle.partitions"))),
        )
        selected = prefix["selected"]
        selected_documents = int(prefix["selected_documents"])
        selected_tokens = int(prefix["selected_tokens"])
    else:
        selected = candidates
        selected_documents = 0
        selected_tokens = 0
    source_complete = bool(manifest["source_complete"]) and (
        selected_documents == candidate_documents
    )
    end_line = (
        int(manifest["chunks"][-1]["end_line"])
        if source_complete or prefix is None
        else int(prefix["cutoff"]["source_line"])
    )
    s3 = boto3.client("s3", region_name=str(plan["region"]))
    parquet_uri = None
    inventory = {"object_bytes": 0, "object_count": 0, "objects_sha256": None}
    if selected_documents:
        output_prefix = f"{namespace}/mapped/reservations/{args.lumi_job_token}/candidates"
        selected.write.mode("errorifexists").parquet(f"s3://{bucket}/{output_prefix}")
        inventory = _list_output(s3, bucket=bucket, prefix=f"{output_prefix}/")
        parquet_uri = f"s3://{bucket}/{output_prefix}"
    output = {
        "candidate_documents": selected_documents,
        "candidate_tokens": selected_tokens,
        "category": source["category"],
        "end_line": end_line,
        "object_bytes": inventory["object_bytes"],
        "object_count": inventory["object_count"],
        "objects_sha256": inventory["objects_sha256"],
        "parquet_s3_uri": parquet_uri,
        "plan_fingerprint": plan["plan_fingerprint"],
        "reservation_token": args.lumi_job_token,
        "source_complete": source_complete,
        "source_ordinal": cursor["source_ordinal"],
        "source_sha256": source["source_sha256"],
        "staged_manifest_s3_key": manifest["s3_key"],
        "staged_manifest_sha256": manifest["manifest_sha256"],
        "start_after_line": cursor["next_line"],
    }
    output["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(output))
    report = {
        "authorization_run_token": authorization["run_token"],
        "gate": "lumi_emr_production_reserved_map_result_v0.1",
        "job_token": args.lumi_job_token,
        "map_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "output": output,
        "plan_fingerprint": plan["plan_fingerprint"],
        "runtime": runtime_report,
        "schema_version": 1,
    }
    result_key = (
        f"{namespace}/control/distributed/production/map-runs/"
        f"{args.lumi_job_token}/result.json"
    )
    _put_immutable(report, bucket=bucket, key=result_key, s3_client=s3)
    print(json.dumps({**report, "result_key": result_key}, indent=2, sort_keys=True))
    if prefix is not None:
        prefix["selected"].unpersist()
    candidates.unpersist()


if __name__ == "__main__":
    main()
