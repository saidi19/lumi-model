#!/usr/bin/env python3
"""Run one checkpointed, hard-capped Lumi production mapping job on Glue."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GLUE_REQUESTED_OPTIONS = (
    "JOB_NAME",
    "LUMI_AUTHORIZATION_FILE",
    "LUMI_CHECKPOINT_FILE",
    "LUMI_RELEASE_FILE",
)


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
            raise ValueError("existing Glue mapping control object size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing Glue mapping control object hash changed") from error
        version_id = head.get("VersionId")
    else:
        version_id = response.get("VersionId")
    return {
        "bytes": len(body),
        "key": key,
        "sha256": digest,
        "version_id": None if version_id is None else str(version_id),
    }


def _list_parquet_output(
    s3_client: Any, *, bucket: str, prefix: str
) -> dict[str, Any]:
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for row in page.get("Contents", []):
            key = str(row["Key"])
            if key.endswith("/_SUCCESS"):
                continue
            objects.append(
                {
                    "bytes": int(row["Size"]),
                    "etag": str(row.get("ETag", "")).strip('"'),
                    "key": key,
                }
            )
    objects.sort(key=lambda row: row["key"])
    if not objects:
        raise RuntimeError("Glue production mapping Parquet output is empty")
    return {
        "object_bytes": sum(row["bytes"] for row in objects),
        "object_count": len(objects),
        "objects_sha256": _sha256_bytes(_canonical_json_bytes(objects)),
    }


def _load_stage_queues(
    release: dict[str, Any], *, s3_client: Any
) -> dict[str, list[dict[str, Any]]]:
    from lumi.emr_production_staging import validate_published_staged_source

    bucket = str(release["storage"]["bucket"])
    schedule = {int(row["ordinal"]): row for row in release["source_schedule"]}
    queues = {row["category"]: [] for row in release["categories"]}
    for staged in release["staged_sources"]:
        key = str(staged["manifest_s3_key"])
        request: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if staged.get("manifest_version_id"):
            request["VersionId"] = str(staged["manifest_version_id"])
        body = s3_client.get_object(**request)["Body"].read()
        if _sha256_bytes(body) != staged["manifest_sha256"]:
            raise ValueError("Glue production staged manifest SHA-256 changed")
        manifest = json.loads(body)
        if not isinstance(manifest, dict):
            raise ValueError("Glue production staged manifest is not an object")
        manifest["manifest_sha256"] = staged["manifest_sha256"]
        manifest["s3_key"] = key
        manifest["version_id"] = staged.get("manifest_version_id")
        prefix = key.split("/manifests/", 1)[0]
        validate_published_staged_source(
            manifest,
            bucket=bucket,
            key_prefix=prefix,
            hard_transient_bytes=100_000_000_000,
        )
        ordinal = int(staged["source_ordinal"])
        source = schedule[ordinal]
        if (
            manifest["source_sha256"] != source["source_sha256"]
            or int(manifest["start_after_line"]) != int(staged["start_after_line"])
        ):
            raise ValueError("Glue production staged manifest source identity changed")
        for chunk in manifest["chunks"]:
            queues[source["category"]].append(
                {
                    "category": source["category"],
                    "chunk": {**chunk, "bucket": bucket},
                    "source": source,
                }
            )
    for row in release["category_inventory"]:
        name = row["category"]
        queues[name].sort(
            key=lambda item: (
                int(item["source"]["ordinal"]),
                int(item["chunk"]["start_line"]),
                str(item["chunk"]["sha256"]),
            )
        )
        if len(queues[name]) != int(row["staged_chunks"]):
            raise ValueError("Glue production staged category chunk count changed")
    return queues


def _wave_manifest(
    *,
    release: dict[str, Any],
    checkpoint: dict[str, Any],
    plan: dict[str, Any],
    results: list[dict[str, Any]],
    manifest_key: str,
) -> dict[str, Any]:
    manifest = {
        "candidate_documents": sum(row["candidate_documents"] for row in results),
        "candidate_tokens": sum(row["candidate_tokens"] for row in results),
        "categories": results,
        "gate": "lumi_glue_production_map_wave_result_v0.1",
        "manifest_s3_key": manifest_key,
        "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parquet_bytes": sum(row["parquet_bytes"] for row in results),
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "wave_plan_sha256": plan["wave_sha256"],
        "wave_sequence": plan["wave_sequence"],
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    import boto3
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles
    from pyspark.sql import functions as F

    from lumi.glue_candidate_mapping import (
        map_staged_chunk_partition,
        verified_spark_file,
        verify_glue_mapping_runtime,
    )
    from lumi.glue_corpus import (
        candidate_spark_schema,
        select_ordered_token_prefix_at_most,
    )
    from lumi.glue_production_mapping import (
        advance_mapping_checkpoint,
        plan_mapping_wave,
        validate_glue_production_mapping_authorization,
        validate_glue_production_mapping_release,
        validate_mapping_checkpoint,
    )

    args = getResolvedOptions(sys.argv, list(GLUE_REQUESTED_OPTIONS))
    if not args.get("JOB_RUN_ID"):
        raise RuntimeError("AWS Glue did not inject its reserved JOB_RUN_ID")
    context = SparkContext.getOrCreate()
    glue_context = GlueContext(context)
    spark = glue_context.spark_session
    release = _read_json(Path(SparkFiles.get(args["LUMI_RELEASE_FILE"])))
    authorization = _read_json(Path(SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"])))
    checkpoint = _read_json(Path(SparkFiles.get(args["LUMI_CHECKPOINT_FILE"])))
    validate_glue_production_mapping_release(release)
    validate_glue_production_mapping_authorization(
        authorization, release, checkpoint=checkpoint
    )
    validate_mapping_checkpoint(checkpoint, release)
    if checkpoint["checkpoint_sha256"] != authorization["bounds"][
        "starting_checkpoint_sha256"
    ]:
        raise ValueError("authorized Glue mapping run started from another checkpoint")
    runtime_report = verify_glue_mapping_runtime()
    artifacts = release["artifacts"]
    tokenizer = verified_spark_file(artifacts["frozen_tokenizer"])
    exclusions = verified_spark_file(artifacts["training_exclusion_ledger"])
    benchmark = verified_spark_file(artifacts["benchmark_registry"])
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    started = time.monotonic()
    bucket = str(release["storage"]["bucket"])
    region = str(release["region"])
    s3 = boto3.client("s3", region_name=region)
    queues = _load_stage_queues(release, s3_client=s3)
    runtime = {
        "benchmark_file": benchmark.name,
        "data_quality": release["data_quality"],
        "exclusion_file": exclusions.name,
        "filters": release["filters"],
        "region": region,
        "tokenizer_file": tokenizer.name,
    }
    run_base = (
        f"{release['output']['result_prefix']}/{release['release_fingerprint']}/"
        f"{args['JOB_RUN_ID']}"
    )
    checkpoint_object = None
    waves_completed = 0
    while checkpoint["status"] == "mapping":
        elapsed_minutes = (time.monotonic() - started) / 60
        if elapsed_minutes >= int(release["job"]["soft_stop_minutes"]):
            break
        wave = plan_mapping_wave(checkpoint, release)
        work_items = []
        for row in wave["categories"]:
            work_items.extend(
                queues[row["category"]][
                    int(row["start_chunk_index"]) : int(row["end_chunk_index"])
                ]
            )
        if len(work_items) != int(wave["chunk_count"]):
            raise RuntimeError("Glue production mapping wave input count changed")
        rdd = context.parallelize(work_items, len(work_items)).mapPartitions(
            lambda items: map_staged_chunk_partition(items, runtime)
        )
        candidates = spark.createDataFrame(rdd, schema=candidate_spark_schema()).persist()
        candidates.count()
        wave_results = []
        for planned in wave["categories"]:
            name = planned["category"]
            frame = candidates.where(F.col("category") == F.lit(name))
            totals = frame.agg(
                F.count(F.lit(1)).alias("documents"),
                F.sum("token_count").alias("tokens"),
            ).first()
            available_documents = int(totals["documents"] or 0)
            available_tokens = int(totals["tokens"] or 0)
            remaining = int(planned["remaining_target_tokens"])
            selected = frame
            selected_documents = available_documents
            selected_tokens = available_tokens
            cutoff = None
            closed = False
            selected_is_persisted = False
            if available_tokens >= remaining and available_documents:
                prefix = select_ordered_token_prefix_at_most(
                    frame,
                    maximum_tokens=remaining,
                    partitions=max(1, int(spark.conf.get("spark.sql.shuffle.partitions"))),
                )
                selected = prefix["selected"]
                selected_is_persisted = True
                selected_documents = int(prefix["selected_documents"])
                selected_tokens = int(prefix["selected_tokens"])
                cutoff = prefix["cutoff"]
                closed = True
            output_uri = None
            inventory = {
                "object_bytes": 0,
                "object_count": 0,
                "objects_sha256": None,
            }
            if selected_documents:
                output_prefix = (
                    f"{release['output']['mapped_prefix']}/"
                    f"{release['release_fingerprint']}/{args['JOB_RUN_ID']}/"
                    f"wave-{int(wave['wave_sequence']):06d}/{name}"
                )
                selected.write.mode("errorifexists").parquet(
                    f"s3://{bucket}/{output_prefix}"
                )
                inventory = _list_parquet_output(
                    s3, bucket=bucket, prefix=f"{output_prefix}/"
                )
                output_uri = f"s3://{bucket}/{output_prefix}"
            wave_results.append(
                {
                    "available_documents": available_documents,
                    "available_tokens": available_tokens,
                    "candidate_documents": selected_documents,
                    "candidate_tokens": selected_tokens,
                    "category": name,
                    "closed": closed,
                    "cutoff": cutoff,
                    "end_chunk_index": int(planned["end_chunk_index"]),
                    "object_count": int(inventory["object_count"]),
                    "objects_sha256": inventory["objects_sha256"],
                    "parquet_bytes": int(inventory["object_bytes"]),
                    "parquet_s3_uri": output_uri,
                    "start_chunk_index": int(planned["start_chunk_index"]),
                }
            )
            if selected_is_persisted:
                selected.unpersist()
        manifest_key = (
            f"{run_base}/waves/wave-{int(wave['wave_sequence']):06d}/manifest.json"
        )
        manifest = _wave_manifest(
            release=release,
            checkpoint=checkpoint,
            plan=wave,
            results=wave_results,
            manifest_key=manifest_key,
        )
        _put_immutable(manifest, bucket=bucket, key=manifest_key, s3_client=s3)
        checkpoint = advance_mapping_checkpoint(checkpoint, release, manifest)
        checkpoint_key = (
            f"{release['output']['checkpoint_prefix']}/"
            f"{release['release_fingerprint']}/{checkpoint['checkpoint_sha256']}.json"
        )
        checkpoint_object = _put_immutable(
            checkpoint, bucket=bucket, key=checkpoint_key, s3_client=s3
        )
        waves_completed += 1
        print(
            "lumi_map_checkpoint "
            + json.dumps(
                {
                    "checkpoint_key": checkpoint_key,
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "mapped_tokens": checkpoint["total_mapped_tokens"],
                    "status": checkpoint["status"],
                    "wave": checkpoint["checkpoint_sequence"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        candidates.unpersist()
    complete = checkpoint["status"] == "ready_for_global_dedup"
    report = {
        "authorization_sha256": authorization["authorization_sha256"],
        "checkpoint_object": checkpoint_object,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "final_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "gate": "lumi_glue_production_map_run_result_v0.1",
        "job_name": args["JOB_NAME"],
        "job_run_id": args["JOB_RUN_ID"],
        "mapped_documents": checkpoint["total_mapped_documents"],
        "mapped_parquet_bytes": checkpoint["mapped_parquet_bytes"],
        "mapped_tokens": checkpoint["total_mapped_tokens"],
        "model_training_started": False,
        "release_fingerprint": release["release_fingerprint"],
        "runtime": runtime_report,
        "schema_version": 1,
        "status": "pass_ready_for_global_dedup" if complete else "paused_soft_deadline",
        "total_candidate_tokens": int(release["resume"]["accepted_tokens"])
        + int(checkpoint["total_mapped_tokens"]),
        "waves_completed": waves_completed,
    }
    report_key = f"{run_base}/map-run-result.json"
    report_object = _put_immutable(report, bucket=bucket, key=report_key, s3_client=s3)
    print(json.dumps({**report, "report_object": report_object}, indent=2, sort_keys=True))
    job.commit()


if __name__ == "__main__":
    main()
