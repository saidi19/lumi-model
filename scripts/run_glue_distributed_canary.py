#!/usr/bin/env python3
"""Run the bounded Lumi distributed-corpus parity canary on AWS Glue."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spark_file_name(s3_uri: str) -> str:
    name = Path(urlparse(s3_uri).path).name
    if not name:
        raise ValueError("Glue canary artifact URI has no file name")
    return name


def _verified_spark_file(artifact: dict[str, Any]) -> Path:
    from pyspark import SparkFiles

    path = Path(SparkFiles.get(_spark_file_name(str(artifact["s3_uri"]))))
    if not path.is_file():
        raise FileNotFoundError(f"missing Glue canary artifact: {path.name}")
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"Glue canary artifact byte size changed: {path.name}")
    if _sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"Glue canary artifact SHA-256 changed: {path.name}")
    return path


def _map_source_partition(
    source_iterator: Any,
    runtime: dict[str, Any],
) -> Any:
    from pyspark import SparkFiles, TaskContext
    from tokenizers import Tokenizer

    from lumi.data_quality import DataQualityGate
    from lumi.distributed_corpus import (
        DistributedCandidateMapper,
        MinHashCodec,
        candidate_to_spark_row,
        load_training_exclusions,
    )
    from lumi.production_corpus import _download_source, _iter_gzip_records

    quality = json.loads(json.dumps(runtime["data_quality"]))
    quality["benchmark_decontamination"]["registry"] = SparkFiles.get(
        runtime["benchmark_file"]
    )
    codec = MinHashCodec.from_config(quality["near_duplicate"])
    tokenizer = Tokenizer.from_file(SparkFiles.get(runtime["tokenizer_file"]))
    exclusions = load_training_exclusions(Path(SparkFiles.get(runtime["exclusion_file"])))
    quality_gate = DataQualityGate(quality, Path(SparkFiles.getRootDirectory()))
    mapper = DistributedCandidateMapper(
        tokenizer=tokenizer,
        quality_gate=quality_gate,
        minhash_codec=codec,
        filters=runtime["filters"],
        excluded_document_ids=exclusions[0],
        excluded_content_sha256=exclusions[1],
        excluded_near_sha256=exclusions[2],
    )
    context = TaskContext.get()
    partition_id = -1 if context is None else context.partitionId()
    for source in source_iterator:
        scratch = Path(tempfile.mkdtemp(prefix=f"lumi-glue-{partition_id}-"))
        try:
            source_path = _download_source(
                source,
                scratch,
                int(source["source_bytes"]) + 16 * 1024 * 1024,
            )
            records = []
            processed = 0
            for line_number, record in _iter_gzip_records(source_path):
                if line_number < int(source["start_line"]):
                    continue
                if processed >= int(source["maximum_records"]):
                    break
                processed += 1
                records.append((line_number, record))
                if len(records) < 32:
                    continue
                batch = mapper.map_records(
                    source=source,
                    source_provenance=source["source_provenance"],
                    records=records,
                )
                for candidate in batch.candidates:
                    yield candidate_to_spark_row(candidate)
                records.clear()
            if records:
                batch = mapper.map_records(
                    source=source,
                    source_provenance=source["source_provenance"],
                    records=records,
                )
                for candidate in batch.candidates:
                    yield candidate_to_spark_row(candidate)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _candidate_from_spark(row: Any) -> dict[str, Any]:
    value = row.asDict(recursive=True)
    return {
        "category": value["category"],
        "component": value["component"],
        "content_sha256": value["content_sha256"],
        "dataset_id": value["dataset_id"],
        "dataset_revision": value["dataset_revision"],
        "document_id": value["document_id"],
        "lsh_buckets": [
            {"band": int(item["band"]), "bucket": bytes(item["bucket"])}
            for item in value["lsh_buckets"]
        ],
        "minhash_signature": bytes(value["minhash_signature"]),
        "provenance": json.loads(value["provenance_json"]),
        "quality": json.loads(value["quality_json"]),
        "schema_version": int(value["schema_version"]),
        "segment_index": int(value["segment_index"]),
        "source_file": value["source_file"],
        "source_line": int(value["source_line"]),
        "source_ordinal": int(value["source_ordinal"]),
        "source_sha256": value["source_sha256"],
        "text": value["text"],
        "text_utf8_bytes": int(value["text_utf8_bytes"]),
        "token_bytes": bytes(value["token_bytes"]),
        "token_count": int(value["token_count"]),
        "transformations": json.loads(value["transformations_json"]),
    }


def _anchor_from_spark(row: Any) -> dict[str, Any]:
    value = row.asDict(recursive=True)
    return {
        "content_sha256": value["content_sha256"],
        "document_id": value["document_id"],
        "lsh_buckets": [
            {"band": int(item["band"]), "bucket": bytes(item["bucket"])}
            for item in value["lsh_buckets"]
        ],
        "minhash_signature": bytes(value["minhash_signature"]),
        "schema_version": int(value["schema_version"]),
        "segment_index": int(value["segment_index"]),
        "source_line": int(value["source_line"]),
        "source_ordinal": int(value["source_ordinal"]),
    }


def _put_report(report: dict[str, Any], *, bucket: str, prefix: str) -> dict[str, Any]:
    import boto3

    body = json.dumps(report, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    key = f"{prefix}/{digest}.json"
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={"sha256": digest},
        ServerSideEncryption="AES256",
        IfNoneMatch="*",
    )
    return {"bytes": len(body), "key": key, "sha256": digest}


def main() -> None:
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark import SparkContext, SparkFiles
    from pyspark.sql import functions as F

    from lumi.distributed_baseline import anchor_from_json
    from lumi.distributed_corpus import MinHashCodec, deduplicate_candidates
    from lumi.glue_canary import (
        validate_glue_canary_authorization,
        validate_glue_canary_release,
    )
    from lumi.glue_corpus import (
        candidate_spark_schema,
        dedup_anchor_spark_schema,
        spark_deduplicate_candidates,
    )

    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "JOB_RUN_ID",
            "LUMI_AUTHORIZATION_FILE",
            "LUMI_RELEASE_FILE",
        ],
    )
    context = SparkContext.getOrCreate()
    glue_context = GlueContext(context)
    spark = glue_context.spark_session
    release_path = Path(SparkFiles.get(args["LUMI_RELEASE_FILE"]))
    authorization_path = Path(SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"]))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_glue_canary_release(release)
    validate_glue_canary_authorization(authorization, release)
    artifacts = release["artifacts"]
    tokenizer_path = _verified_spark_file(artifacts["frozen_tokenizer"])
    exclusion_path = _verified_spark_file(artifacts["training_exclusion_ledger"])
    benchmark_path = _verified_spark_file(artifacts["benchmark_registry"])
    anchor_path = _verified_spark_file(artifacts["resume_anchors"])

    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    started = time.monotonic()
    quality = release["data_quality"]
    codec = MinHashCodec.from_config(quality["near_duplicate"])
    anchors = []
    with gzip.open(anchor_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            anchors.append(anchor_from_json(json.loads(line), codec))
    anchor_frame = spark.createDataFrame(anchors, schema=dedup_anchor_spark_schema()).persist()
    if anchor_frame.count() != release["resume_baseline"]["accepted_documents"]:
        raise RuntimeError("Glue canary resume anchor count changed")

    runtime = {
        "benchmark_file": benchmark_path.name,
        "data_quality": quality,
        "exclusion_file": exclusion_path.name,
        "filters": release["filters"],
        "tokenizer_file": tokenizer_path.name,
    }
    source_rdd = context.parallelize(release["sources"], len(release["sources"]))
    candidate_rdd = source_rdd.mapPartitions(
        lambda iterator: _map_source_partition(iterator, runtime)
    )
    candidates = spark.createDataFrame(candidate_rdd, schema=candidate_spark_schema()).persist()
    candidate_documents = candidates.count()
    if candidate_documents == 0:
        raise RuntimeError("Glue canary produced no candidates")
    parsed = urlparse(artifacts["resume_anchors"]["s3_uri"])
    bucket = parsed.netloc
    run_prefix = (
        f"{release['output']['result_prefix']}/"
        f"{release['release_fingerprint']}/{args['JOB_RUN_ID']}"
    )
    checkpoint_uri = (
        f"s3://{bucket}/{release['output']['checkpoint_prefix']}/"
        f"{release['release_fingerprint']}/{args['JOB_RUN_ID']}"
    )
    candidates.write.mode("errorifexists").parquet(f"s3://{bucket}/{run_prefix}/candidates")
    result = spark_deduplicate_candidates(
        candidates,
        codec=codec,
        checkpoint_uri=checkpoint_uri,
        anchors=anchor_frame,
        maximum_rounds=release["deduplication"]["maximum_rounds"],
        maximum_lsh_bucket_documents=release["deduplication"][
            "maximum_lsh_bucket_documents"
        ],
        maximum_candidate_pairs=release["deduplication"]["maximum_candidate_pairs"],
    )
    result["accepted_ids"].write.mode("errorifexists").parquet(
        f"s3://{bucket}/{run_prefix}/accepted-ids"
    )
    result["rejected_ids"].write.mode("errorifexists").parquet(
        f"s3://{bucket}/{run_prefix}/rejected-ids"
    )

    order = ["source_ordinal", "source_line", "segment_index", "document_id"]
    candidate_values = [
        _candidate_from_spark(row)
        for row in candidates.orderBy(*[F.col(name).asc() for name in order]).collect()
    ]
    anchor_values = [
        _anchor_from_spark(row)
        for row in anchor_frame.orderBy(*[F.col(name).asc() for name in order]).collect()
    ]
    reference = deduplicate_candidates(candidate_values, codec, anchors=anchor_values)
    spark_accepted = {
        str(row["document_id"]) for row in result["accepted_ids"].select("document_id").collect()
    }
    spark_rejections = {
        str(row["document_id"]): "exact_duplicate"
        for row in result["exact_rejected"].select("document_id").collect()
    }
    spark_rejections.update(
        {
            str(row["document_id"]): "near_duplicate"
            for row in result["near_rejected"].select("document_id").collect()
        }
    )
    if spark_accepted != set(reference.accepted_document_ids):
        raise RuntimeError("Glue Spark accepted IDs differ from sequential reference")
    if spark_rejections != reference.rejection_reasons:
        raise RuntimeError("Glue Spark rejection reasons differ from sequential reference")
    token_row = candidates.agg(F.sum("token_count").alias("tokens")).first()
    report = {
        "authorization_sha256": _sha256_file(authorization_path),
        "candidate_documents": candidate_documents,
        "candidate_tokens": int(token_row["tokens"]),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gate": "lumi_distributed_glue_canary_result_v0.1",
        "job_name": args["JOB_NAME"],
        "job_run_id": args["JOB_RUN_ID"],
        "metrics": result["metrics"],
        "output_prefix": run_prefix,
        "parity": {
            "accepted_ids": "pass",
            "rejection_reasons": "pass",
            "sequential_reference": "pass",
        },
        "production_shards_written": False,
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "source_commit": release["source_commit"],
        "source_files": len(release["sources"]),
        "status": "pass",
    }
    report_object = _put_report(report, bucket=bucket, prefix=f"{run_prefix}/reports")
    print(json.dumps({**report, "report_object": report_object}, indent=2, sort_keys=True))
    candidates.unpersist()
    anchor_frame.unpersist()
    job.commit()


if __name__ == "__main__":
    main()
