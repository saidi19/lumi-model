#!/usr/bin/env python3
"""Stream one Lumi category into immutable 100M-token production shards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import shutil
import tempfile
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
            raise ValueError("existing shard-writer result byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing shard-writer result SHA-256 changed") from error


def _candidate(row: Any) -> dict[str, Any]:
    value = row.asDict(recursive=True)
    value["lsh_buckets"] = [
        {"band": int(item["band"]), "bucket": bytes(item["bucket"])}
        for item in value["lsh_buckets"]
    ]
    value["minhash_signature"] = bytes(value["minhash_signature"])
    value["token_bytes"] = bytes(value["token_bytes"])
    return value


def _partition_by_shard(key: tuple[Any, ...]) -> int:
    return int(key[0])


def _seal_partition(rows: Any, config: Mapping[str, Any]) -> Any:
    """Seal sorted shard groups independently on Spark Python workers."""

    import boto3

    from lumi.emr_production_shards import (
        assemble_candidate_shards,
        publish_candidate_shard,
    )

    s3 = boto3.client("s3", region_name=str(config["region"]))
    for sequence, grouped in itertools.groupby(rows, key=lambda item: int(item[0][0])):
        scratch = Path(tempfile.mkdtemp(prefix=f"lumi-shard-{sequence:06d}-"))
        try:
            def publish(shard: dict[str, Any]) -> Mapping[str, Any]:
                return publish_candidate_shard(
                    shard,
                    bucket=str(config["bucket"]),
                    namespace=str(config["namespace"]),
                    s3_client=s3,
                )

            result = assemble_candidate_shards(
                (_candidate(item[1]) for item in grouped),
                directory=scratch / "shards",
                category=str(config["category"]),
                next_sequence=sequence,
                source_manifest_sha256=str(config["source_manifest_sha256"]),
                target_tokens=int(config["target_tokens"]),
                seal_final_partial=True,
                sealed_callback=publish,
            )
            if len(result["sealed_shards"]) != 1 or result["partial"] is not None:
                raise RuntimeError("distributed shard group did not seal exactly once")
            if result["next_sequence"] != sequence + 1:
                raise RuntimeError("distributed shard sequence changed")
            yield result["sealed_shards"][0]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumi-accepted-checkpoint-file", required=True)
    parser.add_argument("--lumi-authorization-file", required=True)
    parser.add_argument("--lumi-category", required=True)
    parser.add_argument("--lumi-dedup-result-file", required=True)
    parser.add_argument("--lumi-job-token", required=True)
    parser.add_argument("--lumi-plan-file", required=True)
    parser.add_argument("--lumi-run-token", required=True)
    return parser.parse_args()


def main() -> None:
    import boto3
    from pyspark import SparkFiles
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from lumi.emr_production_shard_writer import (
        shard_writer_job_token,
        validate_global_dedup_result,
    )
    from lumi.emr_production_shards import (
        assemble_candidate_shards,
        materialize_partial_shard,
        publish_candidate_shard,
    )
    from lumi.emr_serverless_production import (
        validate_paid_production_authorization,
        validate_production_checkpoint,
    )
    from lumi.glue_corpus import assign_ordered_token_shards, candidate_spark_schema

    args = _args()
    spark = SparkSession.builder.getOrCreate()
    plan = _read_json(Path(SparkFiles.get(args.lumi_plan_file)))
    accepted_checkpoint = _read_json(
        Path(SparkFiles.get(args.lumi_accepted_checkpoint_file))
    )
    authorization = _read_json(Path(SparkFiles.get(args.lumi_authorization_file)))
    dedup_result = _read_json(Path(SparkFiles.get(args.lumi_dedup_result_file)))
    validate_production_checkpoint(accepted_checkpoint, plan)
    validate_paid_production_authorization(authorization, plan, accepted_checkpoint)
    validate_global_dedup_result(dedup_result, plan, authorization)
    if authorization.get("run_token") != args.lumi_run_token:
        raise ValueError("shard-writer authorization token changed")
    if shard_writer_job_token(authorization, dedup_result, args.lumi_category) != (
        args.lumi_job_token
    ):
        raise ValueError("shard-writer job token changed")
    category_result = next(
        row for row in dedup_result["categories"] if row["category"] == args.lumi_category
    )
    accepted_uri = str(dedup_result["accepted_candidates"]["s3_uri"])
    frame = (
        spark.read.schema(candidate_spark_schema())
        .parquet(accepted_uri)
        .where(F.col("category") == F.lit(args.lumi_category))
        .persist()
    )
    totals = frame.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    selected_documents = int(totals["documents"] or 0)
    selected_tokens = int(totals["tokens"] or 0)
    if selected_documents != int(category_result["selected_documents"]):
        raise RuntimeError("shard-writer category document total changed")
    if selected_tokens != int(category_result["selected_tokens"]):
        raise RuntimeError("shard-writer category token total changed")
    if selected_documents == 0 or selected_tokens == 0:
        raise RuntimeError("shard-writer category has no selected candidates")
    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    s3 = boto3.client("s3", region_name=str(plan["region"]))
    scratch = Path(tempfile.mkdtemp(prefix=f"lumi-partial-{args.lumi_category}-"))
    try:
        partial = accepted_checkpoint.get("partial_shard")
        local_partial = None
        if isinstance(partial, Mapping) and partial.get("category") == args.lumi_category:
            local_partial = materialize_partial_shard(
                partial,
                directory=scratch / "partial-input",
                bucket=bucket,
                s3_client=s3,
            )
        def publish(shard: dict[str, Any]) -> Mapping[str, Any]:
            return publish_candidate_shard(
                shard,
                bucket=bucket,
                namespace=namespace,
                s3_client=s3,
            )

        carried_tokens = 0 if local_partial is None else int(local_partial["tokens"])
        carried_documents = 0 if local_partial is None else int(local_partial["documents"])
        carried_shards = []
        start_sequence = 0
        if local_partial is not None:
            carried = assemble_candidate_shards(
                [],
                directory=scratch / "carried-partial",
                category=args.lumi_category,
                next_sequence=int(local_partial["sequence"]),
                source_manifest_sha256=plan["inputs"]["source_file_manifest"]["sha256"],
                target_tokens=int(plan["limits"]["clean_shard_target_tokens"]),
                partial=local_partial,
                seal_final_partial=True,
                sealed_callback=publish,
            )
            if len(carried["sealed_shards"]) != 1 or carried["partial"] is not None:
                raise RuntimeError("baseline partial did not seal exactly once")
            carried_shards = carried["sealed_shards"]
            start_sequence = int(carried["next_sequence"])
        partitions = max(1, int(spark.conf.get("spark.sql.shuffle.partitions")))
        assignment = assign_ordered_token_shards(
            frame,
            target_tokens=int(plan["limits"]["clean_shard_target_tokens"]),
            maximum_overshoot_tokens=int(
                plan["limits"]["maximum_category_overshoot_tokens"]
            ),
            start_sequence=start_sequence,
            partitions=partitions,
        )
        order = ["source_ordinal", "source_line", "segment_index", "document_id"]
        keyed = assignment["assigned"].rdd.map(
            lambda row: (
                (
                    int(row["shard_sequence"]),
                    *[row[name] for name in order],
                ),
                row,
            )
        )
        config = {
            "bucket": bucket,
            "category": args.lumi_category,
            "namespace": namespace,
            "region": plan["region"],
            "source_manifest_sha256": plan["inputs"]["source_file_manifest"]["sha256"],
            "target_tokens": int(plan["limits"]["clean_shard_target_tokens"]),
        }
        distributed_shards = (
            keyed.repartitionAndSortWithinPartitions(
                numPartitions=partitions,
                partitionFunc=_partition_by_shard,
            )
            .mapPartitions(lambda rows: _seal_partition(rows, config))
            .collect()
        )
        assignment["assigned"].unpersist()
        distributed_shards.sort(key=lambda row: int(row["sequence"]))
        shards = [*carried_shards, *distributed_shards]
        shard_tokens = sum(int(row["tokens"]) for row in shards)
        shard_documents = sum(int(row["documents"]) for row in shards)
        if shard_tokens != selected_tokens + carried_tokens:
            raise RuntimeError("shard-writer published token conservation failed")
        if shard_documents != selected_documents + carried_documents:
            raise RuntimeError("shard-writer published document conservation failed")
        first_sequence = int(shards[0]["sequence"])
        expected_sequences = list(range(first_sequence, first_sequence + len(shards)))
        if [int(row["sequence"]) for row in shards] != expected_sequences:
            raise RuntimeError("shard-writer published sequence is not contiguous")
        next_sequence = expected_sequences[-1] + 1
        report = {
            "authorization_run_token": authorization["run_token"],
            "carried_partial_documents": carried_documents,
            "carried_partial_tokens": carried_tokens,
            "category": args.lumi_category,
            "dedup_result_sha256": hashlib.sha256(
                _canonical_json_bytes(dedup_result)
            ).hexdigest(),
            "gate": "lumi_emr_production_shard_writer_result_v0.1",
            "job_token": args.lumi_job_token,
            "next_sequence": next_sequence,
            "plan_fingerprint": plan["plan_fingerprint"],
            "schema_version": 1,
            "selected_documents": selected_documents,
            "selected_tokens": selected_tokens,
            "shards": shards,
            "status": "complete",
        }
        result_key = (
            f"{namespace}/control/distributed/production/shard-runs/"
            f"{args.lumi_job_token}/result.json"
        )
        _put_immutable(report, bucket=bucket, key=result_key, s3_client=s3)
        print(json.dumps({**report, "result_key": result_key}, indent=2, sort_keys=True))
    finally:
        frame.unpersist()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
