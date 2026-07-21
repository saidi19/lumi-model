"""AWS Glue 5.1 / Python 3.11 adapter for immutable Lumi source chunks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .emr_candidate_mapping import _download_chunk, _iter_chunk_records, sha256_file


def verify_glue_mapping_runtime() -> dict[str, Any]:
    expected = {
        "datasketch": "2.0.0",
        "lingua-language-detector": "2.2.0",
        "tokenizers": "0.22.2",
        "zstandard": "0.25.0",
    }
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError(
            f"production Glue mapping requires Python 3.11, got {platform.python_version()}"
        )
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"production Glue mapping dependencies changed: {versions}")
    return {"python": platform.python_version(), "versions": versions}


def verified_spark_file(artifact: dict[str, Any]) -> Path:
    from pyspark import SparkFiles

    path = Path(SparkFiles.get(Path(str(artifact["s3_uri"])).name))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"production Glue mapping artifact size changed: {path.name}")
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"production Glue mapping artifact SHA-256 changed: {path.name}")
    return path


def map_staged_chunk_partition(items: Any, runtime: dict[str, Any]) -> Any:
    """Map mixed-category immutable chunks through the frozen production gates."""

    from pyspark import SparkFiles, TaskContext
    from tokenizers import Tokenizer

    from .data_quality import DataQualityGate
    from .distributed_corpus import (
        DistributedCandidateMapper,
        MinHashCodec,
        candidate_to_spark_row,
        load_training_exclusions,
    )

    verify_glue_mapping_runtime()
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
    import boto3

    s3 = boto3.client("s3", region_name=runtime["region"])
    context = TaskContext.get()
    partition = -1 if context is None else context.partitionId()
    scratch = Path(tempfile.mkdtemp(prefix=f"lumi-glue-map-{partition}-"))
    try:
        for item in items:
            source = item["source"]
            chunk = item["chunk"]
            if source["category"] != item["category"]:
                raise ValueError("Glue mapping work item category changed")
            path = _download_chunk(chunk, scratch, s3)
            try:
                records = []
                for source_line, value in _iter_chunk_records(path, chunk):
                    records.append((source_line, value))
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
                path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
