"""Shared exact Python 3.12 candidate mapping for Lumi EMR jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime() -> dict[str, Any]:
    expected = {
        "boto3": "1.43.51",
        "datasketch": "2.0.0",
        "lingua-language-detector": "2.2.0",
        "numpy": "2.2.6",
        "pyarrow": "25.0.0",
        "scipy": "1.16.3",
        "tokenizers": "0.22.2",
        "zstandard": "0.25.0",
    }
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError(f"production EMR requires Python 3.12, got {platform.python_version()}")
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"production EMR dependency versions changed: {versions}")
    return {"python": platform.python_version(), "versions": versions}


def verified_spark_file(artifact: dict[str, Any], key: str) -> Path:
    from pyspark import SparkFiles

    path = Path(SparkFiles.get(Path(key).name))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"production EMR artifact byte size changed: {path.name}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"production EMR artifact SHA-256 changed: {path.name}")
    return path


def _download_chunk(chunk: dict[str, Any], directory: Path, s3_client: Any) -> Path:
    path = directory / Path(str(chunk["s3_key"])).name
    with path.open("wb") as handle:
        s3_client.download_fileobj(str(chunk["bucket"]), str(chunk["s3_key"]), handle)
    if path.stat().st_size != int(chunk["bytes"]):
        raise ValueError("production source chunk byte size changed")
    if sha256_file(path) != str(chunk["sha256"]):
        raise ValueError("production source chunk SHA-256 changed")
    return path


def _iter_chunk_records(path: Path, chunk: dict[str, Any]) -> Any:
    import zstandard

    rows = 0
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for source_line, line in enumerate(text, start=int(chunk["start_line"])):
                    rows += 1
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        value = None
                    if value is not None and not isinstance(value, dict):
                        value = None
                    yield source_line, value
    if rows != int(chunk["rows"]):
        raise ValueError("production source chunk row count changed")
    if rows and int(chunk["start_line"]) + rows - 1 != int(chunk["end_line"]):
        raise ValueError("production source chunk line interval changed")


def map_chunk_partition(
    chunks: Any,
    runtime: dict[str, Any],
    source: dict[str, Any],
) -> Any:
    """Map a partition of immutable S3 chunks through every production filter."""

    from pyspark import SparkFiles, TaskContext
    from tokenizers import Tokenizer

    from .data_quality import DataQualityGate
    from .distributed_corpus import (
        DistributedCandidateMapper,
        MinHashCodec,
        candidate_to_spark_row,
        load_training_exclusions,
    )

    verify_runtime()
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

    s3_client = boto3.client("s3")
    context = TaskContext.get()
    partition = -1 if context is None else context.partitionId()
    scratch = Path(tempfile.mkdtemp(prefix=f"lumi-production-map-{partition}-"))
    try:
        for chunk in chunks:
            path = _download_chunk(chunk, scratch, s3_client)
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
            path.unlink()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
