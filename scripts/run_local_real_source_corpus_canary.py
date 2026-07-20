#!/usr/bin/env python3
"""Run Lumi's bounded real-source corpus parity canary on a local Mac."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from lumi.data_quality import DataQualityGate
from lumi.distributed_baseline import anchor_from_json
from lumi.distributed_corpus import (
    DistributedCandidateMapper,
    MinHashCodec,
    candidate_to_spark_row,
    deduplicate_candidates,
    load_training_exclusions,
)
from lumi.glue_corpus import (
    candidate_spark_schema,
    dedup_anchor_spark_schema,
    spark_deduplicate_candidates,
)
from lumi.production_corpus import _iter_gzip_records
from lumi.s3_staged_corpus import (
    _download_source,
    bind_staged_sources,
    validate_s3_staging_plan,
)

SOURCE_BYTES = 1_456_514_440
SOURCE_CAP = 1_500_000_000
DISK_CAP = 20_000_000_000
SOURCE_FILES = 8
RECORDS_PER_SOURCE = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--staging-plan", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-exclusion-ledger", type=Path, required=True)
    parser.add_argument("--benchmark-registry", type=Path, required=True)
    parser.add_argument("--resume-anchors", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--spark-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def log_event(path: Path, event: str, **fields: Any) -> None:
    row = {"event": event, "time_unix": round(time.time(), 3), **fields}
    rendered = canonical_bytes(row).decode()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(rendered, flush=True)


def verify_artifact(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.stat().st_size != int(expected["bytes"]):
        raise ValueError(f"{label} byte size changed")
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise ValueError(f"{label} SHA-256 changed")
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": digest}


def download_sources(
    sources: list[dict[str, Any]], raw: Path, log: Path
) -> dict[int, Path]:
    results: dict[int, Path] = {}

    def download(source: dict[str, Any]) -> tuple[int, Path]:
        return int(source["ordinal"]), _download_source(source, raw, SOURCE_CAP)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, source): source for source in sources}
        for future in as_completed(futures):
            source = futures[future]
            ordinal, path = future.result()
            results[ordinal] = path
            log_event(
                log,
                "source_verified",
                ordinal=ordinal,
                source_file=source["source_file"],
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
    if len(results) != SOURCE_FILES:
        raise RuntimeError("local canary did not verify all eight sources")
    return results


def map_candidates(
    release: Mapping[str, Any],
    paths: Mapping[int, Path],
    tokenizer_path: Path,
    exclusion_path: Path,
    benchmark_path: Path,
    repository_root: Path,
    log: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality = copy.deepcopy(release["data_quality"])
    quality["benchmark_decontamination"]["registry"] = str(benchmark_path.resolve())
    codec = MinHashCodec.from_config(quality["near_duplicate"])
    gate = DataQualityGate(quality, repository_root)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 65_536:
        raise ValueError("local canary tokenizer vocabulary changed")
    excluded = load_training_exclusions(exclusion_path)
    mapper = DistributedCandidateMapper(
        tokenizer=tokenizer,
        quality_gate=gate,
        minhash_codec=codec,
        filters=release["filters"],
        excluded_document_ids=excluded[0],
        excluded_content_sha256=excluded[1],
        excluded_near_sha256=excluded[2],
    )
    candidates: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    redactions: Counter[str] = Counter()
    raw_documents = 0
    raw_text_bytes = 0

    def flush(
        source: Mapping[str, Any],
        records: list[tuple[int, Mapping[str, Any] | None]],
    ) -> None:
        nonlocal raw_documents, raw_text_bytes
        if not records:
            return
        batch = mapper.map_records(
            source=source,
            source_provenance=source["source_provenance"],
            records=records,
        )
        candidates.extend(batch.candidates)
        rejections.update(batch.rejections)
        languages.update(batch.language_labels)
        redactions.update(batch.pii_redactions)
        raw_documents += batch.raw_documents
        raw_text_bytes += batch.raw_text_bytes
        records.clear()

    for source in sorted(release["sources"], key=lambda row: int(row["ordinal"])):
        records: list[tuple[int, Mapping[str, Any] | None]] = []
        processed = 0
        before = len(candidates)

        for line_number, record in _iter_gzip_records(paths[int(source["ordinal"])]):
            if line_number < int(source["start_line"]):
                continue
            if processed >= RECORDS_PER_SOURCE:
                break
            processed += 1
            records.append((line_number, record))
            if len(records) == 32:
                flush(source, records)
        flush(source, records)
        log_event(
            log,
            "source_mapped",
            ordinal=int(source["ordinal"]),
            records=processed,
            candidates=len(candidates) - before,
        )
    if not candidates:
        raise RuntimeError("local canary produced no candidates")
    return candidates, {
        "candidate_documents": len(candidates),
        "candidate_tokens": sum(int(row["token_count"]) for row in candidates),
        "language_labels": dict(sorted(languages.items())),
        "pii_redactions": dict(sorted(redactions.items())),
        "raw_documents": raw_documents,
        "raw_text_bytes": raw_text_bytes,
        "rejections": dict(sorted(rejections.items())),
    }


def load_anchors(
    path: Path, codec: MinHashCodec, expected_documents: int
) -> list[dict[str, Any]]:
    anchors = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            anchors.append(anchor_from_json(json.loads(line), codec))
    if len(anchors) != expected_documents:
        raise ValueError("local canary resume-anchor count changed")
    return anchors


def spark_parity(
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    reference: Any,
    release: Mapping[str, Any],
    output: Path,
    threads: int,
    log: Path,
) -> dict[str, Any]:
    if threads < 1 or threads > 8:
        raise ValueError("Spark thread count must be between one and eight")
    from pyspark.sql import SparkSession

    spark_local = output / "spark-local"
    spark_local.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder.master(f"local[{threads}]")
        .appName("lumi-local-real-source-corpus-canary-v0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.driver.memory", "4g")
        .config("spark.local.dir", str(spark_local))
        .getOrCreate()
    )
    try:
        codec = MinHashCodec.from_config(release["data_quality"]["near_duplicate"])
        candidate_rows = [candidate_to_spark_row(row) for row in candidates]
        candidate_partitions = min(max(threads * 8, 16), len(candidate_rows))
        anchor_partitions = min(max(threads * 16, 32), len(anchors))
        candidate_frame = spark.createDataFrame(
            spark.sparkContext.parallelize(candidate_rows, candidate_partitions),
            schema=candidate_spark_schema(),
        ).persist()
        anchor_frame = spark.createDataFrame(
            spark.sparkContext.parallelize(anchors, anchor_partitions),
            schema=dedup_anchor_spark_schema(),
        ).persist()
        log_event(
            log,
            "spark_input_partitioned",
            anchor_documents=len(anchors),
            anchor_partitions=anchor_partitions,
            candidate_documents=len(candidate_rows),
            candidate_partitions=candidate_partitions,
        )
        result = spark_deduplicate_candidates(
            candidate_frame,
            codec=codec,
            checkpoint_uri=(output / "spark-checkpoints").resolve().as_uri(),
            anchors=anchor_frame,
            allow_local_checkpoint=True,
            maximum_rounds=int(release["deduplication"]["maximum_rounds"]),
            maximum_lsh_bucket_documents=int(
                release["deduplication"]["maximum_lsh_bucket_documents"]
            ),
            maximum_candidate_pairs=int(
                release["deduplication"]["maximum_candidate_pairs"]
            ),
        )
        accepted = {
            str(row["document_id"])
            for row in result["accepted_ids"].select("document_id").collect()
        }
        rejections = {
            str(row["document_id"]): "exact_duplicate"
            for row in result["exact_rejected"].select("document_id").collect()
        }
        rejections.update(
            {
                str(row["document_id"]): "near_duplicate"
                for row in result["near_rejected"].select("document_id").collect()
            }
        )
        if accepted != set(reference.accepted_document_ids):
            raise RuntimeError("Spark accepted IDs differ from sequential reference")
        if rejections != reference.rejection_reasons:
            raise RuntimeError("Spark rejection reasons differ from sequential reference")
        parity_hash = hashlib.sha256(
            canonical_bytes(
                {"accepted": sorted(accepted), "rejections": rejections}
            )
        ).hexdigest()
        log_event(
            log,
            "spark_parity_passed",
            accepted_documents=len(accepted),
            rejected_documents=len(rejections),
            parity_sha256=parity_hash,
        )
        return {
            "accepted_documents": len(accepted),
            "metrics": result["metrics"],
            "parity_sha256": parity_hash,
            "rejected_documents": len(rejections),
            "status": "exact_match",
        }
    finally:
        spark.stop()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_manifest(output: Path) -> Path:
    members = [output / "report.json"]
    members.extend(sorted((output / "raw").glob("source-*.json.gz")))
    manifest = output / "manifest.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output)}\n" for path in members
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    if (output / "manifest.sha256").exists():
        raise FileExistsError("sealed local corpus canary already exists")
    if shutil.disk_usage(output).free < DISK_CAP:
        raise RuntimeError("local canary requires at least 20GB free")
    started = time.monotonic()
    log_event(log, "canary_started", source_cap=SOURCE_CAP, disk_cap=DISK_CAP)

    release = json.loads(args.release.read_text(encoding="utf-8"))
    plan = json.loads(args.staging_plan.read_text(encoding="utf-8"))
    verified_plan = validate_s3_staging_plan(plan)
    if verified_plan != {
        "plan_fingerprint": release["source_transport"]["plan_fingerprint"],
        "source_bytes": SOURCE_BYTES,
        "source_files": SOURCE_FILES,
        "status": "verified_local_no_cloud_mutation",
    }:
        raise ValueError("local canary source plan changed")
    if sha256_file(args.staging_plan) != release["source_transport"][
        "staging_plan_sha256"
    ]:
        raise ValueError("local canary staging-plan artifact changed")
    if any(release["authorization"].values()):
        raise ValueError("local release unexpectedly authorizes cloud mutation")
    if (
        len(release["sources"]) != SOURCE_FILES
        or sum(int(row["source_bytes"]) for row in release["sources"])
        != SOURCE_BYTES
        or any(
            int(row["maximum_records"]) != RECORDS_PER_SOURCE
            for row in release["sources"]
        )
    ):
        raise ValueError("local canary source envelope changed")
    if bind_staged_sources(release["sources"], plan) != release["sources"]:
        raise ValueError("local release sources differ from staging plan")

    artifacts = {
        "benchmark_registry": verify_artifact(
            args.benchmark_registry,
            release["artifacts"]["benchmark_registry"],
            "benchmark registry",
        ),
        "resume_anchors": verify_artifact(
            args.resume_anchors,
            release["artifacts"]["resume_anchors"],
            "resume anchors",
        ),
        "tokenizer": verify_artifact(
            args.tokenizer, release["artifacts"]["frozen_tokenizer"], "tokenizer"
        ),
        "training_exclusion_ledger": verify_artifact(
            args.training_exclusion_ledger,
            release["artifacts"]["training_exclusion_ledger"],
            "training exclusion ledger",
        ),
    }
    log_event(log, "control_artifacts_verified")

    paths = download_sources(
        [dict(row) for row in release["sources"]], output / "raw", log
    )
    if directory_bytes(output) > DISK_CAP:
        raise RuntimeError("local canary exceeded its disk cap after download")
    log_event(log, "all_sources_verified", bytes=directory_bytes(output / "raw"))

    candidates, mapping = map_candidates(
        release,
        paths,
        args.tokenizer,
        args.training_exclusion_ledger,
        args.benchmark_registry,
        repository_root,
        log,
    )
    codec = MinHashCodec.from_config(release["data_quality"]["near_duplicate"])
    anchors = load_anchors(
        args.resume_anchors,
        codec,
        int(release["resume_baseline"]["accepted_documents"]),
    )
    reference = deduplicate_candidates(candidates, codec, anchors=anchors)
    log_event(
        log,
        "sequential_reference_complete",
        accepted_documents=len(reference.accepted_document_ids),
        rejected_documents=len(reference.rejection_reasons),
        rounds=reference.rounds,
    )
    spark = spark_parity(
        candidates, anchors, reference, release, output, args.spark_threads, log
    )
    elapsed = round(time.monotonic() - started, 3)
    report = {
        "artifacts": artifacts,
        "authorization": {
            "cloud_mutation": False,
            "model_training": False,
            "paid_compute": False,
            "production_acquisition": False,
        },
        "elapsed_seconds": elapsed,
        "gate": "lumi_local_real_source_distributed_corpus_canary_v0.1",
        "mapping": mapping,
        "release": {
            "release_fingerprint": release["release_fingerprint"],
            "source_commit": release["source_commit"],
            "source_plan_fingerprint": release["source_plan_fingerprint"],
            "staging_plan_sha256": release["source_transport"][
                "staging_plan_sha256"
            ],
        },
        "runtime": {
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "datasketch",
                    "lingua-language-detector",
                    "numpy",
                    "pyspark",
                    "scipy",
                    "tokenizers",
                )
            },
            "platform": platform.platform(),
            "python": platform.python_version(),
            "spark_threads": args.spark_threads,
        },
        "schema_version": 1,
        "source_envelope": {
            "categories": sorted({str(row["category"]) for row in release["sources"]}),
            "compressed_bytes": SOURCE_BYTES,
            "maximum_records_per_source": RECORDS_PER_SOURCE,
            "source_files": SOURCE_FILES,
        },
        "spark_sequential_parity": spark,
        "status": "pass",
        "training_started": False,
    }
    write_json_atomic(output / "report.json", report)
    manifest = write_manifest(output)
    if directory_bytes(output) > DISK_CAP:
        raise RuntimeError("local canary exceeded its disk cap before seal")
    log_event(
        log,
        "canary_sealed",
        elapsed_seconds=elapsed,
        manifest_sha256=sha256_file(manifest),
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        detail = traceback.format_exc()
        try:
            failure_args = parse_args()
            failure_output = failure_args.output_directory.resolve()
            failure_output.mkdir(parents=True, exist_ok=True)
            (failure_output / "failure.log").write_text(detail, encoding="utf-8")
            log_event(
                failure_output / "run.log",
                "canary_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
        except BaseException:
            pass
        raise
