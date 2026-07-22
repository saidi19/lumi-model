from __future__ import annotations

import hashlib
import json
import random
import runpy
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from lumi.data_quality import QualityEvaluation
from lumi.distributed_baseline import (
    _quality_fingerprint,
    export_resume_anchors,
    verify_resume_anchor_export,
)
from lumi.distributed_corpus import (
    CANDIDATE_SCHEMA_VERSION,
    DEDUP_ANCHOR_SCHEMA_VERSION,
    DistributedCandidateMapper,
    MinHashCodec,
    build_distributed_corpus_plan,
    candidate_to_anchor,
    candidate_to_spark_row,
    deduplicate_candidates,
    parallel_ordered_greedy,
    sequential_ordered_greedy,
)
from lumi.glue_canary import (
    CANARY_APPROVAL_TEXT,
    _select_canary_sources,
    validate_glue_canary_authorization,
    validate_glue_canary_release,
)
from lumi.production_corpus import build_round_robin_source_schedule

ROOT = Path(__file__).parents[1]
QUALITY = json.loads((ROOT / "configs" / "data-quality-v0.1.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(codec: MinHashCodec, value: int) -> tuple[bytes, SimpleNamespace]:
    dtype = np.uint32 if codec.width == 4 else np.uint64
    hashvalues = np.full(codec.num_perm, value, dtype=dtype)
    signature = SimpleNamespace(hashvalues=hashvalues)
    return codec.signature_bytes(signature), signature


def _candidate(
    codec: MinHashCodec,
    document_id: str,
    ordinal: int,
    text: str,
    signature_value: int,
) -> dict[str, object]:
    signature_bytes, signature = _signature(codec, signature_value)
    return {
        "category": "science_and_education",
        "component": "fixture",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "dataset_id": "fixture/dataset",
        "dataset_revision": "b" * 40,
        "document_id": document_id,
        "lsh_buckets": codec.buckets(signature),
        "minhash_signature": signature_bytes,
        "provenance": {},
        "quality": {
            "near_duplicate": {"fingerprint_sha256": codec.fingerprint(signature_bytes)}
        },
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "segment_index": 0,
        "source_file": "fixture.json.gz",
        "source_line": ordinal + 1,
        "source_ordinal": 0,
        "source_sha256": "a" * 64,
        "text": text,
        "text_utf8_bytes": len(text.encode()),
        "token_bytes": b"\x01\x00",
        "token_count": 1,
        "transformations": {},
    }


def test_parallel_ordered_greedy_matches_sequential_for_chains_and_random_graphs() -> None:
    order = {name: (0, index, 0, name) for index, name in enumerate("abcdef")}
    chain = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"))
    accepted, rejected, rounds = parallel_ordered_greedy(order, chain)
    assert accepted == ("a", "c", "e")
    assert rejected == {"b", "d", "f"}
    assert rounds == 3
    assert accepted == sequential_ordered_greedy(order, chain)

    generator = random.Random(314159)
    for _ in range(100):
        names = [f"d{index:03d}" for index in range(30)]
        priorities = {name: (0, index, 0, name) for index, name in enumerate(names)}
        edges = [
            (left, right)
            for left_index, left in enumerate(names)
            for right in names[left_index + 1 :]
            if generator.random() < 0.08
        ]
        parallel, _, _ = parallel_ordered_greedy(priorities, edges)
        assert parallel == sequential_ordered_greedy(priorities, edges)


def test_dedup_resume_anchor_and_exact_reason_match_sequential_semantics() -> None:
    codec = MinHashCodec.from_config(QUALITY["near_duplicate"])
    anchor_candidate = _candidate(codec, "anchor", 0, "anchor text", 11)
    anchor = candidate_to_anchor(anchor_candidate, codec)
    near_anchor = _candidate(codec, "near-anchor", 1, "different text", 11)
    independent = _candidate(codec, "independent", 2, "independent text", 22)
    result = deduplicate_candidates([near_anchor, independent], codec, anchors=[anchor])
    assert result.accepted_document_ids == ("independent",)
    assert result.rejection_reasons == {"near-anchor": "near_duplicate"}

    root = _candidate(codec, "root", 0, "root text", 33)
    duplicate_a = _candidate(codec, "duplicate-a", 1, "same exact text", 33)
    duplicate_b = _candidate(codec, "duplicate-b", 2, "same exact text", 33)
    classification = deduplicate_candidates([root, duplicate_a, duplicate_b], codec)
    assert classification.accepted_document_ids == ("root",)
    assert classification.rejection_reasons == {
        "duplicate-a": "near_duplicate",
        "duplicate-b": "near_duplicate",
    }


def test_mapper_does_not_leak_transformation_fields_to_top_level() -> None:
    codec = MinHashCodec.from_config(QUALITY["near_duplicate"])
    signature_bytes, signature = _signature(codec, 44)

    class EmptyNearIndex:
        @staticmethod
        def count() -> int:
            return 0

    class FakeQualityGate:
        near_duplicate = EmptyNearIndex()

        @staticmethod
        def evaluate(text: str, category: str) -> QualityEvaluation:
            del text, category
            return QualityEvaluation(
                None,
                {
                    "language": {"label": "en"},
                    "near_duplicate": {
                        "fingerprint_sha256": codec.fingerprint(signature_bytes)
                    },
                },
                signature,
            )

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "alpha": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    mapper = DistributedCandidateMapper(
        tokenizer=tokenizer,
        quality_gate=FakeQualityGate(),  # type: ignore[arg-type]
        minhash_codec=codec,
        filters={
            "max_control_character_fraction": 0.01,
            "max_repeated_line_fraction": 0.9,
            "max_segment_utf8_bytes": 131072,
            "min_document_characters": 200,
        },
        excluded_document_ids=set(),
        excluded_content_sha256=set(),
        excluded_near_sha256=set(),
    )
    text = " ".join(["alpha"] * 60)
    mapped = mapper.map_records(
        source={
            "category": "science_and_education",
            "component": "fixture",
            "dataset_id": "fixture/dataset",
            "ordinal": 0,
            "revision": "b" * 40,
            "source_file": "fixture.json.gz",
            "source_sha256": "a" * 64,
        },
        source_provenance={},
        records=[(1, {"text": text})],
    )
    assert len(mapped.candidates) == 1
    candidate = mapped.candidates[0]
    assert "pii_redactions" not in candidate
    assert "raw_sha256" not in candidate
    assert candidate["transformations"]["pii_redactions"] == {}


def test_distributed_plan_is_roadmap_bounded_and_paid_compute_is_off(tmp_path: Path) -> None:
    contract = ROOT / "configs" / "distributed-production-corpus-v0.1.json"
    plan = build_distributed_corpus_plan(
        contract,
        repository_root=ROOT,
        source_commit="1" * 40,
    )
    assert plan["limits"]["candidate_total_floor_tokens"] == 120_000_000_000
    assert plan["limits"]["final_training_tokens"] == 80_000_000_000
    assert plan["resume_baseline"]["accepted_tokens"] == 237_769_640
    assert plan["cost"] == {
        "canary_hard_maximum_usd": "1.76",
        "dpu_hour_usd": "0.44",
        "hard_production_usd_per_hour": "56.32",
        "initial_production_usd_per_hour": "28.16",
    }
    assert plan["decision"]["paid_glue_canary"] == "not_authorized"

    tampered = json.loads(contract.read_text())
    tampered["authorization"]["paid_glue_canary"] = True
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="must not authorize paid compute"):
        build_distributed_corpus_plan(
            tampered_path,
            repository_root=ROOT,
            source_commit="1" * 40,
        )

    tampered = json.loads(contract.read_text())
    tampered["canary"]["maximum_total_source_compressed_bytes"] += 1
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="canary envelope changed"):
        build_distributed_corpus_plan(
            tampered_path,
            repository_root=ROOT,
            source_commit="1" * 40,
        )


def _write_fixture_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence = tmp_path / "evidence"
    (evidence / "partial-current-shard").mkdir(parents=True)
    codec = MinHashCodec.from_config(QUALITY["near_duplicate"])
    first = candidate_to_anchor(_candidate(codec, "first", 0, "first text", 101), codec)
    second = candidate_to_anchor(_candidate(codec, "second", 1, "second text", 202), codec)
    database = evidence / "acquisition-state.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE exact_content(
          content_sha256 TEXT PRIMARY KEY, document_id TEXT UNIQUE,
          source_ordinal INTEGER, source_line INTEGER, segment_index INTEGER
        );
        CREATE TABLE production_near_duplicate_metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE production_near_duplicate_signatures(
          document_id TEXT PRIMARY KEY, signature BLOB
        );
        CREATE TABLE production_near_duplicate_buckets(
          band INTEGER, bucket BLOB, document_id TEXT,
          PRIMARY KEY(band, bucket, document_id)
        );
        """
    )
    connection.executemany(
        "INSERT INTO production_near_duplicate_metadata VALUES (?, ?)",
        (("fingerprint", _quality_fingerprint(QUALITY, codec)), ("schema_version", "1")),
    )
    for anchor in (first, second):
        connection.execute(
            "INSERT INTO exact_content VALUES (?, ?, ?, ?, ?)",
            (
                anchor["content_sha256"],
                anchor["document_id"],
                anchor["source_ordinal"],
                anchor["source_line"],
                anchor["segment_index"],
            ),
        )
        connection.execute(
            "INSERT INTO production_near_duplicate_signatures VALUES (?, ?)",
            (anchor["document_id"], anchor["minhash_signature"]),
        )
        connection.executemany(
            "INSERT INTO production_near_duplicate_buckets VALUES (?, ?, ?)",
            (
                (row["band"], row["bucket"], anchor["document_id"])
                for row in anchor["lsh_buckets"]
            ),
        )
    connection.commit()
    connection.close()

    clean = evidence / "partial-current-shard" / "clean.jsonl"
    clean.write_text(
        json.dumps(
            {
                "content_sha256": second["content_sha256"],
                "id": "second",
                "text": "second text",
                "token_count": 2,
                "token_offset": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    tokens = evidence / "partial-current-shard" / "tokens.uint16.bin"
    tokens.write_bytes(b"\x01\x00\x02\x00")
    manual = evidence / "manual-stop-evidence.json"
    manual.write_text(
        json.dumps(
            {"partial_tokens": 2, "sealed_tokens": 1, "total_accepted_tokens": 3}
        )
    )
    for name in ("bootstrap.log", "production-report.json", "worker-release.json", "worker.log"):
        (evidence / name).write_text("{}\n")
    names = (
        "acquisition-state.sqlite3",
        "bootstrap.log",
        "manual-stop-evidence.json",
        "partial-current-shard/clean.jsonl",
        "partial-current-shard/tokens.uint16.bin",
        "production-report.json",
        "worker-release.json",
        "worker.log",
    )
    manifest = {
        "files": [
            {
                "bytes": (evidence / name).stat().st_size,
                "path": name,
                "sha256": _sha256(evidence / name),
            }
            for name in names
        ],
        "schema_version": 1,
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    baseline = {
        "archive": {"internal_manifest_sha256": _sha256(evidence / "manifest.json")},
        "gate": "verified_single_writer_resume_baseline_v0.1",
        "partial_prefix": {
            "accepted_documents": 1,
            "accepted_tokens": 2,
            "clean_bytes": clean.stat().st_size,
            "clean_sha256": _sha256(clean),
            "evidence_member_clean": "partial-current-shard/clean.jsonl",
            "evidence_member_tokens": "partial-current-shard/tokens.uint16.bin",
            "token_bytes": tokens.stat().st_size,
            "token_sha256": _sha256(tokens),
        },
        "schema_version": 1,
        "totals": {
            "accepted_documents": 2,
            "accepted_tokens": 3,
            "sealed_documents": 1,
            "sealed_tokens": 1,
        },
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline, sort_keys=True))
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(QUALITY, sort_keys=True))
    return evidence, baseline_path, quality_path


def test_resume_anchor_export_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    evidence, baseline, quality = _write_fixture_evidence(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = export_resume_anchors(
        baseline_contract_path=baseline,
        quality_contract_path=quality,
        evidence_directory=evidence,
        output_directory=first_output,
    )
    export_resume_anchors(
        baseline_contract_path=baseline,
        quality_contract_path=quality,
        evidence_directory=evidence,
        output_directory=second_output,
    )
    assert first["anchors"]["rows"] == 2
    assert first["partial_prefix"] == {"documents": 1, "token_bytes": 4, "tokens": 2}
    assert _sha256(first_output / "resume-anchors-v0.1.jsonl.gz") == _sha256(
        second_output / "resume-anchors-v0.1.jsonl.gz"
    )
    verified = verify_resume_anchor_export(first_output, quality)
    assert verified["anchors"] == 2
    assert verified["status"] == "verified"


def test_glue_adapter_imports_without_pyspark() -> None:
    import lumi.glue_corpus as glue_corpus

    assert callable(glue_corpus.spark_deduplicate_candidates)
    assert DEDUP_ANCHOR_SCHEMA_VERSION == 1


def _synthetic_glue_release() -> dict[str, object]:
    categories = (
        "science_and_education",
        "code",
        "reference",
        "questions_and_discussion",
        "open_web",
        "books",
        "government_legal_and_patents",
        "science_and_education",
    )
    wheel_uri = (
        "s3://lumi-corpus-prod-test/lumi-corpus/v0.1/control/distributed/"
        "glue-cp311/bbbbbbbbbbbb/codebuild/output/"
        "lingua_language_detector-2.2.0-cp311-cp311-manylinux_2_34_x86_64.whl"
    )
    wheel_pin = {
        "codebuild": {
            "build_id": "lumi-cb-lingua-cp311-bbbbbbbbbbbb:build-id",
            "build_report_s3_uri": (
                "s3://lumi-corpus-prod-test/lumi-corpus/v0.1/control/distributed/"
                "glue-cp311/bbbbbbbbbbbb/codebuild/output/"
                "lingua-glue-cp311-build-v0.1.json"
            ),
            "build_report_sha256": "d" * 64,
            "build_report_version_id": "build-report-version",
            "build_status": "SUCCEEDED",
            "parity_report_s3_uri": (
                "s3://lumi-corpus-prod-test/lumi-corpus/v0.1/control/distributed/"
                "glue-cp311/bbbbbbbbbbbb/codebuild/output/"
                "lingua-glue-cp311-parity-v0.1.json"
            ),
            "parity_report_sha256": "e" * 64,
            "parity_report_version_id": "parity-report-version",
            "source_archive_sha256": "b" * 64,
            "source_commit": "f" * 40,
            "source_version": "pinned-version-id",
        },
        "gate": "lumi_lingua_glue_cp311_wheel_pin_v0.1",
        "schema_version": 1,
        "wheel": {
            "bytes": 1_000,
            "s3_uri": wheel_uri,
            "sha256": "c" * 64,
            "version_id": "wheel-version",
        },
    }
    release: dict[str, object] = {
        "artifacts": {
            "lingua_cp311_wheel": {
                **wheel_pin["wheel"],
                "state": "existing_verified_codebuild_object",
            }
        },
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_canary": False,
            "paid_glue_production": False,
            "paid_gpu_compute": False,
        },
        "canary": {
            "maximum_dpu": 8,
            "maximum_records_per_source": 2_000,
            "maximum_runtime_minutes": 30,
            "maximum_source_compressed_bytes": 700_000_000,
            "maximum_total_source_compressed_bytes": 1_500_000_000,
            "requires_new_owner_approval": True,
            "source_files": 8,
        },
        "gate": "lumi_distributed_glue_canary_release_v0.1",
        "lingua_compatibility": wheel_pin,
        "job": {
            "additional_python_modules": (
                f"tokenizers==0.22.2,datasketch==2.0.0,{wheel_uri}"
            ),
            "auto_scaling": False,
            "execution_class": "STANDARD",
            "glue_version": "5.1",
            "maximum_concurrent_runs": 1,
            "number_of_workers": 8,
            "python_version": "3.11",
            "timeout_minutes": 30,
            "worker_type": "G.1X",
        },
        "output": {"write_production_shards": False},
        "schema_version": 1,
        "source_commit": "a" * 40,
        "sources": [
            {
                "category": category,
                "maximum_records": 2_000,
                "ordinal": index,
                "source_bytes": 1_000,
            }
            for index, category in enumerate(categories)
        ],
    }
    material = json.dumps(release, separators=(",", ":"), sort_keys=True).encode()
    release["release_fingerprint"] = hashlib.sha256(material).hexdigest()
    release["decision"] = {"paid_glue_canary": "not_authorized"}
    return release


def test_glue_canary_needs_separate_exact_owner_authorization() -> None:
    now = datetime(2026, 7, 19, tzinfo=UTC)
    release = _synthetic_glue_release()
    assert validate_glue_canary_release(release)["status"] == "verified_not_authorized"
    authorization = {
        "approval_text": CANARY_APPROVAL_TEXT,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_canary": True,
            "paid_glue_production": False,
            "paid_gpu_compute": False,
        },
        "bounds": {
            "maximum_dpu": 8,
            "maximum_records_per_source": 2_000,
            "maximum_source_compressed_bytes": 700_000_000,
            "maximum_runtime_minutes": 30,
            "maximum_source_files": 8,
            "maximum_total_source_compressed_bytes": 1_500_000_000,
            "maximum_usd": "1.76",
        },
        "expires_at": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "gate": "lumi_distributed_glue_canary_authorization_v0.1",
        "owner": "Afnan Technologies Inc.",
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "source_commit": release["source_commit"],
    }
    verified = validate_glue_canary_authorization(authorization, release, now=now)
    assert verified["status"] == "authorized_bounded_canary_only"

    authorization["bounds"]["maximum_dpu"] = 9
    with pytest.raises(ValueError, match="bounds changed"):
        validate_glue_canary_authorization(authorization, release, now=now)


def test_glue_canary_rejects_unpinned_lingua_wheel() -> None:
    release = _synthetic_glue_release()
    release["lingua_compatibility"]["wheel"]["s3_uri"] = (
        "s3://unbounded-bucket/lingua.whl"
    )
    material = {
        key: value
        for key, value in release.items()
        if key not in {"decision", "release_fingerprint"}
    }
    release["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="outside the compatibility prefix"):
        validate_glue_canary_release(release)


def test_glue_canary_does_not_redeclare_reserved_job_run_id() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_distributed_canary.py",
        run_name="lumi_glue_canary_test",
    )
    requested = namespace["GLUE_REQUESTED_OPTIONS"]
    assert requested == (
        "JOB_NAME",
        "LUMI_AUTHORIZATION_FILE",
        "LUMI_RELEASE_FILE",
    )
    assert "JOB_RUN_ID" not in requested


def test_glue_canary_source_selection_is_all_category_and_byte_bounded() -> None:
    manifest = json.loads((ROOT / "configs" / "common-pile-source-files-v0.1.json").read_text())
    mixture = json.loads((ROOT / "configs" / "data-mix-v0.1.json").read_text())
    contract = json.loads(
        (ROOT / "configs" / "distributed-production-corpus-v0.1.json").read_text()
    )
    schedule, _ = build_round_robin_source_schedule(manifest, mixture)
    selected = _select_canary_sources(
        schedule,
        contract["canary"],
        {"source_ordinal": 0},
    )
    assert len(selected) == 8
    assert selected[0]["ordinal"] == 0
    assert len({row["category"] for row in selected}) == 7
    assert max(int(row["source_bytes"]) for row in selected) == 652_928_967
    assert sum(int(row["source_bytes"]) for row in selected) == 1_456_514_440


def test_local_spark_dedup_matches_independent_sequential_reference(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    from lumi.glue_corpus import (
        DEDUP_CANDIDATE_COLUMNS,
        assign_ordered_token_shards,
        candidate_spark_schema,
        dedup_anchor_spark_schema,
        dedup_candidate_spark_schema,
        select_ordered_token_prefix,
        select_ordered_token_prefix_at_most,
        spark_deduplicate_candidates,
    )

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("lumi-distributed-corpus-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    try:
        codec = MinHashCodec.from_config(QUALITY["near_duplicate"])
        anchor = candidate_to_anchor(_candidate(codec, "anchor", 0, "anchor", 11), codec)
        candidates = [
            _candidate(codec, "near-anchor", 1, "near anchor", 11),
            _candidate(codec, "accepted", 2, "accepted exact text", 22),
            _candidate(codec, "near-accepted", 3, "different near text", 22),
            _candidate(codec, "exact-accepted", 4, "accepted exact text", 22),
            _candidate(codec, "independent", 5, "independent", 33),
        ]
        candidate_frame = spark.createDataFrame(
            [candidate_to_spark_row(row) for row in candidates],
            schema=candidate_spark_schema(),
        )
        anchor_frame = spark.createDataFrame([anchor], schema=dedup_anchor_spark_schema())
        with pytest.raises(ValueError, match="must be private S3"):
            spark_deduplicate_candidates(
                candidate_frame,
                codec=codec,
                checkpoint_uri=tmp_path.as_uri(),
                anchors=anchor_frame,
            )
        with pytest.raises(RuntimeError, match="bucket exceeds"):
            spark_deduplicate_candidates(
                candidate_frame,
                codec=codec,
                checkpoint_uri=(tmp_path / "oversized").as_uri(),
                anchors=anchor_frame,
                allow_local_checkpoint=True,
                maximum_lsh_bucket_documents=2,
                maximum_candidate_pairs=1_000,
            )
        result = spark_deduplicate_candidates(
            candidate_frame,
            codec=codec,
            checkpoint_uri=tmp_path.as_uri(),
            anchors=anchor_frame,
            allow_local_checkpoint=True,
            maximum_lsh_bucket_documents=100,
            maximum_candidate_pairs=1_000,
        )
        reference = deduplicate_candidates(candidates, codec, anchors=[anchor])
        spark_accepted = {
            row["document_id"] for row in result["accepted_ids"].select("document_id").collect()
        }
        spark_rejections = {
            row["document_id"]: "exact_duplicate"
            for row in result["exact_rejected"].select("document_id").collect()
        }
        spark_rejections.update(
            {
                row["document_id"]: "near_duplicate"
                for row in result["near_rejected"].select("document_id").collect()
            }
        )
        assert spark_accepted == set(reference.accepted_document_ids)
        assert spark_rejections == reference.rejection_reasons
        assert result["metrics"]["anchor_documents"] == 1
        assert result["metrics"]["candidate_documents"] == 5
        repeated = spark_deduplicate_candidates(
            candidate_frame,
            codec=codec,
            checkpoint_uri=(tmp_path / "repeated").as_uri(),
            anchors=anchor_frame,
            allow_local_checkpoint=True,
            maximum_lsh_bucket_documents=100,
            maximum_candidate_pairs=1_000,
        )
        assert {
            row["document_id"]
            for row in repeated["accepted_ids"].select("document_id").collect()
        } == spark_accepted
        assert repeated["metrics"] == result["metrics"]

        assert [field.name for field in dedup_candidate_spark_schema().fields] == list(
            DEDUP_CANDIDATE_COLUMNS
        )
        narrow_frame = candidate_frame.select(*DEDUP_CANDIDATE_COLUMNS)
        narrow = spark_deduplicate_candidates(
            narrow_frame,
            codec=codec,
            checkpoint_uri=(tmp_path / "narrow").as_uri(),
            anchors=anchor_frame,
            allow_local_checkpoint=True,
            maximum_lsh_bucket_documents=100,
            maximum_candidate_pairs=1_000,
        )
        assert {
            row["document_id"]
            for row in narrow["accepted_ids"].select("document_id").collect()
        } == spark_accepted
        assert set(narrow["accepted"].columns) == {
            *DEDUP_CANDIDATE_COLUMNS,
            "dedup_round",
        }
        assert narrow["metrics"] == result["metrics"]

        prefix_candidates = []
        for ordinal, tokens in enumerate((3, 4, 5, 2, 6, 1)):
            row = _candidate(
                codec,
                f"prefix-{ordinal}",
                ordinal + 10,
                f"prefix text {ordinal}",
                100 + ordinal,
            )
            row["token_count"] = tokens
            row["token_bytes"] = b"\x01\x00" * tokens
            prefix_candidates.append(row)
        random.Random(7).shuffle(prefix_candidates)
        prefix_frame = spark.createDataFrame(
            [candidate_to_spark_row(row) for row in prefix_candidates],
            schema=candidate_spark_schema(),
        )
        prefix = select_ordered_token_prefix(
            prefix_frame,
            target_tokens=10,
            maximum_overshoot_tokens=2,
            partitions=3,
        )
        assert prefix["reached_target"] is True
        assert prefix["selected_tokens"] == 12
        assert prefix["overshoot_tokens"] == 2
        assert [
            row["document_id"]
            for row in prefix["selected"]
            .orderBy("source_ordinal", "source_line", "segment_index", "document_id")
            .collect()
        ] == ["prefix-0", "prefix-1", "prefix-2"]
        prefix["selected"].unpersist()
        all_rows = select_ordered_token_prefix(
            prefix_frame,
            target_tokens=100,
            maximum_overshoot_tokens=2,
            partitions=3,
        )
        assert all_rows["reached_target"] is False
        assert all_rows["selected_documents"] == 6
        assert all_rows["selected_tokens"] == 21
        all_rows["selected"].unpersist()
        with pytest.raises(RuntimeError, match="overshoot"):
            select_ordered_token_prefix(
                prefix_frame,
                target_tokens=10,
                maximum_overshoot_tokens=1,
                partitions=3,
            )
        at_most = select_ordered_token_prefix_at_most(
            prefix_frame,
            maximum_tokens=10,
            partitions=3,
        )
        assert at_most["selected_tokens"] == 7
        assert at_most["shortfall_tokens"] == 3
        assert [
            row["document_id"]
            for row in at_most["selected"]
            .orderBy("source_ordinal", "source_line", "segment_index", "document_id")
            .collect()
        ] == ["prefix-0", "prefix-1"]
        at_most["selected"].unpersist()
        shards = assign_ordered_token_shards(
            prefix_frame,
            target_tokens=10,
            maximum_overshoot_tokens=5,
            partitions=3,
        )
        shard_totals = {
            int(row["shard_sequence"]): int(row["tokens"])
            for row in shards["assigned"]
            .groupBy("shard_sequence")
            .sum("token_count")
            .withColumnRenamed("sum(token_count)", "tokens")
            .collect()
        }
        assert shard_totals == {0: 12, 1: 8, 2: 1}
        assert shards["documents"] == 6
        assert shards["tokens"] == 21
        assert shards["shard_count"] == 3
        shards["assigned"].unpersist()
    finally:
        spark.stop()
