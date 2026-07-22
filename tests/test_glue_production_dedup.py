from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.emr_serverless_production import build_emr_production_plan
from lumi.glue_production_dedup import (
    INPUT_INDEX_FILE,
    MAP_CHECKPOINT_FILE,
    MAP_RELEASE_FILE,
    RELEASE_FILE,
    build_glue_global_dedup_authorization,
    build_glue_global_dedup_input_index,
    build_glue_global_dedup_job_definition,
    build_glue_production_global_dedup_package,
    expected_glue_global_dedup_approval,
    glue_global_dedup_job_token,
    validate_glue_global_dedup_authorization,
    validate_glue_global_dedup_input_index,
    validate_glue_global_dedup_result,
    validate_glue_production_global_dedup_release,
)
from lumi.glue_production_mapping import (
    advance_mapping_checkpoint,
    build_initial_mapping_checkpoint,
    plan_mapping_wave,
    validate_glue_production_mapping_release,
)
from lumi.production_corpus import _canonical_json_bytes

ROOT = Path(__file__).parents[1]
MAPPING_HELPERS = runpy.run_path(
    ROOT / "tests" / "test_glue_production_mapping.py",
    run_name="lumi_glue_mapping_fixture_helpers",
)


def _refingerprint(value: dict[str, object], field: str) -> None:
    material = dict(value)
    material.pop(field, None)
    value[field] = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _final_map_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, list[Path], dict[str, object], dict[str, object]]:
    release = MAPPING_HELPERS["_release"]()
    commit = "b" * 40
    plan = build_emr_production_plan(
        ROOT / "configs" / "emr-production-resume-v0.1.json",
        repository_root=ROOT,
        source_commit=commit,
    )
    bucket = plan["storage"]["bucket"]
    namespace = plan["storage"]["namespace"]
    release["source_commit"] = commit
    release["storage"] = {"bucket": bucket, "namespace": namespace}
    release["data_quality"] = plan["data_quality"]
    release["filters"] = plan["filters"]
    release["output"] = {
        "checkpoint_prefix": f"{namespace}/control/test-map-checkpoints",
        "mapped_prefix": f"{namespace}/transient/test-mapped",
        "result_prefix": f"{namespace}/control/test-map-runs",
        "temp_prefix": f"{namespace}/control/test-map-temp",
    }
    counts = {str(row["category"]): 0 for row in release["categories"]}
    for row in release["staged_sources"]:
        row["chunk_count"] = 20
        counts[str(row["category"])] += 20
    release["category_inventory"] = [
        {"category": row["category"], "staged_chunks": counts[str(row["category"])]}
        for row in release["categories"]
    ]
    release["stage"]["staged_chunks"] = sum(counts.values())
    _refingerprint(release, "release_fingerprint")
    validate_glue_production_mapping_release(release)

    checkpoint = build_initial_mapping_checkpoint(release)
    manifests: list[dict[str, object]] = []
    for wave_sequence in range(1, 13):
        wave = plan_mapping_wave(checkpoint, release)
        close = wave_sequence == 12
        results = []
        for row in wave["categories"]:
            name = str(row["category"])
            selected_tokens = (
                int(row["remaining_target_tokens"]) - 1 if close else 1
            )
            digest = hashlib.sha256(f"{wave_sequence}:{name}".encode()).hexdigest()
            results.append(
                {
                    "available_documents": 1,
                    "available_tokens": selected_tokens,
                    "candidate_documents": 1,
                    "candidate_tokens": selected_tokens,
                    "category": name,
                    "closed": close,
                    "cutoff": {"document_id": f"{wave_sequence}:{name}"}
                    if close
                    else None,
                    "end_chunk_index": row["end_chunk_index"],
                    "object_count": 1,
                    "objects_sha256": digest,
                    "parquet_bytes": 10,
                    "parquet_s3_uri": (
                        f"s3://{bucket}/{release['output']['mapped_prefix']}/"
                        f"{release['release_fingerprint']}/fixture/wave-{wave_sequence:06d}/"
                        f"{name}"
                    ),
                    "start_chunk_index": row["start_chunk_index"],
                }
            )
        manifest: dict[str, object] = {
            "candidate_documents": len(results),
            "candidate_tokens": sum(int(row["candidate_tokens"]) for row in results),
            "categories": results,
            "gate": "lumi_glue_production_map_wave_result_v0.1",
            "manifest_s3_key": (
                f"{release['output']['result_prefix']}/{release['release_fingerprint']}/"
                f"fixture/waves/wave-{wave_sequence:06d}/manifest.json"
            ),
            "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "parquet_bytes": sum(int(row["parquet_bytes"]) for row in results),
            "release_fingerprint": release["release_fingerprint"],
            "schema_version": 1,
            "wave_plan_sha256": wave["wave_sha256"],
            "wave_sequence": wave_sequence,
        }
        _refingerprint(manifest, "manifest_sha256")
        checkpoint = advance_mapping_checkpoint(checkpoint, release, manifest)
        manifests.append(manifest)
    assert checkpoint["status"] == "ready_for_global_dedup"
    assert checkpoint["checkpoint_sequence"] == 12

    release_path = tmp_path / "map-release.json"
    checkpoint_path = tmp_path / "map-checkpoint.json"
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    manifest_paths = []
    for manifest in manifests:
        path = tmp_path / f"wave-{int(manifest['wave_sequence']):02d}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_paths.append(path)
    return release_path, checkpoint_path, manifest_paths, release, checkpoint


def _package_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    package = tmp_path / "dedup-package"
    release = build_glue_production_global_dedup_package(
        contract_path=ROOT / "configs" / "glue-production-global-dedup-v0.1.json",
        repository_root=ROOT,
        source_commit="b" * 40,
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
        output_directory=package,
    )
    return release, package, map_release, checkpoint


def test_input_index_binds_all_waves_and_conserves_the_150b_map(tmp_path: Path) -> None:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    index = build_glue_global_dedup_input_index(
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
    )
    assert len(index["waves"]) == 12
    assert len(index["inputs"]) > 12
    assert index["mapped_tokens"] == checkpoint["total_mapped_tokens"]
    assert index["mapped_documents"] == checkpoint["total_mapped_documents"]
    assert index["mapped_parquet_bytes"] == checkpoint["mapped_parquet_bytes"]
    assert index["total_candidate_tokens"] == 150_000_000_000 - 7
    assert validate_glue_global_dedup_input_index(
        index, map_release, checkpoint
    ) == index


def test_input_index_refuses_missing_reordered_and_corrupt_manifests(
    tmp_path: Path,
) -> None:
    release_path, checkpoint_path, manifests, _, _ = _final_map_fixture(tmp_path)
    with pytest.raises(ValueError, match="count changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=manifests[:-1],
        )
    with pytest.raises(ValueError, match="order changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=list(reversed(manifests)),
        )
    changed = json.loads(manifests[0].read_text())
    changed["candidate_tokens"] += 1
    corrupt = tmp_path / "corrupt-wave.json"
    corrupt.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        build_glue_global_dedup_input_index(
            map_release_path=release_path,
            map_checkpoint_path=checkpoint_path,
            wave_manifest_paths=[corrupt, *manifests[1:]],
        )


def test_input_index_refuses_namespace_escape_and_duplicate_objects(
    tmp_path: Path,
) -> None:
    release_path, checkpoint_path, manifests, map_release, checkpoint = (
        _final_map_fixture(tmp_path)
    )
    index = build_glue_global_dedup_input_index(
        map_release_path=release_path,
        map_checkpoint_path=checkpoint_path,
        wave_manifest_paths=manifests,
    )
    escaped = deepcopy(index)
    escaped["inputs"][0]["parquet_s3_uri"] = "s3://attacker/outside.parquet"
    _refingerprint(escaped, "input_index_sha256")
    with pytest.raises(ValueError, match="mapped namespace"):
        validate_glue_global_dedup_input_index(escaped, map_release, checkpoint)

    duplicated = deepcopy(index)
    duplicated["inputs"][1]["parquet_s3_uri"] = duplicated["inputs"][0][
        "parquet_s3_uri"
    ]
    _refingerprint(duplicated, "input_index_sha256")
    with pytest.raises(ValueError, match="repeats a Parquet input"):
        validate_glue_global_dedup_input_index(duplicated, map_release, checkpoint)

    repeated_inventory = deepcopy(index)
    repeated_inventory["inputs"][1]["objects_sha256"] = repeated_inventory["inputs"][
        0
    ]["objects_sha256"]
    _refingerprint(repeated_inventory, "input_index_sha256")
    with pytest.raises(ValueError, match="object inventory"):
        validate_glue_global_dedup_input_index(
            repeated_inventory, map_release, checkpoint
        )


def test_package_is_hash_bound_and_does_not_authorize_paid_work(tmp_path: Path) -> None:
    release, package, _, _ = _package_fixture(tmp_path)
    verified = validate_glue_production_global_dedup_release(release)
    assert verified["status"] == "verified_not_authorized"
    assert release["authorization"]["paid_glue_dedup"] is False
    assert release["decision"]["full_4b_training"] == "not_authorized"
    assert release["map"]["wave_count"] == 12
    assert release["jobs"]["canary"]["number_of_workers"] == 8
    assert release["jobs"]["production"]["number_of_workers"] == 100
    assert release["jobs"]["production"]["maximum_compute_usd"] == "1056.000000"
    assert release["deduplication"] == {
        "maximum_candidate_pairs": 50_000_000,
        "documents_without_usable_lsh_buckets_policy": "keep_exact_only",
        "maximum_documents_without_usable_lsh_buckets": 16,
        "maximum_lsh_bucket_documents": 10_000,
        "maximum_oversized_lsh_buckets": 64,
        "maximum_oversized_lsh_memberships": 1_000_000,
        "maximum_pair_budget_lsh_buckets": 5_000_000,
        "maximum_pair_budget_lsh_memberships": 100_000_000,
        "maximum_rounds": 256,
        "oversized_lsh_bucket_policy": "drop_as_stop_bucket",
        "pair_budget_lsh_bucket_policy": "drop_as_stop_bucket",
    }
    assert (package / RELEASE_FILE).is_file()
    assert (package / INPUT_INDEX_FILE).is_file()
    assert (package / MAP_RELEASE_FILE).is_file()
    assert (package / MAP_CHECKPOINT_FILE).is_file()
    lines = (package / "manifest.sha256").read_text().splitlines()
    assert len(lines) == 18
    assert any("wave-manifests/wave-000012.json" in line for line in lines)


def test_canary_and_production_authorizations_are_separate_and_bounded(
    tmp_path: Path,
) -> None:
    release, _, _, _ = _package_fixture(tmp_path)
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="does not match"):
        build_glue_global_dedup_authorization(
            release,
            mode="canary",
            approval_text="approve it",
            now=now,
        )
    canary = build_glue_global_dedup_authorization(
        release,
        mode="canary",
        approval_text=expected_glue_global_dedup_approval(release, mode="canary"),
        now=now,
    )
    production = build_glue_global_dedup_authorization(
        release,
        mode="production",
        approval_text=expected_glue_global_dedup_approval(
            release, mode="production"
        ),
        now=now,
    )
    assert canary["authorization_sha256"] != production["authorization_sha256"]
    assert canary["bounds"]["hard_maximum_dpu"] == 8
    assert canary["bounds"]["maximum_usd"] == "1.760000"
    assert canary["permissions"]["production_global_dedup"] is False
    assert production["bounds"]["hard_maximum_dpu"] == 100
    assert production["bounds"]["maximum_usd"] == "1056.000000"
    assert production["permissions"]["production_global_dedup"] is True
    assert production["permissions"]["model_training"] is False
    assert validate_glue_global_dedup_authorization(
        canary, release, now=now
    )["status"] == "authorized_one_canary_global_dedup_run"
    with pytest.raises(ValueError, match="not currently valid"):
        validate_glue_global_dedup_authorization(
            canary, release, now=now + timedelta(hours=25)
        )

    bucket = release["storage"]["bucket"]
    namespace = release["storage"]["namespace"]
    canary_job = build_glue_global_dedup_job_definition(
        release,
        canary,
        role_arn="arn:aws:iam::123456789012:role/lumi-glue-dedup",
        release_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/release.json",
        authorization_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/canary-auth.json",
    )
    production_job = build_glue_global_dedup_job_definition(
        release,
        production,
        role_arn="arn:aws:iam::123456789012:role/lumi-glue-dedup",
        release_s3_uri=f"s3://{bucket}/{namespace}/control/dedup/release.json",
        authorization_s3_uri=(
            f"s3://{bucket}/{namespace}/control/dedup/production-auth.json"
        ),
    )
    assert (canary_job["NumberOfWorkers"], canary_job["Timeout"]) == (8, 30)
    assert (production_job["NumberOfWorkers"], production_job["Timeout"]) == (
        100,
        1_440,
    )
    assert canary_job["MaxRetries"] == production_job["MaxRetries"] == 0
    assert "model" not in canary_job["Description"].lower().split(";", 1)[0]


def test_runtime_reuses_the_proven_dedup_core_and_initializes_spark_first() -> None:
    source = (ROOT / "scripts" / "run_glue_production_global_dedup.py").read_text()
    assert "spark_deduplicate_candidates" in source
    assert "maximum_candidate_pairs" in source
    assert "oversized_lsh_bucket_policy" in source
    assert "maximum_oversized_lsh_memberships" in source
    assert "pair_budget_lsh_bucket_policy" in source
    assert "maximum_pair_budget_lsh_memberships" in source
    assert "documents_without_usable_lsh_buckets_policy" in source
    assert "errorifexists" in source
    assert "StorageLevel.DISK_ONLY" in source
    assert "candidates = candidates.persist()" not in source
    assert "selected_payload = candidates.join(" in source
    assert 'spark.sql.parquet.enableVectorizedReader", "false"' in source
    context = source.index("context = SparkContext.getOrCreate()")
    release = source.index('SparkFiles.get(args["LUMI_RELEASE_FILE"])')
    assert context < release
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_script_test",
    )
    assert namespace["GLUE_REQUESTED_OPTIONS"] == (
        "JOB_NAME",
        "LUMI_AUTHORIZATION_FILE",
        "LUMI_INPUT_INDEX_FILE",
        "LUMI_MAP_CHECKPOINT_FILE",
        "LUMI_MAP_RELEASE_FILE",
        "LUMI_RELEASE_FILE",
    )


def test_runtime_projects_wide_payload_away_until_final_selection() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_memory_projection_test",
    )
    operations: list[tuple[object, ...]] = []

    class Frame:
        def __init__(self, label: str) -> None:
            self.label = label

        def select(self, *columns: str) -> Frame:
            operations.append(("select", self.label, *columns))
            return Frame(f"{self.label}-selected")

        def join(self, other: object, *args: str) -> Frame:
            operations.append(("join", self.label, other, *args))
            return Frame(f"{self.label}-joined")

    dedup_columns = (
        "schema_version",
        "document_id",
        "source_ordinal",
        "source_line",
        "segment_index",
        "content_sha256",
        "minhash_signature",
        "lsh_buckets",
    )
    candidates = Frame("candidates")
    accepted_ids = Frame("accepted-ids")
    narrow = namespace["_narrow_dedup_candidates"](
        candidates, columns=dedup_columns
    )
    selection = namespace["_accepted_selection_candidates"](
        candidates, accepted_ids
    )

    assert narrow.label == "candidates-selected"
    assert selection.label == "candidates-selected-joined"
    assert operations[0] == ("select", "candidates", *dedup_columns)
    assert "text" not in operations[0]
    assert "token_bytes" not in operations[0]
    assert operations[1] == (
        "select",
        "candidates",
        *namespace["PRODUCTION_SELECTION_COLUMNS"],
    )
    assert "text" not in operations[1]
    assert "token_bytes" not in operations[1]
    assert operations[2] == (
        "select",
        "accepted-ids",
        "document_id",
        "dedup_round",
    )
    assert operations[3][0:2] == ("join", "candidates-selected")
    assert operations[3][-2:] == ("document_id", "inner")


def test_runtime_enforces_memory_safe_parquet_configuration() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_memory_config_test",
    )

    class Conf:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}

        def set(self, name: str, value: str) -> None:
            self.values[name] = value

        def get(self, name: str) -> str:
            return self.values[name]

    class Spark:
        def __init__(self) -> None:
            self.conf = Conf()

    spark = Spark()
    actual = namespace["_configure_memory_safe_parquet_reader"](spark)
    assert actual == dict(namespace["MEMORY_SAFE_PARQUET_CONFIG"])
    assert actual["spark.sql.parquet.enableVectorizedReader"] == "false"
    assert actual["spark.sql.parquet.columnarReaderBatchSize"] == "64"
    assert actual["spark.sql.files.maxPartitionBytes"] == str(64 * 1024 * 1024)


def test_memory_safe_projection_streams_long_text_only_after_selection(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from lumi.glue_corpus import DEDUP_CANDIDATE_COLUMNS, candidate_spark_schema

    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_long_payload_test",
    )
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("lumi-glue-global-dedup-long-payload-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        payload_bytes = 8 * 1024 * 1024
        row = {
            "schema_version": 1,
            "document_id": "long-payload-document",
            "source_ordinal": 1,
            "source_line": 1,
            "segment_index": 0,
            "category": "science_and_education",
            "component": "test",
            "dataset_id": "test/long-payload",
            "dataset_revision": "fixture",
            "source_file": "fixture.jsonl",
            "source_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "minhash_signature": b"m" * 16,
            "lsh_buckets": [{"band": 0, "bucket": b"bucket"}],
            "text": "x" * payload_bytes,
            "text_utf8_bytes": payload_bytes,
            "token_bytes": b"t" * payload_bytes,
            "token_count": 1,
            "provenance_json": "{}",
            "quality_json": "{}",
            "transformations_json": "[]",
        }
        input_path = tmp_path / "wide-input"
        spark.createDataFrame([row], schema=candidate_spark_schema()).write.parquet(
            str(input_path)
        )

        config = namespace["_configure_memory_safe_parquet_reader"](spark)
        assert config["spark.sql.parquet.enableVectorizedReader"] == "false"
        candidates = spark.read.schema(candidate_spark_schema()).parquet(str(input_path))
        totals = candidates.select("token_count").agg(
            F.count(F.lit(1)).alias("documents"),
            F.sum("token_count").alias("tokens"),
        ).first()
        assert (int(totals["documents"]), int(totals["tokens"])) == (1, 1)

        narrow = namespace["_narrow_dedup_candidates"](
            candidates,
            columns=DEDUP_CANDIDATE_COLUMNS,
        )
        assert tuple(narrow.columns) == DEDUP_CANDIDATE_COLUMNS
        assert "text" not in narrow.columns
        assert "token_bytes" not in narrow.columns

        accepted_ids = spark.createDataFrame(
            [("long-payload-document", 0)],
            "document_id string, dedup_round int",
        )
        selected = namespace["_accepted_selection_candidates"](
            candidates, accepted_ids
        )
        selected_payload = candidates.join(
            selected.select("document_id", "dedup_round"),
            "document_id",
            "inner",
        )
        output_path = tmp_path / "selected-output"
        selected_payload.write.parquet(str(output_path))
        output = spark.read.parquet(str(output_path)).select(
            F.length("text").alias("text_bytes"),
            F.length("token_bytes").alias("token_bytes"),
            "dedup_round",
        ).first()
        assert (output["text_bytes"], output["token_bytes"], output["dedup_round"]) == (
            payload_bytes,
            payload_bytes,
            0,
        )
    finally:
        spark.stop()


def test_canary_orders_only_narrow_keys_before_rejoining_wide_rows() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_canary_selection_test",
    )
    operations: list[tuple[object, ...]] = []

    class Column:
        def __init__(self, name: str) -> None:
            self.name = name

        def asc(self) -> tuple[str, str]:
            return ("asc", self.name)

    class Functions:
        @staticmethod
        def col(name: str) -> Column:
            return Column(name)

        @staticmethod
        def broadcast(frame: object) -> object:
            operations.append(("broadcast", frame))
            return frame

    class Keys:
        def orderBy(self, *columns: object) -> Keys:
            operations.append(("orderBy", *columns))
            return self

        def limit(self, maximum: int) -> Keys:
            operations.append(("limit", maximum))
            return self

    class Candidates:
        def select(self, *columns: str) -> Keys:
            operations.append(("select", *columns))
            return Keys()

        def join(self, other: object, *, on: list[str], how: str) -> str:
            operations.append(("join", other, tuple(on), how))
            return "bounded-candidates"

    order = namespace["CANARY_ORDER_COLUMNS"]
    result = namespace["_bounded_canary_candidates"](
        Candidates(), maximum_documents=100_000, functions=Functions
    )

    assert result == "bounded-candidates"
    assert operations[0] == ("select", *order)
    assert operations[1] == (
        "orderBy",
        *(("asc", name) for name in order),
    )
    assert operations[2] == ("limit", 100_000)
    assert operations[3][0] == "broadcast"
    assert operations[4][0] == "join"
    assert operations[4][2:] == (tuple(order), "left_semi")

    with pytest.raises(ValueError, match="document cap must be positive"):
        namespace["_bounded_canary_candidates"](
            Candidates(), maximum_documents=0, functions=Functions
        )


def test_runtime_accepts_the_pinned_aws_spark_build_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_global_dedup.py",
        run_name="lumi_glue_global_dedup_runtime_test",
    )
    monkeypatch.setattr(
        namespace["platform"], "python_version_tuple", lambda: ("3", "11", "9")
    )
    monkeypatch.setattr(namespace["platform"], "python_version", lambda: "3.11.9")
    monkeypatch.setattr(
        namespace["importlib"].metadata,
        "version",
        lambda package: "2.0.0" if package == "datasketch" else "unexpected",
    )

    class Spark:
        def __init__(self, version: str) -> None:
            self.version = version

    runtime = namespace["_runtime"](
        Spark("3.5.6-amzn-1"), expected_spark_version="3.5.6"
    )
    assert runtime["spark"] == "3.5.6-amzn-1"
    assert runtime["spark_base"] == "3.5.6"

    with pytest.raises(RuntimeError, match="Spark changed"):
        namespace["_runtime"](
            Spark("3.5.7-amzn-1"), expected_spark_version="3.5.6"
        )
    with pytest.raises(RuntimeError, match="Spark changed"):
        namespace["_runtime"](
            Spark("3.5.6-other-1"), expected_spark_version="3.5.6"
        )


def test_result_contract_separates_canary_from_production(tmp_path: Path) -> None:
    release, _, _, _ = _package_fixture(tmp_path)
    now = datetime.now(UTC)
    canary_auth = build_glue_global_dedup_authorization(
        release,
        mode="canary",
        approval_text=expected_glue_global_dedup_approval(release, mode="canary"),
        now=now,
    )
    canary_token = glue_global_dedup_job_token(canary_auth)
    canary_prefix = f"{release['output']['canary_prefix']}/{canary_token}"
    inventory = {
        "object_bytes": 10,
        "object_count": 1,
        "objects_sha256": "a" * 64,
    }
    canary = {
        "accepted_ids": {
            **inventory,
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{canary_prefix}/accepted-ids"
            ),
        },
        "authorization_sha256": canary_auth["authorization_sha256"],
        "deduplication": {
            "accepted_documents": 100,
            "anchor_documents": release["resume"]["accepted_documents"],
            "anchor_documents_without_usable_lsh_buckets": 0,
            "candidate_documents": 100,
            "candidate_documents_without_usable_lsh_buckets": 0,
            "documents_without_usable_lsh_buckets": 0,
            "documents_without_usable_lsh_buckets_policy": "keep_exact_only",
            "exact_rejected_documents": 0,
            "lsh_bucket_memberships": 900,
            "lsh_candidate_pairs": 0,
            "lsh_pair_upper_bound_after_stop_buckets": 0,
            "lsh_pair_upper_bound_before_stop_buckets": 0,
            "maximum_lsh_bucket_documents_observed": 1,
            "near_rejected_documents": 0,
            "oversized_lsh_buckets": 0,
            "oversized_lsh_memberships": 0,
            "oversized_lsh_pair_upper_bound_removed": 0,
            "pair_budget_lsh_buckets": 0,
            "pair_budget_lsh_memberships": 0,
            "pair_budget_lsh_pair_upper_bound_removed": 0,
        },
        "gate": "lumi_glue_production_global_dedup_canary_result_v0.1",
        "input_documents": 100,
        "input_index_sha256": release["input_index_sha256"],
        "input_tokens": 1_000,
        "job_token": canary_token,
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "mode": "canary",
        "rejected_ids": {
            **{**inventory, "objects_sha256": "b" * 64},
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{canary_prefix}/rejected-ids"
            ),
        },
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {},
        "schema_version": 1,
        "status": "pass_structural_canary",
    }
    assert validate_glue_global_dedup_result(
        canary, release, canary_auth
    ) == canary

    production_auth = build_glue_global_dedup_authorization(
        release,
        mode="production",
        approval_text=expected_glue_global_dedup_approval(
            release, mode="production"
        ),
        now=now,
    )
    production_token = glue_global_dedup_job_token(production_auth)
    production_prefix = (
        f"{release['output']['production_prefix']}/{production_token}"
    )
    categories = []
    selected_tokens = 0
    for row in release["categories"]:
        selected = int(row["floor_tokens"]) - int(row["accepted_tokens"])
        selected_tokens += selected
        categories.append(
            {
                "available_documents": 10,
                "available_tokens": selected,
                "category": row["category"],
                "deficit_tokens": 0,
                "final_documents": int(row["accepted_documents"]) + 1,
                "final_tokens": int(row["floor_tokens"]),
                "floor_tokens": int(row["floor_tokens"]),
                "selected_documents": 1,
                "selected_tokens": selected,
            }
        )
    production = {
        "accepted_candidates": {
            **inventory,
            "documents": 7,
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{production_prefix}/"
                "accepted-candidates"
            ),
            "tokens": selected_tokens,
        },
        "authorization_sha256": production_auth["authorization_sha256"],
        "categories": categories,
        "deduplication": {
            "accepted_documents": release["map"]["input_documents"],
            "anchor_documents": release["resume"]["accepted_documents"],
            "anchor_documents_without_usable_lsh_buckets": 0,
            "candidate_documents": release["map"]["input_documents"],
            "candidate_documents_without_usable_lsh_buckets": 0,
            "documents_without_usable_lsh_buckets": 0,
            "documents_without_usable_lsh_buckets_policy": "keep_exact_only",
            "exact_rejected_documents": 0,
            "lsh_bucket_memberships": 9 * release["map"]["input_documents"],
            "lsh_candidate_pairs": 0,
            "lsh_pair_upper_bound_after_stop_buckets": 0,
            "lsh_pair_upper_bound_before_stop_buckets": 0,
            "maximum_lsh_bucket_documents_observed": 1,
            "near_rejected_documents": 0,
            "oversized_lsh_buckets": 0,
            "oversized_lsh_memberships": 0,
            "oversized_lsh_pair_upper_bound_removed": 0,
            "pair_budget_lsh_buckets": 0,
            "pair_budget_lsh_memberships": 0,
            "pair_budget_lsh_pair_upper_bound_removed": 0,
        },
        "gate": "lumi_glue_production_global_dedup_result_v0.1",
        "input_index_sha256": release["input_index_sha256"],
        "job_token": production_token,
        "map_checkpoint_sha256": release["map"]["checkpoint_sha256"],
        "mode": "production",
        "rejected_ids": {
            **{**inventory, "objects_sha256": "b" * 64},
            "s3_uri": (
                f"s3://{release['storage']['bucket']}/{production_prefix}/rejected-ids"
            ),
        },
        "release_fingerprint": release["release_fingerprint"],
        "runtime": {},
        "schema_version": 1,
        "status": "pass_category_floors",
        "total_deficit_tokens": 0,
    }
    assert validate_glue_global_dedup_result(
        production, release, production_auth
    ) == production
    stop_bucket_result = deepcopy(production)
    oversized_pair_upper = 19_150 * 19_149 // 2
    stop_bucket_result["deduplication"].update(
        {
            "lsh_bucket_memberships": 19_150,
            "lsh_pair_upper_bound_before_stop_buckets": oversized_pair_upper,
            "maximum_lsh_bucket_documents_observed": 19_150,
            "oversized_lsh_buckets": 1,
            "oversized_lsh_memberships": 19_150,
            "oversized_lsh_pair_upper_bound_removed": oversized_pair_upper,
        }
    )
    assert validate_glue_global_dedup_result(
        stop_bucket_result, release, production_auth
    ) == stop_bucket_result
    excessive_stop_buckets = deepcopy(stop_bucket_result)
    excessive_stop_buckets["deduplication"]["oversized_lsh_buckets"] = 65
    with pytest.raises(ValueError, match="stop-bucket count cap"):
        validate_glue_global_dedup_result(
            excessive_stop_buckets, release, production_auth
        )
    uncovered = deepcopy(stop_bucket_result)
    uncovered["deduplication"].update(
        {
            "candidate_documents_without_usable_lsh_buckets": 4,
            "documents_without_usable_lsh_buckets": 4,
        }
    )
    assert validate_glue_global_dedup_result(
        uncovered, release, production_auth
    ) == uncovered
    inconsistent_uncovered = deepcopy(uncovered)
    inconsistent_uncovered["deduplication"][
        "documents_without_usable_lsh_buckets"
    ] = 5
    with pytest.raises(ValueError, match="inconsistent uncovered-LSH counts"):
        validate_glue_global_dedup_result(
            inconsistent_uncovered, release, production_auth
        )
    excessive_uncovered = deepcopy(uncovered)
    excessive_uncovered["deduplication"].update(
        {
            "candidate_documents_without_usable_lsh_buckets": 17,
            "documents_without_usable_lsh_buckets": 17,
        }
    )
    with pytest.raises(ValueError, match="without LSH coverage"):
        validate_glue_global_dedup_result(
            excessive_uncovered, release, production_auth
        )
    pair_budget_result = deepcopy(production)
    pair_budget_result["deduplication"].update(
        {
            "lsh_bucket_memberships": 1_000,
            "lsh_candidate_pairs": 10,
            "lsh_pair_upper_bound_after_stop_buckets": 40,
            "lsh_pair_upper_bound_before_stop_buckets": 140,
            "maximum_lsh_bucket_documents_observed": 20,
            "pair_budget_lsh_buckets": 1,
            "pair_budget_lsh_memberships": 20,
            "pair_budget_lsh_pair_upper_bound_removed": 100,
        }
    )
    assert validate_glue_global_dedup_result(
        pair_budget_result, release, production_auth
    ) == pair_budget_result
    inconsistent_pair_budget = deepcopy(pair_budget_result)
    inconsistent_pair_budget["deduplication"][
        "lsh_pair_upper_bound_after_stop_buckets"
    ] = 41
    with pytest.raises(ValueError, match="pair-budget accounting"):
        validate_glue_global_dedup_result(
            inconsistent_pair_budget, release, production_auth
        )
    changed = deepcopy(production)
    changed["categories"][0]["final_tokens"] -= 1
    with pytest.raises(ValueError, match="token conservation"):
        validate_glue_global_dedup_result(changed, release, production_auth)
