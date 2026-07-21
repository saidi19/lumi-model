from __future__ import annotations

import gzip
import hashlib
import io
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.emr_production_staging import split_gzip_source
from lumi.glue_production_mapping import (
    RECOVERED_CHECKPOINT_FILE,
    RELEASE_FILE,
    _mapping_targets,
    advance_mapping_checkpoint,
    build_glue_production_mapping_authorization,
    build_glue_production_mapping_recovery_package,
    build_initial_mapping_checkpoint,
    plan_mapping_wave,
    validate_glue_production_mapping_authorization,
    validate_glue_production_mapping_release,
    validate_mapping_checkpoint,
)
from lumi.production_corpus import _canonical_json_bytes

ROOT = Path(__file__).parents[1]


def _refingerprint_release(release: dict[str, object]) -> None:
    material = dict(release)
    material.pop("release_fingerprint", None)
    release["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _wheel_pin() -> dict[str, object]:
    archive = "a" * 64
    prefix = (
        "s3://lumi-corpus-prod-fixture/lumi-corpus/v0.1/control/distributed/"
        f"glue-cp311/{archive[:12]}/codebuild/output"
    )
    return {
        "codebuild": {
            "build_id": f"lumi-cb-lingua-cp311-{archive[:12]}:fixture",
            "build_report_s3_uri": f"{prefix}/lingua-glue-cp311-build-v0.1.json",
            "build_report_sha256": "b" * 64,
            "build_report_version_id": "build-version",
            "build_status": "SUCCEEDED",
            "parity_report_s3_uri": f"{prefix}/lingua-glue-cp311-parity-v0.1.json",
            "parity_report_sha256": "c" * 64,
            "parity_report_version_id": "parity-version",
            "source_archive_sha256": archive,
            "source_commit": "d" * 40,
            "source_version": "source-version",
        },
        "gate": "lumi_lingua_glue_cp311_wheel_pin_v0.1",
        "schema_version": 1,
        "wheel": {
            "bytes": 170_336_000,
            "s3_uri": (
                f"{prefix}/"
                "lingua_language_detector-2.2.0-cp311-cp311-manylinux_2_34_x86_64.whl"
            ),
            "sha256": "e" * 64,
            "version_id": "wheel-version",
        },
    }


def _category_plan() -> dict[str, object]:
    floors = [36, 24, 18, 18, 12, 6, 6]
    names = [
        "science_and_education",
        "code",
        "reference",
        "questions_and_discussion",
        "open_web",
        "books",
        "government_legal_and_patents",
    ]
    return {
        "categories": [
            {
                "accepted_tokens": 237_769_640 if index == 0 else 0,
                "category": name,
                "floor_tokens": floor * 1_000_000_000,
                "ordinal": index,
            }
            for index, (name, floor) in enumerate(zip(names, floors, strict=True))
        ]
    }


def _release() -> dict[str, object]:
    plan = _category_plan()
    categories = _mapping_targets(plan, 150_000_000_000)
    names = [row["category"] for row in categories]
    schedule = []
    staged = []
    counts = {name: 0 for name in names}
    for index in range(758):
        name = names[index % len(names)]
        counts[name] += 1
        digest = f"{index + 1:064x}"
        schedule.append(
            {
                "category": name,
                "component": "fixture",
                "dataset_id": "fixture/data",
                "ordinal": index,
                "revision": "f" * 40,
                "source_bytes": 1_000_000,
                "source_file": f"fixture-{index:04d}.json.gz",
                "source_provenance": {},
                "source_sha256": digest,
            }
        )
        staged.append(
            {
                "category": name,
                "chunk_count": 1,
                "component": "fixture",
                "manifest_s3_key": (
                    "lumi-corpus/v0.1/transient/glue-production/raw/stage/"
                    f"manifests/{digest}/{digest}.json"
                ),
                "manifest_sha256": digest,
                "manifest_version_id": f"version-{index}",
                "source_bytes": 1_000_000,
                "source_complete": True,
                "source_file": f"fixture-{index:04d}.json.gz",
                "source_ordinal": index,
                "source_sha256": digest,
                "staged_compressed_bytes": 1,
                "staged_rows": 1,
                "staged_uncompressed_bytes": 1,
                "start_after_line": 38_848 if index == 0 else 0,
            }
        )
    value: dict[str, object] = {
        "artifacts": {},
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_glue_dedup": False,
            "paid_glue_mapping": False,
            "paid_glue_sharding": False,
            "paid_glue_staging": False,
            "paid_gpu_compute": False,
            "production_candidate_acquisition": True,
        },
        "budget": {
            "hard_total_usd": "4000.00",
            "maximum_glue_compute_usd": "3600.00",
            "maximum_s3_logs_requests_usd": "200.00",
            "reserve_usd": "200.00",
        },
        "categories": categories,
        "category_inventory": [
            {"category": name, "staged_chunks": counts[name]} for name in names
        ],
        "data_quality": {},
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_mapping": "not_authorized",
            "status": "mapping_package_ready_waiting_for_exact_authorization",
        },
        "filters": {},
        "gate": "lumi_glue_production_map_release_v0.1",
        "implementation": {},
        "job": {
            "additional_python_modules": "fixture",
            "execution_class": "STANDARD",
            "glue_version": "5.1",
            "hard_maximum_dpu": 100,
            "maximum_concurrent_runs": 1,
            "number_of_workers": 100,
            "python_version": "3.11",
            "soft_stop_minutes": 3_960,
            "timeout_minutes": 4_320,
            "wave_maximum_chunks": 500,
            "worker_type": "G.1X",
        },
        "limits": {
            "maximum_category_shortfall_tokens": 100_000_000,
            "maximum_compute_usd": "3168.000000",
            "maximum_mapped_parquet_bytes": 1_500_000_000_000,
            "target_new_candidate_tokens": 149_762_230_360,
            "target_total_candidate_tokens": 150_000_000_000,
        },
        "lingua_compatibility": _wheel_pin(),
        "output": {
            "checkpoint_prefix": "lumi-corpus/v0.1/control/map-checkpoints",
            "mapped_prefix": "lumi-corpus/v0.1/transient/mapped",
            "result_prefix": "lumi-corpus/v0.1/control/map-runs",
            "temp_prefix": "lumi-corpus/v0.1/control/map-temp",
        },
        "region": "us-east-1",
        "resume": {
            "accepted_documents": 38_349,
            "accepted_tokens": 237_769_640,
            "source_next_line": 38_848,
            "source_ordinal": 0,
        },
        "schema_version": 1,
        "source_commit": "1" * 40,
        "source_schedule": schedule,
        "stage": {
            "release_fingerprint": "2" * 64,
            "release_sha256": "3" * 64,
            "result_sha256": "4" * 64,
            "staged_chunks": 758,
            "staged_compressed_bytes": 500_000_000_000,
            "staged_rows": 758,
            "verification_sha256": "5" * 64,
        },
        "staged_sources": staged,
        "storage": {
            "bucket": "lumi-corpus-prod-fixture",
            "namespace": "lumi-corpus/v0.1",
        },
    }
    value["release_fingerprint"] = hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return value


def test_mapping_release_targets_100_dpu_and_total_150b() -> None:
    release = _release()
    verified = validate_glue_production_mapping_release(release)
    assert verified["status"] == "verified_not_authorized"
    assert verified["target_total_candidate_tokens"] == 150_000_000_000
    assert release["job"]["hard_maximum_dpu"] == 100
    assert release["job"]["number_of_workers"] == 100
    assert release["job"]["worker_type"] == "G.1X"
    assert release["limits"]["maximum_compute_usd"] == "3168.000000"

    changed = deepcopy(release)
    changed["job"]["hard_maximum_dpu"] = 512
    material = dict(changed)
    material.pop("release_fingerprint")
    changed["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="DPU target changed"):
        validate_glue_production_mapping_release(changed)


def test_mapping_checkpoint_plans_and_commits_a_hard_capped_wave() -> None:
    release = _release()
    checkpoint = build_initial_mapping_checkpoint(release)
    assert validate_mapping_checkpoint(checkpoint, release)["status"] == "mapping"
    wave = plan_mapping_wave(checkpoint, release)
    assert wave["chunk_count"] == 500
    assert len(wave["categories"]) == 7
    results = []
    for row in wave["categories"]:
        target = int(row["remaining_target_tokens"])
        results.append(
            {
                "available_documents": 1,
                "available_tokens": target,
                "candidate_documents": 1,
                "candidate_tokens": target - 1,
                "category": row["category"],
                "closed": True,
                "cutoff": {"document_id": row["category"]},
                "end_chunk_index": row["end_chunk_index"],
                "object_count": 1,
                "objects_sha256": "6" * 64,
                "parquet_bytes": 100,
                "parquet_s3_uri": (
                    "s3://lumi-corpus-prod-fixture/lumi-corpus/v0.1/transient/"
                    f"mapped/{release['release_fingerprint']}/run/{row['category']}"
                ),
                "start_chunk_index": row["start_chunk_index"],
            }
        )
    manifest: dict[str, object] = {
        "candidate_documents": 7,
        "candidate_tokens": sum(row["candidate_tokens"] for row in results),
        "categories": results,
        "gate": "lumi_glue_production_map_wave_result_v0.1",
        "manifest_s3_key": "lumi-corpus/v0.1/control/map-runs/wave-1.json",
        "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parquet_bytes": 700,
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "wave_plan_sha256": wave["wave_sha256"],
        "wave_sequence": 1,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    committed = advance_mapping_checkpoint(checkpoint, release, manifest)
    assert committed["status"] == "ready_for_global_dedup"
    assert committed["mapped_parquet_bytes"] == 700
    assert committed["total_mapped_tokens"] + release["resume"]["accepted_tokens"] < (
        150_000_000_000
    )
    assert 150_000_000_000 - (
        committed["total_mapped_tokens"] + release["resume"]["accepted_tokens"]
    ) == 7

    changed = deepcopy(manifest)
    changed["categories"][0]["candidate_tokens"] += 2
    with pytest.raises(ValueError, match="fingerprint changed"):
        advance_mapping_checkpoint(checkpoint, release, changed)


def _open_web_exhaustion_fixture(
    *,
    open_web_tokens: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    release = _release()
    for row in release["category_inventory"]:
        if row["category"] == "open_web":
            row["staged_chunks"] = 1
    _refingerprint_release(release)
    validate_glue_production_mapping_release(release)
    checkpoint = build_initial_mapping_checkpoint(release)
    wave = plan_mapping_wave(checkpoint, release)
    assert next(row for row in wave["categories"] if row["category"] == "open_web")[
        "end_chunk_index"
    ] == 1
    results = []
    for row in wave["categories"]:
        selected_tokens = (
            open_web_tokens if row["category"] == "open_web" else 1
        )
        results.append(
            {
                "available_documents": 1,
                "available_tokens": selected_tokens,
                "candidate_documents": 1,
                "candidate_tokens": selected_tokens,
                "category": row["category"],
                "closed": False,
                "cutoff": None,
                "end_chunk_index": row["end_chunk_index"],
                "object_count": 1,
                "objects_sha256": "8" * 64,
                "parquet_bytes": 1,
                "parquet_s3_uri": (
                    "s3://lumi-corpus-prod-fixture/lumi-corpus/v0.1/transient/"
                    f"mapped/{release['release_fingerprint']}/run/{row['category']}"
                ),
                "start_chunk_index": row["start_chunk_index"],
            }
        )
    manifest: dict[str, object] = {
        "candidate_documents": len(results),
        "candidate_tokens": sum(int(row["candidate_tokens"]) for row in results),
        "categories": results,
        "gate": "lumi_glue_production_map_wave_result_v0.1",
        "manifest_s3_key": "lumi-corpus/v0.1/control/map-runs/wave-1.json",
        "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parquet_bytes": len(results),
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "wave_plan_sha256": wave["wave_sha256"],
        "wave_sequence": 1,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    return release, checkpoint, manifest


def test_mapping_reallocates_a_floor_satisfying_exhausted_category() -> None:
    release, checkpoint, manifest = _open_web_exhaustion_fixture(
        open_web_tokens=13_000_000_000
    )
    committed = advance_mapping_checkpoint(checkpoint, release, manifest)
    assert committed["status"] == "mapping"
    assert len(committed["target_reallocations"]) == 1
    event = committed["target_reallocations"][0]
    assert event["exhausted_category"] == "open_web"
    assert event["shortfall_tokens"] == 2_005_956_042
    assert sum(row["added_tokens"] for row in event["recipient_adjustments"]) == event[
        "shortfall_tokens"
    ]
    open_web = next(
        row for row in committed["categories"] if row["category"] == "open_web"
    )
    assert open_web == {
        "category": "open_web",
        "closed": True,
        "cutoff": None,
        "mapped_documents": 1,
        "mapped_tokens": 13_000_000_000,
        "next_chunk_index": 1,
        "target_tokens": 13_000_000_000,
    }
    assert sum(row["target_tokens"] for row in committed["categories"]) == release[
        "limits"
    ]["target_new_candidate_tokens"]
    assert "open_web" not in {
        row["category"] for row in plan_mapping_wave(committed, release)["categories"]
    }

    tampered = deepcopy(committed)
    tampered["target_reallocations"][0]["shortfall_tokens"] += 1
    material = dict(tampered)
    material.pop("checkpoint_sha256")
    tampered["checkpoint_sha256"] = hashlib.sha256(
        _canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(ValueError, match="reallocation fingerprint changed"):
        validate_mapping_checkpoint(tampered, release)


def test_mapping_refuses_exhaustion_below_the_clean_floor() -> None:
    release, checkpoint, manifest = _open_web_exhaustion_fixture(
        open_web_tokens=11_000_000_000
    )
    with pytest.raises(ValueError, match="exhausted below its clean floor"):
        advance_mapping_checkpoint(checkpoint, release, manifest)


def test_mapping_recovery_package_migrates_one_verified_wave(tmp_path: Path) -> None:
    release, checkpoint, manifest = _open_web_exhaustion_fixture(
        open_web_tokens=13_000_000_000
    )
    previous_package = tmp_path / "previous"
    previous_package.mkdir()
    release_path = previous_package / RELEASE_FILE
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    package_checkpoint = previous_package / "initial-map-checkpoint-v0.1.json"
    package_checkpoint.write_bytes(checkpoint_path.read_bytes())
    (previous_package / "lumi-glue-production-map-code-v0.1.zip").write_bytes(b"zip")
    (previous_package / "run_glue_production_mapping.py").write_text("# fixture\n")
    packaged = [
        release_path,
        package_checkpoint,
        previous_package / "lumi-glue-production-map-code-v0.1.zip",
        previous_package / "run_glue_production_mapping.py",
    ]
    (previous_package / "manifest.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in packaged
        )
    )
    wave_path = tmp_path / "wave.json"
    wave_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    failure = {
        "active_replacement_runs": 0,
        "failure": {"category": "open_web"},
        "gate": "lumi_glue_production_mapping_failure_evidence_v0.1",
        "job_name": "fixture-failed-job",
        "job_run_id": "jr_fixture",
        "last_durable_checkpoint": {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "documents": checkpoint["total_mapped_documents"],
            "local_file_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "newly_mapped_tokens": checkpoint["total_mapped_tokens"],
            "parquet_bytes": checkpoint["mapped_parquet_bytes"],
            "sequence": checkpoint["checkpoint_sequence"],
        },
        "recoverable_wave_7": {
            "candidate_documents": manifest["candidate_documents"],
            "candidate_tokens": manifest["candidate_tokens"],
            "local_file_sha256": hashlib.sha256(wave_path.read_bytes()).hexdigest(),
            "manifest_sha256": manifest["manifest_sha256"],
            "parquet_bytes": manifest["parquet_bytes"],
            "verified_s3_bytes": manifest["parquet_bytes"],
            "verified_s3_positive_objects": sum(
                row["object_count"] for row in manifest["categories"]
            ),
        },
        "schema_version": 1,
        "state": "FAILED",
    }
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "recovery"
    package = build_glue_production_mapping_recovery_package(
        previous_package=previous_package,
        parent_checkpoint_path=checkpoint_path,
        recovered_wave_manifest_path=wave_path,
        failure_evidence_path=failure_path,
        repository_root=ROOT,
        source_commit="2" * 40,
        output_directory=output,
    )
    recovered_release = package["release"]
    recovered_checkpoint = package["checkpoint"]
    assert recovered_release["release_fingerprint"] != release["release_fingerprint"]
    assert recovered_checkpoint["checkpoint_sequence"] == 1
    assert recovered_checkpoint["release_migration"]["from_release_fingerprint"] == release[
        "release_fingerprint"
    ]
    assert recovered_checkpoint["total_mapped_tokens"] == manifest["candidate_tokens"]
    assert validate_mapping_checkpoint(recovered_checkpoint, recovered_release)[
        "status"
    ] == "mapping"
    with pytest.raises(ValueError, match="requires its migrated checkpoint"):
        build_initial_mapping_checkpoint(recovered_release)
    assert (output / RECOVERED_CHECKPOINT_FILE).is_file()


def test_mapping_authorization_is_one_run_and_mapping_only() -> None:
    release = _release()
    now = datetime(2026, 7, 21, 12, tzinfo=UTC)
    authorization = build_glue_production_mapping_authorization(release, now=now)
    assert authorization["bounds"]["hard_maximum_dpu"] == 100
    assert authorization["bounds"]["target_total_candidate_tokens"] == 150_000_000_000
    assert authorization["permissions"]["paid_glue_mapping"] is True
    assert authorization["permissions"]["model_training"] is False
    assert authorization["bounds"]["starting_checkpoint_sequence"] == 0
    assert authorization["bounds"]["starting_checkpoint_sha256"] == (
        build_initial_mapping_checkpoint(release)["checkpoint_sha256"]
    )
    assert validate_glue_production_mapping_authorization(
        authorization, release, now=now
    )["status"] == "authorized_one_100_dpu_mapping_run"
    with pytest.raises(ValueError, match="not currently valid"):
        validate_glue_production_mapping_authorization(
            authorization, release, now=now + timedelta(hours=25)
        )


def test_mapping_resume_authorization_binds_exact_open_checkpoint() -> None:
    release = _release()
    counts = {row["category"]: 0 for row in release["categories"]}
    for staged in release["staged_sources"]:
        staged["chunk_count"] = 5
        counts[staged["category"]] += 5
    release["category_inventory"] = [
        {"category": row["category"], "staged_chunks": counts[row["category"]]}
        for row in release["categories"]
    ]
    release["stage"]["staged_chunks"] = sum(counts.values())
    material = dict(release)
    material.pop("release_fingerprint")
    release["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    validate_glue_production_mapping_release(release)

    initial = build_initial_mapping_checkpoint(release)
    wave = plan_mapping_wave(initial, release)
    results = [
        {
            "available_documents": 1,
            "available_tokens": 1,
            "candidate_documents": 1,
            "candidate_tokens": 1,
            "category": row["category"],
            "closed": False,
            "cutoff": None,
            "end_chunk_index": row["end_chunk_index"],
            "object_count": 1,
            "objects_sha256": "7" * 64,
            "parquet_bytes": 1,
            "parquet_s3_uri": (
                "s3://lumi-corpus-prod-fixture/lumi-corpus/v0.1/transient/"
                f"mapped/{release['release_fingerprint']}/run/{row['category']}"
            ),
            "start_chunk_index": row["start_chunk_index"],
        }
        for row in wave["categories"]
    ]
    manifest: dict[str, object] = {
        "candidate_documents": len(results),
        "candidate_tokens": len(results),
        "categories": results,
        "gate": "lumi_glue_production_map_wave_result_v0.1",
        "manifest_s3_key": "lumi-corpus/v0.1/control/map-runs/wave-1.json",
        "parent_checkpoint_sha256": initial["checkpoint_sha256"],
        "parquet_bytes": len(results),
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "wave_plan_sha256": wave["wave_sha256"],
        "wave_sequence": 1,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    checkpoint = advance_mapping_checkpoint(initial, release, manifest)
    assert checkpoint["status"] == "mapping"
    assert checkpoint["checkpoint_sequence"] == 1

    now = datetime(2026, 7, 21, 12, tzinfo=UTC)
    authorization = build_glue_production_mapping_authorization(
        release, checkpoint=checkpoint, now=now
    )
    assert authorization["bounds"]["starting_checkpoint_sequence"] == 1
    assert authorization["bounds"]["starting_checkpoint_sha256"] == checkpoint[
        "checkpoint_sha256"
    ]
    assert validate_glue_production_mapping_authorization(
        authorization, release, checkpoint=checkpoint, now=now
    )["status"] == "authorized_one_100_dpu_mapping_run"

    other = initial
    with pytest.raises(ValueError, match="bounds changed"):
        validate_glue_production_mapping_authorization(
            authorization, release, checkpoint=other, now=now
        )


def test_mapping_script_uses_reserved_job_run_id_and_initializes_spark_first() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_mapping.py",
        run_name="lumi_glue_production_mapping_test",
    )
    requested = namespace["GLUE_REQUESTED_OPTIONS"]
    assert requested == (
        "JOB_NAME",
        "LUMI_AUTHORIZATION_FILE",
        "LUMI_CHECKPOINT_FILE",
        "LUMI_RELEASE_FILE",
    )
    assert "JOB_RUN_ID" not in requested
    source = (ROOT / "scripts" / "run_glue_production_mapping.py").read_text()
    context = source.index("context = SparkContext.getOrCreate()")
    release = source.index('SparkFiles.get(args["LUMI_RELEASE_FILE"])')
    assert context < release


def test_mapping_script_reattaches_verified_staged_manifest_digest(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.json.gz"
    with gzip.open(source_path, "wb") as handle:
        handle.write(json.dumps({"text": "verified staged document"}).encode() + b"\n")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = split_gzip_source(
        source_path,
        tmp_path / "chunks",
        source_file=source_path.name,
        source_sha256=source_sha256,
        start_after_line=0,
        target_uncompressed_bytes=1024,
    )
    bucket = "fixture-bucket"
    prefix = "lumi-corpus/v0.1/transient/glue-production/raw/release"
    remote = {
        **manifest,
        "chunks": [
            {
                **chunk,
                "s3_key": f"{prefix}/chunks/{source_sha256}/{chunk['sha256']}.jsonl.zst",
                "version_id": "chunk-version",
            }
            for chunk in manifest["chunks"]
        ],
    }
    remote.pop("manifest_sha256", None)
    body = _canonical_json_bytes(remote)
    digest = hashlib.sha256(body).hexdigest()
    manifest_key = f"{prefix}/manifests/{source_sha256}/{digest}.json"
    release = {
        "categories": [{"category": "science_and_education"}],
        "category_inventory": [
            {
                "category": "science_and_education",
                "staged_chunks": len(remote["chunks"]),
            }
        ],
        "source_schedule": [
            {
                "category": "science_and_education",
                "ordinal": 0,
                "source_file": source_path.name,
                "source_sha256": source_sha256,
            }
        ],
        "staged_sources": [
            {
                "manifest_s3_key": manifest_key,
                "manifest_sha256": digest,
                "manifest_version_id": "manifest-version",
                "source_ordinal": 0,
                "start_after_line": 0,
            }
        ],
        "storage": {"bucket": bucket},
    }

    class FakeS3:
        def get_object(self, **request: object) -> dict[str, io.BytesIO]:
            assert request == {
                "Bucket": bucket,
                "Key": manifest_key,
                "VersionId": "manifest-version",
            }
            return {"Body": io.BytesIO(body)}

    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_mapping.py",
        run_name="lumi_glue_production_mapping_digest_test",
    )
    queues = namespace["_load_stage_queues"](release, s3_client=FakeS3())
    assert len(queues["science_and_education"]) == len(remote["chunks"])


def test_mapping_launcher_waits_for_new_role_and_keys_retry_job_by_authorization() -> None:
    source = (ROOT / "scripts" / "launch_glue_production_mapping.py").read_text()
    assert 'authorization_short = str(authorization["authorization_sha256"])[:8]' in source
    assert "-{authorization_short}" in source
    assert "if role_was_created:\n        time.sleep(60)" in source


def test_mapping_launcher_grants_wheel_read_without_adding_it_as_an_extra_file() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "launch_glue_production_mapping.py",
        run_name="lumi_glue_production_mapping_launcher_test",
    )
    release = _release()
    bucket = release["storage"]["bucket"]
    artifact_prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/test-artifacts"
    for name in (
        "benchmark_registry",
        "frozen_tokenizer",
        "training_exclusion_ledger",
        "lingua_cp311_wheel",
    ):
        release["artifacts"][name] = {"s3_uri": f"{artifact_prefix}/{name}"}

    artifact_keys = namespace["_release_artifact_keys"](release, bucket=bucket)
    wheel_key = "lumi-corpus/v0.1/control/test-artifacts/lingua_cp311_wheel"
    assert wheel_key in artifact_keys["exact_reads"]
    assert wheel_key not in artifact_keys["extra_files"]

    policy = namespace["_policy"](
        account="123456789012",
        region="us-east-1",
        bucket=bucket,
        exact_read_keys=artifact_keys["exact_reads"],
        staged_prefix="lumi-corpus/v0.1/transient/staged",
        checkpoint_prefix="lumi-corpus/v0.1/control/checkpoints",
        mapped_prefix="lumi-corpus/v0.1/transient/mapped",
        result_prefix="lumi-corpus/v0.1/control/results",
        temp_prefix="lumi-corpus/v0.1/control/temp",
    )
    exact_reads = next(
        statement
        for statement in policy["Statement"]
        if statement["Sid"] == "ReadExactMappingBootstrap"
    )
    assert f"arn:aws:s3:::{bucket}/{wheel_key}" in exact_reads["Resource"]
