from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.glue_production_mapping import (
    _mapping_targets,
    advance_mapping_checkpoint,
    build_glue_production_mapping_authorization,
    build_initial_mapping_checkpoint,
    plan_mapping_wave,
    validate_glue_production_mapping_authorization,
    validate_glue_production_mapping_release,
    validate_mapping_checkpoint,
)

ROOT = Path(__file__).parents[1]


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
