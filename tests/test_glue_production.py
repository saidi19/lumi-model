from __future__ import annotations

import hashlib
import json
import runpy
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.glue_production import (
    OWNER_APPROVAL_TEXT,
    STAGE_RUNTIME_IMPORT_SMOKE,
    _read_contract,
    build_glue_production_stage_authorization,
    validate_glue_production_stage_authorization,
    validate_glue_production_stage_release,
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


def _release() -> dict[str, object]:
    sources = [
        {
            "category": "science_and_education",
            "component": "fixture",
            "dataset_id": "fixture/data",
            "ordinal": index,
            "revision": "f" * 40,
            "source_bytes": 1_000_000,
            "source_file": f"fixture-{index:04d}.json.gz",
            "source_sha256": f"{index + 1:064x}",
            "start_after_line": 38_848 if index == 0 else 0,
        }
        for index in range(758)
    ]
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
        "candidate_target_tokens": 150_000_000_000,
        "gate": "lumi_glue_production_stage_release_v0.1",
        "implementation": {},
        "job": {
            "additional_python_modules": "fixture",
            "auto_scaling": False,
            "execution_class": "STANDARD",
            "glue_version": "5.1",
            "maximum_concurrent_runs": 1,
            "number_of_workers": 100,
            "python_version": "3.11",
            "timeout_minutes": 360,
            "worker_type": "G.1X",
        },
        "limits": {
            "hard_logical_storage_bytes": 2_000_000_000_000,
            "maximum_compute_usd": "264.000000",
            "maximum_source_compressed_bytes": 4_000_000_000,
            "source_files": 758,
            "target_chunk_uncompressed_bytes": 134_217_728,
        },
        "lingua_compatibility": _wheel_pin(),
        "output": {
            "result_prefix": "lumi-corpus/v0.1/control/stage-results",
            "staged_prefix": "lumi-corpus/v0.1/transient/raw",
        },
        "region": "us-east-1",
        "resume": {
            "accepted_documents": 38_349,
            "accepted_tokens": 237_769_640,
            "source_next_line": 38_848,
            "source_ordinal": 0,
        },
        "runtime_import_smoke": dict(STAGE_RUNTIME_IMPORT_SMOKE),
        "schema_version": 1,
        "source_commit": "1" * 40,
        "source_inventory": {
            "compressed_source_bytes": 758_000_000,
            "data_files": 758,
        },
        "storage": {
            "bucket": "lumi-corpus-prod-fixture",
            "namespace": "lumi-corpus/v0.1",
        },
        "sources": sources,
    }
    value["release_fingerprint"] = hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return value


def test_production_contract_binds_150b_and_four_thousand_dollar_guard() -> None:
    contract = _read_contract(
        ROOT / "configs" / "glue-production-150b-v0.1.json", ROOT
    )
    assert contract["limits"]["candidate_target_tokens"] == 150_000_000_000
    assert contract["budget"]["hard_total_usd"] == "4000.00"
    assert contract["staging"]["hard_maximum_dpu"] == 100
    assert contract["subsequent_phases"]["mapping_maximum_dpu"] == 512


def test_stage_release_is_non_authorizing_and_binds_all_sources() -> None:
    release = _release()
    verified = validate_glue_production_stage_release(release)
    assert verified["status"] == "verified_not_authorized"
    assert verified["source_files"] == 758
    assert verified["maximum_compute_usd"] == "264.000000"

    smoke_tampered = deepcopy(release)
    smoke_tampered["runtime_import_smoke"]["status"] = "fail"
    material = dict(smoke_tampered)
    material.pop("release_fingerprint")
    smoke_tampered["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="archive import smoke changed"):
        validate_glue_production_stage_release(smoke_tampered)

    tampered = deepcopy(release)
    tampered["sources"][0]["start_after_line"] = 0
    material = dict(tampered)
    material.pop("release_fingerprint")
    tampered["release_fingerprint"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="resume line changed"):
        validate_glue_production_stage_release(tampered)


def test_recorded_owner_approval_authorizes_only_one_stage_run() -> None:
    now = datetime(2026, 7, 21, 12, tzinfo=UTC)
    release = _release()
    authorization = build_glue_production_stage_authorization(release, now=now)
    assert authorization["approval_text"] == OWNER_APPROVAL_TEXT
    assert authorization["permissions"]["paid_glue_staging"] is True
    assert authorization["permissions"]["paid_glue_mapping"] is False
    assert authorization["bounds"]["authorized_runs"] == 1
    assert validate_glue_production_stage_authorization(
        authorization, release, now=now
    )["status"] == "authorized_one_stage_run"

    with pytest.raises(ValueError, match="not currently valid"):
        validate_glue_production_stage_authorization(
            authorization, release, now=now + timedelta(hours=25)
        )


def test_stage_script_does_not_redeclare_reserved_job_run_id() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "run_glue_production_stage.py",
        run_name="lumi_glue_production_stage_test",
    )
    requested = namespace["GLUE_REQUESTED_OPTIONS"]
    assert requested == (
        "JOB_NAME",
        "LUMI_AUTHORIZATION_FILE",
        "LUMI_RELEASE_FILE",
    )
    assert "JOB_RUN_ID" not in requested


def test_stage_script_initializes_spark_before_resolving_extra_files() -> None:
    source = (ROOT / "scripts" / "run_glue_production_stage.py").read_text()
    context = source.index("context = SparkContext.getOrCreate()")
    release = source.index('SparkFiles.get(args["LUMI_RELEASE_FILE"])')
    authorization = source.index('SparkFiles.get(args["LUMI_AUTHORIZATION_FILE"])')
    assert context < release
    assert context < authorization


def test_stage_result_binds_every_source_and_aggregate() -> None:
    namespace = runpy.run_path(
        ROOT / "scripts" / "verify_glue_production_stage.py",
        run_name="lumi_glue_production_stage_verifier_test",
    )
    validate_stage_result = namespace["validate_stage_result"]
    release = _release()
    now = datetime(2026, 7, 21, 12, tzinfo=UTC)
    authorization = build_glue_production_stage_authorization(release, now=now)
    prefix = f"{release['output']['staged_prefix']}/{release['release_fingerprint']}"
    rows = []
    for source in release["sources"]:
        manifest_sha = hashlib.sha256(
            f"manifest-{source['ordinal']}".encode()
        ).hexdigest()
        rows.append(
            {
                "category": source["category"],
                "chunk_count": 1,
                "component": source["component"],
                "manifest_s3_key": (
                    f"{prefix}/manifests/{source['source_sha256']}/"
                    f"{manifest_sha}.json"
                ),
                "manifest_sha256": manifest_sha,
                "manifest_version_id": f"version-{source['ordinal']}",
                "reused": False,
                "source_bytes": source["source_bytes"],
                "source_complete": True,
                "source_file": source["source_file"],
                "source_ordinal": source["ordinal"],
                "source_sha256": source["source_sha256"],
                "staged_compressed_bytes": 1,
                "staged_rows": 1,
                "staged_uncompressed_bytes": 1,
                "start_after_line": source["start_after_line"],
            }
        )
    result = {
        "authorization_sha256": authorization["authorization_sha256"],
        "elapsed_seconds": 10.0,
        "gate": "lumi_glue_production_stage_result_v0.1",
        "job_name": "lumi-glue-stage-fixture",
        "job_run_id": "jr_fixture",
        "release_fingerprint": release["release_fingerprint"],
        "schema_version": 1,
        "source_commit": release["source_commit"],
        "source_files": len(rows),
        "sources": rows,
        "staged_chunks": len(rows),
        "staged_compressed_bytes": len(rows),
        "staged_rows": len(rows),
        "staged_uncompressed_bytes": len(rows),
        "status": "pass",
        "training_started": False,
    }
    assert validate_stage_result(result, release, authorization)["status"] == (
        "driver_result_verified"
    )
    changed = deepcopy(result)
    changed["staged_chunks"] += 1
    with pytest.raises(ValueError, match="aggregate totals changed"):
        validate_stage_result(changed, release, authorization)
