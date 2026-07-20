from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lumi.emr_serverless_canary import (
    BASE_IMAGE_DIGEST,
    BASE_IMAGE_URI,
    BUILDSPEC_SHA256,
    CODEBUILD_STANDARD_IMAGE,
    EMR_CANARY_APPROVAL_TEXT,
    _cost_envelope,
    _release_fingerprint,
    _require_clean_source_commit,
    _validate_contract,
    bind_emr_serverless_image,
    build_codebuild_project_request,
    build_codebuild_service_policy,
    build_codebuild_start_request,
    build_emr_serverless_application_request,
    build_emr_serverless_job_request,
    validate_emr_serverless_canary_authorization,
    validate_emr_serverless_canary_release,
)
from lumi.s3_staged_corpus import build_staged_object

ROOT = Path(__file__).parents[1]


def _sources() -> list[dict[str, object]]:
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
    sources = [
        {
            "category": category,
            "component": f"component_{index}",
            "dataset_id": "common-pile/fixture",
            "maximum_records": 2_000,
            "ordinal": index,
            "revision": "b" * 40,
            "source_bytes": 1_000,
            "source_file": f"fixture-{index}.json.gz",
            "source_sha256": f"{index + 1:064x}",
        }
        for index, category in enumerate(categories)
    ]
    for source in sources:
        source["staged_object"] = build_staged_object(
            source,
            bucket="lumi-corpus-fixture",
            key_prefix="lumi-corpus/v0.1/raw/source-objects/sha256",
        )
    return sources


def _synthetic_release() -> dict[str, object]:
    bucket = "lumi-corpus-fixture"
    prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/distributed/emr-serverless"
    artifacts = {
        "benchmark_registry": {"s3_uri": f"{prefix}/benchmark/signatures.jsonl.gz"},
        "code_archive": {"s3_uri": f"{prefix}/code/lumi.zip", "sha256": "1" * 64},
        "frozen_tokenizer": {"s3_uri": f"{prefix}/tokenizer/tokenizer.json"},
        "image_context": {
            "s3_uri": f"{prefix}/image/context.zip",
            "sha256": "2" * 64,
        },
        "job_script": {"s3_uri": f"{prefix}/jobs/run.py"},
        "resume_anchors": {"s3_uri": f"{prefix}/anchors/resume.jsonl.gz"},
        "training_exclusion_ledger": {"s3_uri": f"{prefix}/ledger/exclusions.jsonl"},
    }
    release: dict[str, object] = {
        "application": {
            "architecture": "X86_64",
            "auto_start": True,
            "auto_stop_idle_timeout_minutes": 5,
            "initial_capacity_workers": 0,
            "maximum_disk_gib": 80,
            "maximum_memory_gib": 64,
            "maximum_vcpu": 16,
        },
        "artifacts": artifacts,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_codebuild_image": False,
            "paid_emr_serverless_canary": False,
            "paid_emr_serverless_production": False,
            "paid_gpu_compute": False,
        },
        "canary": {
            "maximum_records_per_source": 2_000,
            "maximum_runtime_minutes": 30,
            "maximum_source_compressed_bytes": 700_000_000,
            "maximum_total_source_compressed_bytes": 1_500_000_000,
            "requires_new_owner_approval": True,
            "source_files": 8,
        },
        "cost": {
            "codebuild_compute_hard_maximum_usd": "0.15",
            "combined_compute_hard_maximum_usd": "0.755904",
            "emr_compute_hard_maximum_usd": "0.605904",
            "memory_gb_hour_usd": "0.0057785",
            "standard_storage_extra_gb_hour_usd": "0.000111",
            "vcpu_hour_usd": "0.052624",
        },
        "gate": "lumi_distributed_emr_serverless_canary_release_v0.1",
        "image": {
            "base_digest": BASE_IMAGE_DIGEST,
            "base_uri": BASE_IMAGE_URI,
            "buildspec_sha256": BUILDSPEC_SHA256,
            "context_sha256": "c" * 64,
            "custom_image_bytes": None,
            "custom_image_digest": None,
            "custom_image_uri": None,
            "maximum_image_gib": 10,
            "source_release_fingerprint": None,
            "state": "build_not_authorized",
        },
        "implementation": {
            "containers/emr-serverless-7.13-python312/Dockerfile": "3" * 64,
            "containers/emr-serverless-7.13-python312/requirements-linux-x86_64.lock": "4"
            * 64,
            "containers/emr-serverless-7.13-python312/requirements-staging-linux-x86_64.lock": "5"
            * 64,
            "containers/emr-serverless-7.13-python312/verify_runtime.py": "6" * 64,
            "scripts/stage_s3_corpus_canary.py": "7" * 64,
        },
        "job": {
            "driver": {"cores": 4, "disk_gib": 20, "memory_gib": 14},
            "dynamic_allocation": False,
            "executor": {"cores": 4, "disk_gib": 20, "instances": 3, "memory_gib": 14},
            "python_executable": "/opt/lumi/python312/bin/python",
            "spark_shuffle_partitions": 12,
        },
        "output": {
            "checkpoint_prefix": "lumi-corpus/v0.1/control/checkpoints",
            "result_prefix": "lumi-corpus/v0.1/control/results",
            "write_production_shards": False,
        },
        "runtime": {
            "release_label": "emr-7.13.0",
            "service": "aws_emr_serverless",
            "spark_version": "3.5.6",
        },
        "schema_version": 1,
        "source_commit": "a" * 40,
        "source_transport": {
            "direct_external_http_fallback": False,
            "network_path": "same_region_s3_only",
            "plan_fingerprint": "e" * 64,
            "producer": {
                "compute_type": "BUILD_GENERAL1_SMALL",
                "disk_gb": 64,
                "direct_source_network_required": True,
                "launch_authorized": False,
                "maximum_build_minutes": 30,
                "maximum_parallel_downloads": 2,
                "memory_gib": 4,
                "service": "aws_codebuild_same_image_build",
                "source_files": 8,
                "vcpu": 2,
            },
            "source_bytes": 8_000,
            "source_files": 8,
            "staging_plan_sha256": "f" * 64,
        },
        "sources": _sources(),
    }
    release["release_fingerprint"] = _release_fingerprint(release)
    release["decision"] = {
        "paid_codebuild_image": "not_authorized",
        "paid_emr_serverless_canary": "not_authorized",
        "paid_emr_serverless_production": "not_authorized",
        "status": "verified_local_package_waiting_for_new_owner_approval",
    }
    return release


def _authorization(release: dict[str, object], now: datetime) -> dict[str, object]:
    return {
        "approval_text": EMR_CANARY_APPROVAL_TEXT,
        "approved_at": now.isoformat().replace("+00:00", "Z"),
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_codebuild_image": True,
            "paid_emr_serverless_canary": True,
            "paid_emr_serverless_production": False,
            "paid_gpu_compute": False,
        },
        "bounds": {
            "maximum_codebuild_compute_usd": "0.15",
            "maximum_codebuild_disk_gb": 64,
            "maximum_codebuild_memory_gib": 4,
            "maximum_codebuild_minutes": 30,
            "maximum_codebuild_vcpu": 2,
            "maximum_combined_compute_usd": "0.755904",
            "maximum_emr_compute_usd": "0.605904",
            "maximum_emr_disk_gib": 80,
            "maximum_emr_memory_gib": 64,
            "maximum_emr_runtime_minutes": 30,
            "maximum_emr_vcpu": 16,
            "maximum_records_per_source": 2_000,
            "maximum_source_compressed_bytes": 700_000_000,
            "maximum_source_files": 8,
            "maximum_staging_parallel_downloads": 2,
            "maximum_total_source_compressed_bytes": 1_500_000_000,
        },
        "expires_at": (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "gate": "lumi_emr_serverless_canary_authorization_v0.1",
        "image_context_sha256": release["image"]["context_sha256"],
        "owner": "Afnan Technologies Inc.",
        "release_fingerprint": release["release_fingerprint"],
        "run_token": "1" * 32,
        "schema_version": 1,
        "source_commit": release["source_commit"],
        "source_staging_plan_fingerprint": release["source_transport"][
            "plan_fingerprint"
        ],
        "source_staging_plan_sha256": release["source_transport"][
            "staging_plan_sha256"
        ],
    }


def test_emr_contract_is_local_only_and_cost_is_exact() -> None:
    contract_path = ROOT / "configs" / "distributed-production-corpus-emr-serverless-v0.1.json"
    contract = json.loads(contract_path.read_text())
    paths = _validate_contract(contract, ROOT)
    assert _cost_envelope(contract, paths) == {
        "codebuild_compute_hard_maximum_usd": "0.15",
        "combined_compute_hard_maximum_usd": "0.755904",
        "emr_compute_hard_maximum_usd": "0.605904",
        "memory_gb_hour_usd": "0.0057785",
        "standard_storage_extra_gb_hour_usd": "0.000111",
        "vcpu_hour_usd": "0.052624",
    }
    assert not any(contract["authorization"].values())
    assert contract["canary"]["maximum_total_source_compressed_bytes"] == 1_500_000_000


def test_package_source_commit_must_be_checked_out_head() -> None:
    with pytest.raises(ValueError, match="checked-out Git HEAD"):
        _require_clean_source_commit(ROOT, "0" * 40)


def test_first_codebuild_attempt_failed_safe_and_cleanup_is_complete() -> None:
    report = json.loads(
        (ROOT / "reports" / "emr-codebuild-canary-v0.1.json").read_text()
    )
    assert report["status"] == "failed_safe_before_image_build"
    assert report["approval_consumed"] is True
    assert report["docker_build_started"] is False
    assert report["ecr_images_pushed"] == 0
    assert report["emr_job_started"] is False
    assert report["estimated_compute_usd_upper_bound"] == "0.03"
    assert set(report["cleanup"].values()) <= {
        "absent",
        "absent_404_delete_marker_created",
        0,
    }


def test_emr_image_context_is_digest_and_dependency_hash_pinned() -> None:
    directory = ROOT / "containers" / "emr-serverless-7.13-python312"
    dockerfile = (directory / "Dockerfile").read_text()
    assert dockerfile.startswith(
        "FROM public.ecr.aws/emr-serverless/spark/emr-7.13.0@" + BASE_IMAGE_DIGEST
    )
    assert dockerfile.rstrip().endswith("USER hadoop:hadoop")
    assert "/usr/bin/python3.12 -m venv /opt/lumi/python312" in dockerfile
    assert "--require-hashes" in dockerfile
    assert not any(
        f"ENV {name}" in dockerfile
        for name in ("JAVA_HOME", "SPARK_HOME", "HIVE_HOME", "TEZ_HOME")
    )
    rows = [
        row
        for row in (directory / "requirements-linux-x86_64.lock").read_text().splitlines()
        if row and not row.startswith("#")
    ]
    assert len(rows) == 27
    assert all("==" in row and " --hash=sha256:" in row for row in rows)
    assert any(row.startswith("lingua-language-detector==2.2.0 ") for row in rows)
    buildspec = (directory / "buildspec.yml").read_text()
    assert 'test "$CODEBUILD_SOURCE_VERSION" = "$EXPECTED_SOURCE_VERSION"' in buildspec
    assert "CODEBUILD_RESOLVED_SOURCE_VERSION" not in buildspec
    assert "docker build --pull --no-cache" in buildspec
    assert 'docker push "$REPOSITORY_URI:$IMAGE_TAG"' in buildspec
    assert "stage_s3_corpus_canary.py" in buildspec
    assert buildspec.index("stage_s3_corpus_canary.py") < buildspec.index("docker build")
    staging_rows = [
        row
        for row in (directory / "requirements-staging-linux-x86_64.lock")
        .read_text()
        .splitlines()
        if row and not row.startswith("#")
    ]
    assert len(staging_rows) == 7
    assert all("==" in row and " --hash=sha256:" in row for row in staging_rows)


def test_one_approval_is_bound_to_exact_image_context_and_derived_digest() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    release = _synthetic_release()
    assert validate_emr_serverless_canary_release(release)["image_state"] == (
        "build_not_authorized"
    )
    authorization = _authorization(release, now)
    before = validate_emr_serverless_canary_authorization(authorization, release, now=now)
    assert before["status"] == (
        "authorized_one_bounded_source_staging_image_build_and_canary_only"
    )

    digest = "sha256:" + "d" * 64
    bound = bind_emr_serverless_image(
        release,
        image_uri=f"123456789012.dkr.ecr.us-east-1.amazonaws.com/lumi-emr-serverless@{digest}",
        image_bytes=3_000_000_000,
    )
    assert bound["image"]["source_release_fingerprint"] == release["release_fingerprint"]
    after = validate_emr_serverless_canary_authorization(authorization, bound, now=now)
    assert after["status"] == "authorized_bounded_canary_from_approved_image_context"

    tampered = json.loads(json.dumps(bound))
    tampered["image"]["context_sha256"] = "e" * 64
    material = {
        key: value
        for key, value in tampered.items()
        if key not in {"decision", "release_fingerprint"}
    }
    tampered["release_fingerprint"] = _release_fingerprint(material)
    with pytest.raises(ValueError, match="another image context"):
        validate_emr_serverless_canary_authorization(authorization, tampered, now=now)

    wrong_staging = _authorization(release, now)
    wrong_staging["source_staging_plan_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="another S3 staging plan"):
        validate_emr_serverless_canary_authorization(wrong_staging, release, now=now)


def test_codebuild_requests_are_exact_idempotent_no_retry_and_least_privilege() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    release = _synthetic_release()
    authorization = _authorization(release, now)
    account_id = "123456789012"
    fingerprint = release["release_fingerprint"]
    role = f"arn:aws:iam::{account_id}:role/lumi-codebuild-emr-canary-{fingerprint[:12]}"
    version_id = "fixture+version/1="

    project = build_codebuild_project_request(
        release,
        authorization,
        account_id=account_id,
        context_version_id=version_id,
        service_role_arn=role,
        now=now,
    )
    assert project["autoRetryLimit"] == 0
    assert project["concurrentBuildLimit"] == 1
    assert project["timeoutInMinutes"] == 30
    assert project["sourceVersion"] == version_id
    assert project["environment"]["image"] == CODEBUILD_STANDARD_IMAGE
    assert project["environment"]["computeType"] == "BUILD_GENERAL1_SMALL"
    assert "vpcConfig" not in project
    variables = {
        row["name"]: row["value"] for row in project["environment"]["environmentVariables"]
    }
    assert variables["EXPECTED_SOURCE_VERSION"] == version_id
    assert variables["AUTHORIZED_STAGING_PLAN_FINGERPRINT"] == release[
        "source_transport"
    ]["plan_fingerprint"]
    assert variables["EXPECTED_CODE_ARCHIVE_SHA256"] == "1" * 64

    start = build_codebuild_start_request(
        release,
        authorization,
        account_id=account_id,
        context_version_id=version_id,
        now=now,
    )
    assert start == {
        "autoRetryLimitOverride": 0,
        "idempotencyToken": authorization["run_token"],
        "projectName": project["name"],
        "queuedTimeoutInMinutesOverride": 10,
        "sourceVersion": version_id,
        "timeoutInMinutesOverride": 30,
    }

    policy = build_codebuild_service_policy(release, account_id=account_id)
    stage = next(
        statement
        for statement in policy["Statement"]
        if statement["Sid"] == "StageExactSourceObjects"
    )
    assert stage["Action"] == ["s3:GetObject", "s3:PutObject"]
    assert len(stage["Resource"]) == 8
    assert all(resource.endswith(".json.gz") for resource in stage["Resource"])
    absence = next(
        statement
        for statement in policy["Statement"]
        if statement["Sid"] == "ResolveAbsentSourceObjects"
    )
    assert absence == {
        "Action": "s3:ListBucket",
        "Effect": "Allow",
        "Resource": "arn:aws:s3:::lumi-corpus-fixture",
        "Sid": "ResolveAbsentSourceObjects",
    }
    serialized = json.dumps(policy, sort_keys=True)
    assert "s3:DeleteObject" not in serialized
    assert "emr-serverless:" not in serialized
    assert "ec2:" not in serialized


def test_release_and_authorization_fail_closed_on_boundary_changes() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    release = _synthetic_release()

    wrong_base = json.loads(json.dumps(release))
    wrong_base["image"]["base_uri"] = "public.ecr.aws/example@" + BASE_IMAGE_DIGEST
    wrong_base["release_fingerprint"] = _release_fingerprint(
        {
            key: value
            for key, value in wrong_base.items()
            if key not in {"decision", "release_fingerprint"}
        }
    )
    with pytest.raises(ValueError, match="image boundary"):
        validate_emr_serverless_canary_release(wrong_base)

    wrong_decision = json.loads(json.dumps(release))
    wrong_decision["decision"]["paid_emr_serverless_canary"] = "authorized"
    with pytest.raises(ValueError, match="decision boundary"):
        validate_emr_serverless_canary_release(wrong_decision)

    long_lived = _authorization(release, now)
    long_lived["expires_at"] = (now + timedelta(hours=3)).isoformat().replace(
        "+00:00", "Z"
    )
    with pytest.raises(ValueError, match="lifetime exceeds two hours"):
        validate_emr_serverless_canary_authorization(long_lived, release, now=now)


def test_emr_requests_keep_exact_capacity_runtime_and_no_retry() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    unbound = _synthetic_release()
    authorization = _authorization(unbound, now)
    bound = bind_emr_serverless_image(
        unbound,
        image_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/lumi-emr-serverless@sha256:"
            + "d" * 64
        ),
        image_bytes=3_000_000_000,
    )
    application = build_emr_serverless_application_request(bound)
    assert application["maximumCapacity"] == {
        "cpu": "16 vCPU",
        "disk": "80 GB",
        "memory": "64 GB",
    }
    assert "initialCapacity" not in application
    job = build_emr_serverless_job_request(
        bound,
        authorization,
        application_id="00f00ba4",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-emr-canary",
        release_s3_uri="s3://lumi-corpus-fixture/control/release.json",
        authorization_s3_uri="s3://lumi-corpus-fixture/control/authorization.json",
        now=now,
    )
    assert job["executionTimeoutMinutes"] == 30
    assert job["retryPolicy"] == {"maxAttempts": 1}
    parameters = job["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "spark.executor.instances=3" in parameters
    assert "spark.dynamicAllocation.enabled=false" in parameters
    assert "/opt/lumi/python312/bin/python" in parameters
    assert "awsglue" not in (
        ROOT / "scripts" / "run_emr_serverless_distributed_canary.py"
    ).read_text()
    job_source = (
        ROOT / "scripts" / "run_emr_serverless_distributed_canary.py"
    ).read_text()
    assert "download_staged_source" in job_source
    assert "_download_source" not in job_source
    assert "huggingface.co" not in job_source
