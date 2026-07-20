from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lumi.emr_serverless_production import (
    PRODUCTION_APPROVAL_TEXT,
    PRODUCTION_RUN_APPROVAL_TEXT,
    advance_production_checkpoint,
    build_application_request,
    build_emr_production_plan,
    build_ramp_job_request,
    checkpoint_fingerprint,
    initial_production_checkpoint,
    next_source_cursor,
    production_floors_complete,
    source_job_token,
    staging_maximum_chunks,
    validate_paid_production_authorization,
    validate_paid_ramp_authorization,
    validate_production_checkpoint,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/emr-production-resume-v0.1.json"
SOURCE_COMMIT = "9" * 40


def _plan() -> dict[str, object]:
    return build_emr_production_plan(
        CONTRACT,
        repository_root=ROOT,
        source_commit=SOURCE_COMMIT,
    )


def _fingerprint(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _commit(
    plan: dict[str, object],
    checkpoint: dict[str, object],
    *,
    accepted_tokens: int = 50_000_000,
    accepted_documents: int = 8_000,
    next_line: int = 50_000,
) -> dict[str, object]:
    old_partial = checkpoint["partial_shard"]
    combined_tokens = old_partial["tokens"] + accepted_tokens
    combined_documents = old_partial["documents"] + accepted_documents
    if combined_tokens >= plan["limits"]["clean_shard_target_tokens"]:
        sealed_shards = [
            {
                "category": "science_and_education",
                "documents": combined_documents,
                "manifest_sha256": "a" * 64,
                "tokens": combined_tokens,
            }
        ]
        partial_shard = None
        next_shard_sequence = 3
    else:
        sealed_shards = []
        partial_shard = {
            "category": "science_and_education",
            "clean_sha256": "c" * 64,
            "clean_s3_key": "lumi/partial-clean",
            "clean_version_id": "clean-version",
            "documents": combined_documents,
            "sequence": 2,
            "state": "s3_snapshot",
            "text_bytes": old_partial["text_bytes"] + 30_000_000,
            "token_bytes": combined_tokens * 2,
            "token_sha256": "d" * 64,
            "token_s3_key": "lumi/partial-tokens",
            "token_version_id": "token-version",
            "tokens": combined_tokens,
        }
        next_shard_sequence = 2
    value: dict[str, object] = {
        "anchor_segment": {
            "documents": accepted_documents,
            "manifest_sha256": "e" * 64,
        },
        "category_deltas": {
            "science_and_education": {
                "accepted_documents": accepted_documents,
                "accepted_tokens": accepted_tokens,
            }
        },
        "logical_storage_bytes": 200_000_000,
        "next_shard_sequence": next_shard_sequence,
        "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "partial_shard": partial_shard,
        "plan_fingerprint": plan["plan_fingerprint"],
        "sealed_shards": sealed_shards,
        "source_cursor": {"next_line": next_line, "source_ordinal": 0},
    }
    value["commit_sha256"] = _fingerprint(value)
    return value


def test_plan_binds_verified_prefix_and_exact_roadmap_limits() -> None:
    plan = _plan()
    assert plan["limits"]["candidate_total_floor_tokens"] == 120_000_000_000
    assert plan["limits"]["candidate_hard_maximum_tokens"] == 150_000_000_000
    assert plan["limits"]["hard_logical_storage_bytes"] == 2_000_000_000_000
    assert plan["limits"]["clean_shard_target_tokens"] == 100_000_000
    assert plan["resume"]["accepted_tokens"] == 237_769_640
    assert plan["resume"]["source_next_line"] == 38_848
    assert plan["source_inventory"] == {
        "compressed_source_bytes": 499_817_285_384,
        "data_files": 758,
    }
    assert plan["storage"]["bucket"] == "lumi-corpus-prod-27777c02b29b76efcac3"
    assert plan["storage"]["namespace"] == "lumi-corpus/v0.1"
    assert set(plan["storage"]["artifacts"]) == {
        "benchmark_registry",
        "frozen_tokenizer",
        "initial_anchors",
        "training_exclusion_ledger",
    }
    assert plan["source_schedule"][0]["source_provenance"]
    assert plan["decision"]["paid_emr_production"] == "not_authorized"
    assert plan["decision"]["full_4b_training"] == "not_authorized"
    assert plan["budget"] == {
        "contingency_usd": "250.00",
        "hard_total_usd": "2000.00",
        "maximum_archive_migration_usd": "250.00",
        "maximum_emr_compute_usd": "1200.00",
        "maximum_s3_logs_requests_usd": "100.00",
        "maximum_stager_compute_usd": "200.00",
    }


def test_initial_checkpoint_is_hash_bound_to_verified_prefix() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    assert checkpoint["checkpoint_sequence"] == 0
    assert checkpoint["total_accepted_documents"] == 38_349
    assert checkpoint["total_accepted_tokens"] == 237_769_640
    assert checkpoint["source_cursor"] == {"next_line": 38_848, "source_ordinal": 0}
    assert checkpoint["checkpoint_sha256"] == checkpoint_fingerprint(checkpoint)
    assert validate_production_checkpoint(checkpoint, plan) == checkpoint
    assert production_floors_complete(checkpoint, plan) is False


def test_checkpoint_advance_is_monotonic_and_conserves_categories() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    advanced = advance_production_checkpoint(checkpoint, plan, _commit(plan, checkpoint))
    assert advanced["checkpoint_sequence"] == 1
    assert advanced["previous_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert advanced["total_accepted_tokens"] == 287_769_640
    assert advanced["total_accepted_documents"] == 46_349
    assert advanced["source_cursor"] == {"next_line": 50_000, "source_ordinal": 0}
    assert advanced["logical_storage_bytes"] == 882_985_279
    assert len(advanced["sealed_shards"]) == len(checkpoint["sealed_shards"])
    assert advanced["partial_shard"]["tokens"] == 87_154_932
    assert advanced["anchor_segments"][0]["documents"] == 8_000


def test_checkpoint_rejects_cursor_regression_and_wrong_parent() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    regression = _commit(plan, checkpoint, next_line=38_848)
    with pytest.raises(ValueError, match="cursor did not advance"):
        advance_production_checkpoint(checkpoint, plan, regression)
    wrong_parent = _commit(plan, checkpoint)
    wrong_parent["parent_checkpoint_sha256"] = "b" * 64
    wrong_parent["commit_sha256"] = _fingerprint(
        {key: value for key, value in wrong_parent.items() if key != "commit_sha256"}
    )
    with pytest.raises(ValueError, match="parent checkpoint"):
        advance_production_checkpoint(checkpoint, plan, wrong_parent)


def test_checkpoint_rejects_150b_and_2tb_cap_breaches() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    token_breach = deepcopy(checkpoint)
    token_breach["total_accepted_tokens"] = 150_000_000_001
    token_breach["categories"][0]["accepted_tokens"] += (
        token_breach["total_accepted_tokens"] - checkpoint["total_accepted_tokens"]
    )
    token_breach["checkpoint_sha256"] = checkpoint_fingerprint(token_breach)
    with pytest.raises(ValueError, match="150B"):
        validate_production_checkpoint(token_breach, plan)
    storage_breach = deepcopy(checkpoint)
    storage_breach["logical_storage_bytes"] = 2_000_000_000_001
    storage_breach["checkpoint_sha256"] = checkpoint_fingerprint(storage_breach)
    with pytest.raises(ValueError, match="2TB"):
        validate_production_checkpoint(storage_breach, plan)


def test_checkpoint_rejects_category_overshoot_beyond_one_shard() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    breach = deepcopy(checkpoint)
    science = breach["categories"][0]
    delta = science["floor_tokens"] + 100_000_001 - science["accepted_tokens"]
    science["accepted_tokens"] += delta
    breach["total_accepted_tokens"] += delta
    breach["checkpoint_sha256"] = checkpoint_fingerprint(breach)
    with pytest.raises(ValueError, match="overshoot"):
        validate_production_checkpoint(breach, plan)


def test_source_cursor_resumes_slices_and_skips_a_completed_category() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    partial_manifest = {
        "chunks": [{"end_line": 50_000}],
        "source_complete": False,
        "start_after_line": 38_848,
    }
    assert next_source_cursor(
        plan,
        checkpoint,
        partial_manifest,
        category_floor_reached=False,
    ) == {"next_line": 50_000, "source_ordinal": 0}
    complete_manifest = {
        "chunks": [{"end_line": 75_000}],
        "source_complete": True,
        "start_after_line": 38_848,
    }
    assert next_source_cursor(
        plan,
        checkpoint,
        complete_manifest,
        category_floor_reached=False,
    ) == {"next_line": 0, "source_ordinal": 1}
    skipped = next_source_cursor(
        plan,
        checkpoint,
        partial_manifest,
        category_floor_reached=True,
    )
    assert skipped["next_line"] == 0
    assert plan["source_schedule"][skipped["source_ordinal"]]["category"] != (
        "science_and_education"
    )


def test_application_requests_are_exactly_16_64_or_128_vcpu() -> None:
    plan = _plan()
    bootstrap = build_application_request(plan, maximum_vcpu=16)
    ramp = build_application_request(plan, maximum_vcpu=64)
    production = build_application_request(plan, maximum_vcpu=128)
    assert bootstrap["maximumCapacity"] == {
        "cpu": "16 vCPU",
        "disk": "80 GB",
        "memory": "56 GB",
    }
    assert ramp["maximumCapacity"]["cpu"] == "64 vCPU"
    assert production["maximumCapacity"]["cpu"] == "128 vCPU"
    assert plan["shapes"]["bootstrap"]["executor_instances"] == 3
    assert plan["shapes"]["ramp"]["executor_instances"] == 15
    assert plan["shapes"]["production"]["executor_instances"] == 31
    with pytest.raises(ValueError, match="pinned bootstrap, ramp, or production"):
        build_application_request(plan, maximum_vcpu=32)
    image = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/lumi@sha256:" + "f" * 64
    )
    bound = build_application_request(plan, maximum_vcpu=64, image_uri=image)
    assert bound["imageConfiguration"] == {"imageUri": image}


def test_paid_ramp_requires_exact_separate_authorization() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    authorization = {
        "approval_text": PRODUCTION_APPROVAL_TEXT,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_cpu_stager": True,
            "paid_emr_production": True,
            "paid_gpu_compute": False,
            "production_adaptive_acquisition": True,
        },
        "bounds": {
            "maximum_compute_usd": "25.00",
            "maximum_emr_vcpu": 64,
            "maximum_runtime_hours": "2",
            "maximum_source_files": 1,
        },
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "gate": "lumi_emr_production_ramp_authorization_v0.1",
        "owner": "Afnan Technologies Inc.",
        "plan_fingerprint": plan["plan_fingerprint"],
        "run_token": "1" * 32,
        "schema_version": 1,
    }
    assert validate_paid_ramp_authorization(authorization, plan, checkpoint) == authorization
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/distributed/production"
    request = build_ramp_job_request(
        plan,
        checkpoint,
        authorization,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        checkpoint_s3_uri=f"{prefix}/checkpoint.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        staged_source_s3_uri=f"{prefix}/staged.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/run.py",
    )
    assert request["executionTimeoutMinutes"] == 120
    assert "spark.executor.instances=15" in request["jobDriver"]["sparkSubmit"][
        "sparkSubmitParameters"
    ]
    assert request["retryPolicy"] == {"maxAttempts": 1}
    assert request["clientToken"].endswith(source_job_token(authorization, checkpoint))
    arguments = request["jobDriver"]["sparkSubmit"]["entryPointArguments"]
    assert arguments[arguments.index("--lumi-job-token") + 1] == source_job_token(
        authorization, checkpoint
    )
    bootstrap_authorization = deepcopy(authorization)
    bootstrap_authorization["bounds"]["maximum_compute_usd"] = "5.00"
    bootstrap_authorization["bounds"]["maximum_emr_vcpu"] = 16
    bootstrap_authorization["run_token"] = "2" * 32
    assert (
        validate_paid_ramp_authorization(
            bootstrap_authorization, plan, checkpoint
        )
        == bootstrap_authorization
    )
    bootstrap_request = build_ramp_job_request(
        plan,
        checkpoint,
        bootstrap_authorization,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        checkpoint_s3_uri=f"{prefix}/checkpoint.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        staged_source_s3_uri=f"{prefix}/staged.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/run.py",
    )
    assert "spark.executor.instances=3" in bootstrap_request["jobDriver"][
        "sparkSubmit"
    ]["sparkSubmitParameters"]
    authorization["authorization"]["full_4b_training"] = True
    with pytest.raises(ValueError, match="boundary"):
        validate_paid_ramp_authorization(authorization, plan, checkpoint)


def test_future_production_authorization_is_128_vcpu_and_2000_usd_bounded() -> None:
    plan = _plan()
    checkpoint = initial_production_checkpoint(plan)
    authorization = {
        "approval_text": PRODUCTION_RUN_APPROVAL_TEXT,
        "authorization": {
            "full_4b_training": False,
            "model_training": False,
            "paid_cpu_stager": True,
            "paid_emr_production": True,
            "paid_gpu_compute": False,
            "production_adaptive_acquisition": True,
        },
        "bounds": {
            "maximum_compute_usd": "1200.00",
            "maximum_cumulative_emr_hours": "96",
            "maximum_emr_vcpu": 128,
            "maximum_job_runtime_hours": "24",
            "maximum_source_files": len(plan["source_schedule"]),
            "maximum_total_usd": "2000.00",
        },
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "gate": "lumi_emr_production_authorization_v0.1",
        "owner": "Afnan Technologies Inc.",
        "plan_fingerprint": plan["plan_fingerprint"],
        "run_token": "3" * 32,
        "schema_version": 1,
    }
    assert (
        validate_paid_production_authorization(authorization, plan, checkpoint)
        == authorization
    )
    assert staging_maximum_chunks(authorization, plan) is None
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/distributed/production"
    request = build_ramp_job_request(
        plan,
        checkpoint,
        authorization,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        checkpoint_s3_uri=f"{prefix}/checkpoint.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        staged_source_s3_uri=f"{prefix}/staged.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/run.py",
    )
    assert request["executionTimeoutMinutes"] == 1440
    assert "spark.executor.instances=31" in request["jobDriver"]["sparkSubmit"][
        "sparkSubmitParameters"
    ]
    first_job_token = source_job_token(authorization, checkpoint)
    assert len(first_job_token) == 32
    over_budget = deepcopy(authorization)
    over_budget["bounds"]["maximum_total_usd"] = "2000.01"
    with pytest.raises(ValueError, match="hard cap"):
        validate_paid_production_authorization(over_budget, plan, checkpoint)
    wrong_capacity = deepcopy(authorization)
    wrong_capacity["bounds"]["maximum_emr_vcpu"] = 64
    with pytest.raises(ValueError, match="128-vCPU"):
        validate_paid_production_authorization(wrong_capacity, plan, checkpoint)

    first_commit = _commit(plan, checkpoint)
    first_commit["run_binding"] = {
        "gate": "lumi_emr_production_run_binding_v0.1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "root_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "run_token": authorization["run_token"],
    }
    first_commit["commit_sha256"] = _fingerprint(
        {key: value for key, value in first_commit.items() if key != "commit_sha256"}
    )
    descendant = advance_production_checkpoint(checkpoint, plan, first_commit)
    assert descendant["run_binding"] == first_commit["run_binding"]
    assert (
        validate_paid_production_authorization(authorization, plan, descendant)
        == authorization
    )
    assert source_job_token(authorization, descendant) != first_job_token
    foreign = deepcopy(descendant)
    foreign["run_binding"]["run_token"] = "4" * 32
    foreign["checkpoint_sha256"] = checkpoint_fingerprint(foreign)
    with pytest.raises(ValueError, match="descendant"):
        validate_paid_production_authorization(authorization, plan, foreign)
