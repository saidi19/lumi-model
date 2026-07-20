from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lumi.emr_production_adaptive import reopen_map_checkpoint
from lumi.emr_production_dedup import (
    build_global_dedup_job_request,
    dedup_job_token,
)
from lumi.emr_production_map import (
    advance_map_checkpoint,
    build_map_job_request,
    build_reserved_map_job_request,
    commit_reserved_map_output,
    initial_map_checkpoint,
    map_job_token,
    reserve_map_categories,
    validate_map_checkpoint,
)
from lumi.emr_serverless_production import (
    PRODUCTION_RUN_APPROVAL_TEXT,
    build_emr_production_plan,
    initial_production_checkpoint,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs/emr-production-resume-v0.1.json"


def _plan() -> dict:
    return build_emr_production_plan(
        CONTRACT,
        repository_root=ROOT,
        source_commit="9" * 40,
    )


def _authorization(plan: dict, checkpoint: dict) -> dict:
    return {
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


def _output(plan: dict, checkpoint: dict, *, tokens: int = 50_000_000) -> dict:
    source = plan["source_schedule"][checkpoint["source_cursor"]["source_ordinal"]]
    value = {
        "candidate_documents": 8_000,
        "candidate_tokens": tokens,
        "category": source["category"],
        "end_line": 50_000,
        "object_bytes": 120_000_000,
        "parent_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parquet_s3_uri": (
            f"s3://fixture/mapped/source-{int(source['ordinal']):05d}-"
            f"{int(checkpoint['checkpoint_sequence']):05d}"
        ),
        "plan_fingerprint": plan["plan_fingerprint"],
        "source_complete": False,
        "source_ordinal": source["ordinal"],
        "source_sha256": source["source_sha256"],
        "start_after_line": checkpoint["source_cursor"]["next_line"],
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return value


def _ready_map_checkpoint(plan: dict, accepted: dict, authorization: dict) -> dict:
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    for _ in range(len(checkpoint["categories"])):
        category = next(
            row
            for row in checkpoint["categories"]
            if row["category"]
            == plan["source_schedule"][checkpoint["source_cursor"]["source_ordinal"]][
                "category"
            ]
        )
        remaining = category["mapped_target_tokens"] - category["mapped_tokens"]
        checkpoint = advance_map_checkpoint(
            checkpoint,
            plan,
            authorization,
            _output(plan, checkpoint, tokens=remaining),
        )
    assert checkpoint["status"] == "ready_for_global_dedup"
    return checkpoint


def _needs_more_result(
    plan: dict,
    authorization: dict,
    checkpoint: dict,
    *,
    science_deficit: int,
) -> dict:
    categories = []
    total_documents = 0
    total_tokens = 0
    for row in plan["categories"]:
        deficit = science_deficit if row["category"] == "science_and_education" else 0
        selected = int(row["floor_tokens"]) - int(row["accepted_tokens"]) - deficit
        documents = 1 if selected else 0
        categories.append(
            {
                "available_documents": documents + 10,
                "available_tokens": selected + 1_000_000_000,
                "category": row["category"],
                "deficit_tokens": deficit,
                "final_documents": int(row["accepted_documents"]) + documents,
                "final_tokens": int(row["accepted_tokens"]) + selected,
                "floor_tokens": int(row["floor_tokens"]),
                "selected_documents": documents,
                "selected_tokens": selected,
            }
        )
        total_documents += documents
        total_tokens += selected
    token = dedup_job_token(authorization, checkpoint)
    return {
        "accepted_candidates": {
            "documents": total_documents,
            "s3_uri": (
                f"s3://{plan['storage']['bucket']}/{plan['storage']['namespace']}/"
                f"deduplicated/{token}/accepted"
            ),
            "tokens": total_tokens,
        },
        "authorization_run_token": authorization["run_token"],
        "categories": categories,
        "gate": "lumi_emr_production_global_dedup_result_v0.1",
        "job_token": token,
        "map_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "schema_version": 1,
        "status": "needs_more_mapping",
        "total_deficit_tokens": science_deficit,
    }


def _reserved_output(plan: dict, reservation: dict, *, tokens: int = 50_000_000) -> dict:
    cursor = reservation["source_cursor"]
    source = plan["source_schedule"][cursor["source_ordinal"]]
    value = {
        "candidate_documents": 8_000 if tokens else 0,
        "candidate_tokens": tokens,
        "category": reservation["category"],
        "end_line": int(cursor["next_line"]) + 10_000,
        "object_bytes": 120_000_000 if tokens else 0,
        "parquet_s3_uri": (
            f"s3://{plan['storage']['bucket']}/{plan['storage']['namespace']}/mapped/"
            f"{reservation['reservation_token']}/candidates"
            if tokens
            else None
        ),
        "plan_fingerprint": plan["plan_fingerprint"],
        "reservation_token": reservation["reservation_token"],
        "source_complete": False,
        "source_ordinal": source["ordinal"],
        "source_sha256": source["source_sha256"],
        "start_after_line": cursor["next_line"],
    }
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return value


def test_map_checkpoint_targets_120_percent_of_each_remaining_floor() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    assert checkpoint["status"] == "mapping"
    assert checkpoint["source_cursor"] == {"next_line": 38_848, "source_ordinal": 0}
    science = checkpoint["categories"][0]
    assert science["mapped_target_tokens"] == 42_914_676_432
    assert sum(row["mapped_target_tokens"] for row in checkpoint["categories"]) == (
        143_714_676_432
    )
    assert validate_map_checkpoint(checkpoint, plan, authorization) == checkpoint
    assert plan["pipeline"] == {
        "global_dedup_job_timeout_hours": "24",
        "global_dedup_job_vcpu": 128,
        "map_chunks_per_job": 8,
        "map_job_timeout_hours": "2",
        "map_job_vcpu": 16,
        "mapped_category_multiplier": "1.20",
        "maximum_concurrent_map_jobs": 7,
        "maximum_concurrent_shard_writers": 7,
        "shard_writer_job_vcpu": 16,
        "shard_writer_job_timeout_hours": "12",
        "strategy": "two_pass_map_global_dedup_shard_v0.1",
    }


def test_map_checkpoint_advances_one_content_addressed_slice() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    output = _output(plan, checkpoint)
    advanced = advance_map_checkpoint(checkpoint, plan, authorization, output)
    assert advanced["checkpoint_sequence"] == 1
    assert advanced["previous_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert advanced["source_cursor"] == {"next_line": 50_000, "source_ordinal": 0}
    assert advanced["total_mapped_documents"] == 8_000
    assert advanced["total_mapped_tokens"] == 50_000_000
    assert advanced["categories"][0]["mapped_tokens"] == 50_000_000
    assert advanced["outputs"] == [output]


def test_map_checkpoint_rejects_manifest_drift_and_category_overshoot() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    changed = _output(plan, checkpoint)
    changed["candidate_tokens"] += 1
    with pytest.raises(ValueError, match="manifest SHA-256"):
        advance_map_checkpoint(checkpoint, plan, authorization, changed)
    target = checkpoint["categories"][0]["mapped_target_tokens"]
    oversized = _output(
        plan,
        checkpoint,
        tokens=target + plan["limits"]["maximum_category_overshoot_tokens"] + 1,
    )
    with pytest.raises(ValueError, match="category overshoot"):
        advance_map_checkpoint(checkpoint, plan, authorization, oversized)


def test_seven_category_lanes_reserve_and_commit_out_of_order() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    reserved = reserve_map_categories(checkpoint, plan, authorization)
    assert len(reserved["active_reservations"]) == 7
    assert {row["category"] for row in reserved["active_reservations"]} == {
        row["category"] for row in plan["categories"]
    }
    assert reserved["reservation_sequence"] == 1
    assert reserved["checkpoint_sequence"] == 1
    second = reserved["active_reservations"][1]
    committed = commit_reserved_map_output(
        reserved,
        plan,
        authorization,
        _reserved_output(plan, second),
    )
    assert len(committed["active_reservations"]) == 6
    assert committed["output_sequence"] == 1
    assert committed["checkpoint_sequence"] == 2
    assert committed["categories"][1]["mapped_tokens"] == 50_000_000
    refilled = reserve_map_categories(committed, plan, authorization)
    assert len(refilled["active_reservations"]) == 7
    code = next(
        row for row in refilled["active_reservations"] if row["category"] == "code"
    )
    assert code["source_cursor"]["next_line"] == 10_000
    assert (
        commit_reserved_map_output(
            committed,
            plan,
            authorization,
            _reserved_output(plan, second),
        )
        == committed
    )


def test_reserved_map_request_is_self_staging_and_bounded_to_16_vcpu() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = reserve_map_categories(
        initial_map_checkpoint(plan, accepted, authorization), plan, authorization
    )
    reservation = checkpoint["active_reservations"][0]
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/{plan['storage']['namespace']}/control/production"
    request = build_reserved_map_job_request(
        plan,
        checkpoint,
        authorization,
        reservation_token=reservation["reservation_token"],
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        map_checkpoint_s3_uri=f"{prefix}/map.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        staged_source_s3_uri=f"{prefix}/staged.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/reserved-map.py",
    )
    assert request["executionTimeoutMinutes"] == 120
    assert request["clientToken"].endswith(reservation["reservation_token"])
    parameters = request["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "spark.executor.instances=3" in parameters
    arguments = request["jobDriver"]["sparkSubmit"]["entryPointArguments"]
    assert "--lumi-staged-source-file" in arguments
    assert request["tags"]["LumiCategory"] == "science_and_education"


def test_reserved_map_can_advance_past_a_source_with_zero_candidates() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = reserve_map_categories(
        initial_map_checkpoint(plan, accepted, authorization), plan, authorization
    )
    science = checkpoint["active_reservations"][0]
    output = _reserved_output(plan, science, tokens=0)
    output["source_complete"] = True
    material = dict(output)
    material.pop("manifest_sha256")
    output["manifest_sha256"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    advanced = commit_reserved_map_output(
        checkpoint, plan, authorization, output
    )
    assert advanced["total_mapped_tokens"] == 0
    assert advanced["categories"][0]["mapped_tokens"] == 0
    assert advanced["active_reservations"][0]["category"] != (
        "science_and_education"
    )


def test_map_job_request_is_one_of_seven_16_vcpu_slots() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = initial_map_checkpoint(plan, accepted, authorization)
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/distributed/production"
    request = build_map_job_request(
        plan,
        checkpoint,
        authorization,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        map_checkpoint_s3_uri=f"{prefix}/map-checkpoint.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        staged_source_s3_uri=f"{prefix}/staged.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/run-map.py",
    )
    token = map_job_token(authorization, checkpoint)
    assert request["clientToken"] == f"lumi-production-map-{token}"
    assert request["executionTimeoutMinutes"] == 120
    parameters = request["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "spark.executor.instances=3" in parameters
    assert request["tags"]["LumiScope"] == "bounded-production-corpus-map-only"


def test_global_dedup_request_consumes_only_a_complete_map_checkpoint() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = _ready_map_checkpoint(plan, accepted, authorization)
    bucket = plan["storage"]["bucket"]
    prefix = f"s3://{bucket}/lumi-corpus/v0.1/control/distributed/production"
    request = build_global_dedup_job_request(
        plan,
        checkpoint,
        authorization,
        application_id="00fixture",
        execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
        plan_s3_uri=f"{prefix}/plan.json",
        map_checkpoint_s3_uri=f"{prefix}/map-checkpoint.json",
        authorization_s3_uri=f"{prefix}/authorization.json",
        code_archive_s3_uri=f"{prefix}/code.zip",
        job_script_s3_uri=f"{prefix}/run-dedup.py",
    )
    token = dedup_job_token(authorization, checkpoint)
    assert request["clientToken"] == f"lumi-production-dedup-{token}"
    assert request["executionTimeoutMinutes"] == 1440
    parameters = request["jobDriver"]["sparkSubmit"]["sparkSubmitParameters"]
    assert "spark.executor.instances=31" in parameters
    incomplete = initial_map_checkpoint(plan, accepted, authorization)
    with pytest.raises(ValueError, match="not ready"):
        build_global_dedup_job_request(
            plan,
            incomplete,
            authorization,
            application_id="00fixture",
            execution_role_arn="arn:aws:iam::123456789012:role/lumi-production",
            plan_s3_uri=f"{prefix}/plan.json",
            map_checkpoint_s3_uri=f"{prefix}/map-checkpoint.json",
            authorization_s3_uri=f"{prefix}/authorization.json",
            code_archive_s3_uri=f"{prefix}/code.zip",
            job_script_s3_uri=f"{prefix}/run-dedup.py",
        )


def test_adaptive_shortfall_reopens_exact_category_cursor_once() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = _ready_map_checkpoint(plan, accepted, authorization)
    dedup = _needs_more_result(
        plan, authorization, checkpoint, science_deficit=500_000_000
    )
    reopened = reopen_map_checkpoint(checkpoint, plan, authorization, dedup)
    assert reopened["status"] == "mapping"
    assert reopened["checkpoint_sequence"] == checkpoint["checkpoint_sequence"] + 1
    assert reopened["output_sequence"] == checkpoint["output_sequence"]
    assert reopened["source_cursor"] == {"next_line": 50_000, "source_ordinal": 0}
    science_before = checkpoint["categories"][0]
    science_after = reopened["categories"][0]
    assert science_after["mapped_target_tokens"] == (
        science_before["mapped_target_tokens"] + 600_000_000
    )
    assert reopened["adaptive_rounds"][0]["total_target_increase_tokens"] == (
        600_000_000
    )
    output = _output(plan, reopened, tokens=600_000_000)
    output["end_line"] = 100_000
    material = dict(output)
    material.pop("manifest_sha256")
    output["manifest_sha256"] = hashlib.sha256(
        json.dumps(material, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    advanced = advance_map_checkpoint(reopened, plan, authorization, output)
    assert advanced["status"] == "ready_for_global_dedup"
    assert advanced["source_cursor"]["source_ordinal"] == len(
        plan["source_schedule"]
    )


def test_adaptive_shortfall_fails_before_the_150b_raw_cap() -> None:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = _ready_map_checkpoint(plan, accepted, authorization)
    dedup = _needs_more_result(
        plan, authorization, checkpoint, science_deficit=10_000_000_000
    )
    with pytest.raises(RuntimeError, match="150B raw-token cap"):
        reopen_map_checkpoint(checkpoint, plan, authorization, dedup)
