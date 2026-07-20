from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.emr_production_map import initial_map_checkpoint, reserve_map_categories
from lumi.emr_serverless_production import (
    PRODUCTION_RUN_APPROVAL_TEXT,
    build_emr_production_plan,
    initial_production_checkpoint,
)
from lumi.emr_two_pass_controller import (
    bind_controller_artifacts,
    initial_two_pass_controller_state,
    record_job_terminal,
    record_jobs_started,
    transition_controller_phase,
    validate_two_pass_controller_state,
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


def _fixtures() -> tuple[dict, dict, dict, dict]:
    plan = _plan()
    accepted = initial_production_checkpoint(plan)
    authorization = _authorization(plan, accepted)
    checkpoint = reserve_map_categories(
        initial_map_checkpoint(plan, accepted, authorization), plan, authorization
    )
    state = initial_two_pass_controller_state(plan, checkpoint, authorization)
    return plan, authorization, checkpoint, state


def _map_jobs(plan: dict, checkpoint: dict) -> list[dict]:
    namespace = plan["storage"]["namespace"]
    return [
        {
            "application_id": "00fixture",
            "job_run_id": f"job-{index}",
            "job_token": reservation["reservation_token"],
            "kind": "map",
            "metadata": {
                "category": reservation["category"],
                "source_ordinal": reservation["source_cursor"]["source_ordinal"],
            },
            "result_key": (
                f"{namespace}/control/distributed/production/map-runs/"
                f"{reservation['reservation_token']}/result.json"
            ),
            "timeout_minutes": 120,
        }
        for index, reservation in enumerate(checkpoint["active_reservations"])
    ]


def test_controller_reserves_seven_jobs_and_prices_actual_billed_usage() -> None:
    plan, authorization, checkpoint, state = _fixtures()
    jobs = _map_jobs(plan, checkpoint)
    running = record_jobs_started(state, plan, authorization, jobs)
    assert running["status"] == "running"
    assert sum(row["vcpu"] for row in running["active_jobs"]) == 112
    assert validate_two_pass_controller_state(running, plan, authorization) == running
    for job in jobs:
        running = record_job_terminal(
            running,
            plan,
            authorization,
            job_token=job["job_token"],
            terminal_state="SUCCESS",
            billed={"memoryGBHour": 5.6, "storageGBHour": 1, "vCPUHour": 1.6},
            runtime_seconds=360,
        )
    assert running["status"] == "ready"
    assert running["cumulative"]["job_count"] == 7
    assert running["cumulative"]["equivalent_128_vcpu_hours"] == "0.087500"
    transitioned = transition_controller_phase(
        running, plan, authorization, "global_dedup"
    )
    assert transitioned["phase"] == "global_dedup"


def test_controller_rejects_eighth_map_and_cross_phase_overcommit() -> None:
    plan, authorization, checkpoint, state = _fixtures()
    jobs = _map_jobs(plan, checkpoint)
    running = record_jobs_started(state, plan, authorization, jobs)
    eighth = json.loads(json.dumps(jobs[0]))
    eighth["job_token"] = "f" * 32
    eighth["job_run_id"] = "job-eight"
    with pytest.raises(ValueError, match="map concurrency"):
        record_jobs_started(running, plan, authorization, [eighth])


def test_controller_failure_is_sticky_and_artifacts_are_hash_bound() -> None:
    plan, authorization, checkpoint, state = _fixtures()
    jobs = _map_jobs(plan, checkpoint)
    running = record_jobs_started(state, plan, authorization, jobs[:2])
    failed = record_job_terminal(
        running,
        plan,
        authorization,
        job_token=jobs[0]["job_token"],
        terminal_state="FAILED",
        billed={"memoryGBHour": 1, "storageGBHour": 0, "vCPUHour": 0.2},
        runtime_seconds=60,
    )
    assert failed["status"] == "failed"
    failed = record_job_terminal(
        failed,
        plan,
        authorization,
        job_token=jobs[1]["job_token"],
        terminal_state="SUCCESS",
        billed={"memoryGBHour": 1, "storageGBHour": 0, "vCPUHour": 0.2},
        runtime_seconds=60,
    )
    assert failed["status"] == "failed"
    bound = bind_controller_artifacts(
        failed,
        plan,
        authorization,
        map_checkpoint_key="lumi-corpus/v0.1/control/map.json",
        map_checkpoint_sha256=checkpoint["checkpoint_sha256"],
    )
    assert bound["artifacts"]["map_checkpoint_sha256"] == checkpoint[
        "checkpoint_sha256"
    ]
