"""Durable job and budget ledger for Lumi's two-pass EMR production run."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from .emr_production_map import validate_map_checkpoint
from .emr_serverless_production import validate_paid_production_authorization

SCHEMA_VERSION = 1
ZERO = Decimal("0")
MONEY_QUANTUM = Decimal("0.000001")
HOUR_QUANTUM = Decimal("0.000001")
KINDS = {"global_dedup", "map", "shard_writer"}
PHASES = {"complete", "global_dedup", "mapping", "sealing", "sharding"}
TERMINAL_STATES = {"CANCELLED", "FAILED", "SUCCESS"}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"{label} must be a decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be a finite non-negative decimal")
    return result


def billed_job_cost(
    billed: Mapping[str, Any], pricing: Mapping[str, Any]
) -> dict[str, str]:
    """Price one terminal EMR job from AWS billed-resource utilization."""

    usage = {
        "memory_gb_hours": _decimal(billed.get("memoryGBHour", 0), "memory GB-hours"),
        "storage_gb_hours": _decimal(
            billed.get("storageGBHour", 0), "storage GB-hours"
        ),
        "vcpu_hours": _decimal(billed.get("vCPUHour", 0), "vCPU-hours"),
    }
    cost = (
        usage["vcpu_hours"] * _decimal(pricing["vcpu_hour_usd"], "vCPU price")
        + usage["memory_gb_hours"]
        * _decimal(pricing["memory_gb_hour_usd"], "memory price")
        + usage["storage_gb_hours"]
        * _decimal(pricing["standard_storage_gb_hour_usd"], "storage price")
    )
    return {
        **{name: str(value.quantize(HOUR_QUANTUM)) for name, value in usage.items()},
        "compute_usd": str(cost.quantize(MONEY_QUANTUM)),
    }


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError(f"{label} must be a lowercase 32-character token")
    return value


def _kind_vcpu(kind: str, plan: Mapping[str, Any]) -> int:
    if kind == "global_dedup":
        return int(plan["pipeline"]["global_dedup_job_vcpu"])
    if kind == "map":
        return int(plan["pipeline"]["map_job_vcpu"])
    if kind == "shard_writer":
        return int(plan["pipeline"]["shard_writer_job_vcpu"])
    raise ValueError("unsupported two-pass EMR job kind")


def _kind_timeout_minutes(kind: str, plan: Mapping[str, Any]) -> int:
    key = {
        "global_dedup": "global_dedup_job_timeout_hours",
        "map": "map_job_timeout_hours",
        "shard_writer": "shard_writer_job_timeout_hours",
    }[kind]
    return int(Decimal(str(plan["pipeline"][key])) * Decimal(60))


def _kind_hourly_cost(kind: str, plan: Mapping[str, Any]) -> Decimal:
    key = "production_usd" if kind == "global_dedup" else "bootstrap_usd"
    return _decimal(plan["cost"][key], f"{kind} hourly cost")


def controller_state_fingerprint(state: Mapping[str, Any]) -> str:
    material = dict(state)
    material.pop("state_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def initial_two_pass_controller_state(
    plan: Mapping[str, Any],
    map_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    validate_paid_production_authorization(
        authorization, plan, _accepted_root_placeholder(plan, authorization)
    )
    validate_map_checkpoint(map_checkpoint, plan, authorization)
    state = {
        "active_jobs": [],
        "artifacts": {
            "dedup_result_key": None,
            "dedup_result_sha256": None,
            "final_seal_key": None,
            "final_seal_sha256": None,
            "map_checkpoint_key": None,
            "map_checkpoint_sha256": map_checkpoint["checkpoint_sha256"],
            "shard_result_keys": {},
        },
        "authorization_root_checkpoint_sha256": authorization["checkpoint_sha256"],
        "authorization_run_token": authorization["run_token"],
        "completed_jobs": [],
        "cumulative": {
            "compute_usd": "0.000000",
            "equivalent_128_vcpu_hours": "0.000000",
            "job_count": 0,
            "memory_gb_hours": "0.000000",
            "storage_gb_hours": "0.000000",
            "vcpu_hours": "0.000000",
        },
        "gate": "lumi_emr_two_pass_controller_state_v0.1",
        "phase": "mapping",
        "plan_fingerprint": plan["plan_fingerprint"],
        "previous_state_sha256": None,
        "schema_version": SCHEMA_VERSION,
        "state_sequence": 0,
        "status": "ready",
        "stop_reason": None,
    }
    state["state_sha256"] = controller_state_fingerprint(state)
    return validate_two_pass_controller_state(state, plan, authorization)


def _accepted_root_placeholder(
    plan: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the pinned accepted root solely for authorization validation."""

    from .emr_serverless_production import initial_production_checkpoint

    root = initial_production_checkpoint(plan)
    if root["checkpoint_sha256"] != authorization.get("checkpoint_sha256"):
        raise ValueError("two-pass authorization root checkpoint changed")
    return root


def _next_state(current: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    result = {**current, **changes}
    result["previous_state_sha256"] = current["state_sha256"]
    result["state_sequence"] = int(current["state_sequence"]) + 1
    result.pop("state_sha256", None)
    result["state_sha256"] = controller_state_fingerprint(result)
    return result


def _validate_job(job: Mapping[str, Any], plan: Mapping[str, Any], *, active: bool) -> None:
    token = _token(job.get("job_token"), "controller job token")
    kind = job.get("kind")
    if kind not in KINDS:
        raise ValueError("two-pass controller job kind changed")
    if job.get("vcpu") != _kind_vcpu(str(kind), plan):
        raise ValueError("two-pass controller job vCPU changed")
    timeout = job.get("timeout_minutes")
    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or timeout <= 0
        or timeout > _kind_timeout_minutes(str(kind), plan)
    ):
        raise ValueError("two-pass controller job timeout changed")
    for name in ("application_id", "job_run_id"):
        if not isinstance(job.get(name), str) or not job[name]:
            raise ValueError(f"two-pass controller job is missing {name}")
    if not isinstance(job.get("result_key"), str) or not job["result_key"].startswith(
        f"{plan['storage']['namespace']}/"
    ):
        raise ValueError("two-pass controller result key left the namespace")
    metadata = job.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("two-pass controller job metadata is missing")
    if active:
        if "terminal_state" in job:
            raise ValueError("active two-pass job is already terminal")
    else:
        if job.get("terminal_state") not in TERMINAL_STATES:
            raise ValueError("completed two-pass job has an invalid terminal state")
        _decimal(job.get("compute_usd"), "completed job compute cost")
        _decimal(job.get("memory_gb_hours"), "completed job memory usage")
        _decimal(job.get("storage_gb_hours"), "completed job storage usage")
        vcpu_hours = _decimal(job.get("vcpu_hours"), "completed job vCPU usage")
        if _decimal(
            job.get("equivalent_128_vcpu_hours"), "completed equivalent hours"
        ) != (vcpu_hours / Decimal(128)).quantize(HOUR_QUANTUM):
            raise ValueError("completed two-pass equivalent-hour accounting changed")
        runtime = job.get("runtime_seconds")
        if not isinstance(runtime, int) or isinstance(runtime, bool) or runtime <= 0:
            raise ValueError("completed two-pass runtime is invalid")
        if runtime > int(timeout) * 60 + 300:
            raise ValueError("completed two-pass job exceeded its timeout envelope")
    _token(token, "controller job token")


def validate_two_pass_controller_state(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported two-pass controller schema")
    if state.get("gate") != "lumi_emr_two_pass_controller_state_v0.1":
        raise ValueError("unexpected two-pass controller gate")
    if state.get("state_sha256") != controller_state_fingerprint(state):
        raise ValueError("two-pass controller state SHA-256 changed")
    if state.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("two-pass controller targets another plan")
    if state.get("authorization_run_token") != authorization["run_token"]:
        raise ValueError("two-pass controller authorization changed")
    if state.get("authorization_root_checkpoint_sha256") != authorization[
        "checkpoint_sha256"
    ]:
        raise ValueError("two-pass controller authorization root changed")
    sequence = state.get("state_sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("two-pass controller sequence is invalid")
    previous = state.get("previous_state_sha256")
    if sequence == 0 and previous is not None:
        raise ValueError("initial two-pass controller state cannot have a parent")
    if sequence > 0:
        _sha256(previous, "previous controller state")
    if state.get("phase") not in PHASES:
        raise ValueError("two-pass controller phase is invalid")
    if state.get("status") not in {"complete", "failed", "paused", "ready", "running"}:
        raise ValueError("two-pass controller status is invalid")
    stop_reason = state.get("stop_reason")
    if state["status"] in {"failed", "paused"}:
        if not isinstance(stop_reason, str) or not stop_reason:
            raise ValueError("stopped two-pass controller has no reason")
    elif stop_reason is not None:
        raise ValueError("active two-pass controller unexpectedly has a stop reason")
    active_jobs = state.get("active_jobs")
    completed_jobs = state.get("completed_jobs")
    if not isinstance(active_jobs, list) or not isinstance(completed_jobs, list):
        raise ValueError("two-pass controller job ledgers are missing")
    seen: set[str] = set()
    active_vcpu = 0
    active_reserved_cost = ZERO
    active_reserved_hours = ZERO
    active_kind_counts = {kind: 0 for kind in KINDS}
    for job in active_jobs:
        if not isinstance(job, Mapping):
            raise ValueError("active two-pass controller job is invalid")
        _validate_job(job, plan, active=True)
        token = str(job["job_token"])
        if token in seen:
            raise ValueError("two-pass controller repeats a job token")
        seen.add(token)
        kind = str(job["kind"])
        active_kind_counts[kind] += 1
        active_vcpu += int(job["vcpu"])
        timeout_hours = Decimal(int(job["timeout_minutes"])) / Decimal(60)
        active_reserved_cost += _kind_hourly_cost(kind, plan) * timeout_hours
        active_reserved_hours += Decimal(int(job["vcpu"])) * timeout_hours / Decimal(
            128
        )
    if active_vcpu > int(plan["application"]["production_maximum_vcpu"]):
        raise ValueError("two-pass active jobs exceed 128 vCPU")
    if active_kind_counts["map"] > int(plan["pipeline"]["maximum_concurrent_map_jobs"]):
        raise ValueError("two-pass controller exceeds map concurrency")
    if active_kind_counts["shard_writer"] > int(
        plan["pipeline"]["maximum_concurrent_shard_writers"]
    ):
        raise ValueError("two-pass controller exceeds shard-writer concurrency")
    sums = {
        "compute_usd": ZERO,
        "equivalent_128_vcpu_hours": ZERO,
        "memory_gb_hours": ZERO,
        "storage_gb_hours": ZERO,
        "vcpu_hours": ZERO,
    }
    unique_sources: set[int] = set()
    for job in completed_jobs:
        if not isinstance(job, Mapping):
            raise ValueError("completed two-pass controller job is invalid")
        _validate_job(job, plan, active=False)
        token = str(job["job_token"])
        if token in seen:
            raise ValueError("two-pass controller repeats a job token")
        seen.add(token)
        for name in sums:
            sums[name] += _decimal(job[name], f"completed {name}")
        if job["kind"] == "map":
            source_ordinal = job["metadata"].get("source_ordinal")
            if not isinstance(source_ordinal, int) or isinstance(source_ordinal, bool):
                raise ValueError("completed map job source ordinal is invalid")
            unique_sources.add(source_ordinal)
    cumulative = {
        "compute_usd": str(sums["compute_usd"].quantize(MONEY_QUANTUM)),
        "equivalent_128_vcpu_hours": str(
            sums["equivalent_128_vcpu_hours"].quantize(HOUR_QUANTUM)
        ),
        "job_count": len(completed_jobs),
        "memory_gb_hours": str(sums["memory_gb_hours"].quantize(HOUR_QUANTUM)),
        "storage_gb_hours": str(sums["storage_gb_hours"].quantize(HOUR_QUANTUM)),
        "vcpu_hours": str(sums["vcpu_hours"].quantize(HOUR_QUANTUM)),
    }
    if state.get("cumulative") != cumulative:
        raise ValueError("two-pass controller cumulative accounting changed")
    compute_cap = _decimal(
        authorization["bounds"]["maximum_compute_usd"], "production compute cap"
    )
    hour_cap = _decimal(
        authorization["bounds"]["maximum_cumulative_emr_hours"],
        "production equivalent-hour cap",
    )
    if sums["compute_usd"] + active_reserved_cost > compute_cap:
        raise ValueError("two-pass active jobs exceed the EMR compute cap")
    if sums["equivalent_128_vcpu_hours"] + active_reserved_hours > hour_cap:
        raise ValueError("two-pass active jobs exceed the equivalent-hour cap")
    if len(unique_sources) > int(authorization["bounds"]["maximum_source_files"]):
        raise ValueError("two-pass controller exceeds the source inventory cap")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("two-pass controller artifacts are missing")
    _sha256(artifacts.get("map_checkpoint_sha256"), "controller map checkpoint")
    shard_keys = artifacts.get("shard_result_keys")
    if not isinstance(shard_keys, Mapping):
        raise ValueError("two-pass controller shard results changed")
    if state["status"] == "running" and not active_jobs:
        raise ValueError("two-pass controller is running without an active job")
    if state["status"] == "complete" and state["phase"] != "complete":
        raise ValueError("two-pass controller completed in the wrong phase")
    return dict(state)


def record_jobs_started(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    jobs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    current = validate_two_pass_controller_state(state, plan, authorization)
    if current["status"] in {"complete", "failed", "paused"}:
        raise ValueError("two-pass controller is not accepting new jobs")
    expected_kind = {
        "global_dedup": "global_dedup",
        "mapping": "map",
        "sharding": "shard_writer",
    }.get(str(current["phase"]))
    if expected_kind is None:
        raise ValueError("two-pass controller phase cannot launch jobs")
    additions = []
    for job in jobs:
        kind = str(job.get("kind"))
        if kind != expected_kind:
            raise ValueError("two-pass controller job kind does not match its phase")
        value = {
            "application_id": str(job["application_id"]),
            "job_run_id": str(job["job_run_id"]),
            "job_token": _token(job.get("job_token"), "started job token"),
            "kind": kind,
            "metadata": dict(job.get("metadata", {})),
            "result_key": str(job["result_key"]),
            "timeout_minutes": int(job["timeout_minutes"]),
            "vcpu": _kind_vcpu(kind, plan),
        }
        _validate_job(value, plan, active=True)
        additions.append(value)
    if not additions:
        raise ValueError("two-pass controller cannot start an empty job set")
    result = _next_state(
        current,
        active_jobs=[*current["active_jobs"], *additions],
        status="running",
    )
    return validate_two_pass_controller_state(result, plan, authorization)


def record_job_terminal(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    job_token: str,
    terminal_state: str,
    billed: Mapping[str, Any],
    runtime_seconds: int,
) -> dict[str, Any]:
    current = validate_two_pass_controller_state(state, plan, authorization)
    token = _token(job_token, "terminal job token")
    active = next(
        (job for job in current["active_jobs"] if job["job_token"] == token), None
    )
    if active is None:
        existing = next(
            (job for job in current["completed_jobs"] if job["job_token"] == token),
            None,
        )
        if existing is not None and existing["terminal_state"] == terminal_state:
            return current
        raise ValueError("terminal two-pass job is not active")
    if terminal_state not in TERMINAL_STATES:
        raise ValueError("terminal two-pass job state is invalid")
    priced = billed_job_cost(billed, plan["cost"]["pricing"])
    equivalent = (
        _decimal(priced["vcpu_hours"], "terminal vCPU usage") / Decimal(128)
    ).quantize(HOUR_QUANTUM)
    completed = {
        **active,
        "compute_usd": priced["compute_usd"],
        "equivalent_128_vcpu_hours": str(equivalent),
        "memory_gb_hours": priced["memory_gb_hours"],
        "runtime_seconds": runtime_seconds,
        "storage_gb_hours": priced["storage_gb_hours"],
        "terminal_state": terminal_state,
        "vcpu_hours": priced["vcpu_hours"],
    }
    remaining = [
        dict(job) for job in current["active_jobs"] if job["job_token"] != token
    ]
    completed_jobs = [*current["completed_jobs"], completed]
    sums = {
        name: sum(
            (_decimal(job[name], name) for job in completed_jobs), start=ZERO
        )
        for name in (
            "compute_usd",
            "equivalent_128_vcpu_hours",
            "memory_gb_hours",
            "storage_gb_hours",
            "vcpu_hours",
        )
    }
    cumulative = {
        "compute_usd": str(sums["compute_usd"].quantize(MONEY_QUANTUM)),
        "equivalent_128_vcpu_hours": str(
            sums["equivalent_128_vcpu_hours"].quantize(HOUR_QUANTUM)
        ),
        "job_count": len(completed_jobs),
        "memory_gb_hours": str(sums["memory_gb_hours"].quantize(HOUR_QUANTUM)),
        "storage_gb_hours": str(sums["storage_gb_hours"].quantize(HOUR_QUANTUM)),
        "vcpu_hours": str(sums["vcpu_hours"].quantize(HOUR_QUANTUM)),
    }
    status = (
        "failed"
        if terminal_state != "SUCCESS" or current["status"] == "failed"
        else ("running" if remaining else "ready")
    )
    stop_reason = current.get("stop_reason")
    if terminal_state != "SUCCESS" and stop_reason is None:
        stop_reason = f"EMR job {token} ended as {terminal_state}"
    result = _next_state(
        current,
        active_jobs=remaining,
        completed_jobs=completed_jobs,
        cumulative=cumulative,
        status=status,
        stop_reason=stop_reason,
    )
    return validate_two_pass_controller_state(result, plan, authorization)


def bind_controller_artifacts(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    **artifacts: Any,
) -> dict[str, Any]:
    current = validate_two_pass_controller_state(state, plan, authorization)
    merged = {**current["artifacts"], **artifacts}
    result = _next_state(current, artifacts=merged)
    return validate_two_pass_controller_state(result, plan, authorization)


def stop_two_pass_controller(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    current = validate_two_pass_controller_state(state, plan, authorization)
    if status not in {"failed", "paused"}:
        raise ValueError("two-pass stop status must be failed or paused")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("two-pass stop reason is empty")
    result = _next_state(current, status=status, stop_reason=reason.strip())
    return validate_two_pass_controller_state(result, plan, authorization)


def transition_controller_phase(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    current = validate_two_pass_controller_state(state, plan, authorization)
    if current["status"] in {"failed", "paused"}:
        raise ValueError("two-pass controller cannot transition after a stop condition")
    if current["active_jobs"]:
        raise ValueError("two-pass controller cannot transition with active jobs")
    allowed = {
        "mapping": {"global_dedup"},
        "global_dedup": {"mapping", "sharding"},
        "sharding": {"sealing"},
        "sealing": {"complete"},
        "complete": set(),
    }
    if phase not in allowed[str(current["phase"])]:
        raise ValueError("invalid two-pass controller phase transition")
    status = "complete" if phase == "complete" else "ready"
    result = _next_state(current, phase=phase, status=status, stop_reason=None)
    return validate_two_pass_controller_state(result, plan, authorization)
