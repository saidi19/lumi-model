"""Exact-checkpoint verification contract for the provisional Segment-2 profile."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .distributed_training import (
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_COST_USD,
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES,
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
    PAID_SEGMENT2_CHECKPOINT_GATE_OWNER_ACKNOWLEDGEMENT,
    PAID_SEGMENT2_CHECKPOINT_GATE_RUN_KIND,
    PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_STEP,
    PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_TOKENS,
    PAID_SEGMENT2_CHECKPOINT_GATE_TOKENS_PER_STEP,
    build_distributed_run_identity,
    distributed_checkpoint_binding,
    verify_distributed_checkpoint,
)
from .production_training import (
    FINAL_CORPUS_SHARD_MANIFEST_GATE,
    validate_final_corpus_training_manifest,
)

CHECKPOINT_GATE = "lumi_4b_pre_segment2_checkpoint_gate_v0.1"
SCHEMA_VERSION = 1
FIRST_PHASE_STOP_STEP = 31_520
WARMUP_STEPS = 8
MEASURED_STEPS = 12
RESUME_CANARY_STEPS = (
    PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS - FIRST_PHASE_STOP_STEP
)
CHECKPOINT_STEPS = 20
PROFILE_ORDER = ("baseline_b1_gc", "batch2_no_gc")
MAXIMUM_TRAINING_LOSS_REGRESSION_FRACTION = 0.01
MAXIMUM_HELDOUT_LOSS_REGRESSION_FRACTION = 0.005
MINIMUM_MEDIAN_THROUGHPUT_GAIN_FRACTION = 0.15
MINIMUM_P10_THROUGHPUT_GAIN_FRACTION = 0.10
MAXIMUM_PEAK_MEMORY_BYTES = 76_000_000_000


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Mapping[str, Any], omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_gate_profiles(
    *, optimizer_profile: str = "legacy"
) -> list[dict[str, Any]]:
    return [
        {
            "gradient_accumulation_steps": 32,
            "gradient_checkpointing": True,
            "label": "baseline_b1_gc",
            "optimizer_profile": optimizer_profile,
            "per_rank_batch_size": 1,
        },
        {
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": False,
            "label": "batch2_no_gc",
            "optimizer_profile": optimizer_profile,
            "per_rank_batch_size": 2,
        },
    ]


def _run_identity(
    *,
    config_bytes: bytes,
    manifest: dict[str, Any],
    manifest_sha256: str,
    profile: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    return build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        backend="nccl",
        world_size=8,
        strategy="fsdp_full_shard",
        per_rank_batch_size=profile["per_rank_batch_size"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        learning_rate=0.0002,
        weight_decay=0.1,
        max_grad_norm=1.0,
        keep_checkpoints=1,
        checkpoint_steps=CHECKPOINT_STEPS,
        seed=7,
        gradient_checkpointing=profile["gradient_checkpointing"],
        lr_schedule="cosine",
        warmup_steps=1000,
        schedule_steps=76_294,
        min_learning_rate_ratio=0.1,
        attention_implementation="flash_attention_2",
        document_isolation=True,
        optimizer_profile=profile["optimizer_profile"],
        precision="bf16",
        gradient_sync_mode="last_microbatch",
        torch_compile=False,
        source_commit=source_commit,
    )


def build_checkpoint_gate_bundle(
    *,
    source_checkpoint: str | Path,
    manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
    evaluation_plan_path: str | Path,
    source_commit: str,
    instance_boot_utc: str,
    approved_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact two-profile plan and owner authorization."""

    approved = datetime.now(UTC) if approved_at is None else approved_at.astimezone(UTC)
    manifest_path = Path(manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    config_path = Path(config_path).resolve()
    evaluation_plan_path = Path(evaluation_plan_path).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("gate") == FINAL_CORPUS_SHARD_MANIFEST_GATE:
        manifest = validate_final_corpus_training_manifest(
            manifest_path,
            shard_root=manifest_path.parent / "shards",
            sequence_length=4096,
            vocab_size=65_536,
            doc_token_id=3,
            verify_hashes=False,
        )
    else:
        manifest = raw_manifest
    config_bytes = config_path.read_bytes()
    manifest_sha256 = _sha256_file(manifest_path)
    source_binding = distributed_checkpoint_binding(source_checkpoint)
    source_state = verify_distributed_checkpoint(source_checkpoint)
    source_optimizer = source_state.get("run_specification", {}).get("optimizer")
    if not isinstance(source_optimizer, Mapping):
        raise ValueError("checkpoint gate source optimizer identity is missing")
    source_optimizer_profile = source_optimizer.get("optimizer_profile", "legacy")
    if source_optimizer_profile != "legacy":
        raise ValueError("checkpoint gate source optimizer profile changed")
    expected_source_optimizer = {
        "learning_rate": 0.0002,
        "lr_schedule": "cosine",
        "max_grad_norm": 1.0,
        "min_learning_rate_ratio": 0.1,
        "schedule_steps": 76_294,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
    }
    if any(
        source_optimizer.get(field) != value
        for field, value in expected_source_optimizer.items()
    ):
        raise ValueError("checkpoint gate source optimizer contract changed")
    expected_source = {
        "step": PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_STEP,
        "tokens": PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_TOKENS,
    }
    if any(source_binding[field] != value for field, value in expected_source.items()):
        raise ValueError("checkpoint gate source is not the sealed Segment-1 boundary")

    profiles: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for raw_profile in checkpoint_gate_profiles(
        optimizer_profile=source_optimizer_profile
    ):
        identity = _run_identity(
            config_bytes=config_bytes,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            profile=raw_profile,
            source_commit=source_commit,
        )
        profile = {
            **raw_profile,
            "run_fingerprint": _fingerprint(identity),
            "run_identity": identity,
        }
        profiles.append(profile)
        fingerprints[profile["label"]] = profile["run_fingerprint"]

    plan: dict[str, Any] = {
        "benchmark": {
            "checkpoint_steps": CHECKPOINT_STEPS,
            "first_phase_stop_step": FIRST_PHASE_STOP_STEP,
            "measured_steps": MEASURED_STEPS,
            "resume_canary_steps": RESUME_CANARY_STEPS,
            "terminal_step": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
        },
        "budget": {
            "hourly_rate_usd": 31.92,
            "maximum_cost_usd": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_COST_USD,
            "maximum_runtime_minutes": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES,
        },
        "created_at": approved.isoformat(),
        "evaluation": {
            "heldout_manifest_path": str(heldout_manifest_path),
            "heldout_manifest_sha256": _sha256_file(heldout_manifest_path),
            "plan_path": str(evaluation_plan_path),
            "plan_sha256": _sha256_file(evaluation_plan_path),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_root_path": str(tokenizer_path.parent),
            "tokenizer_sha256": _sha256_file(tokenizer_path),
        },
        "gate": CHECKPOINT_GATE,
        "hardware": {
            "instance_ip": "209.20.157.24",
            "instance_type": "gpu_8x_h100_sxm5",
            "world_size": 8,
        },
        "instance_boot_utc": instance_boot_utc,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model_config_path": str(config_path),
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "profiles": profiles,
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint": {
            **source_binding,
            "path": str(source_checkpoint),
            "read_only": True,
        },
        "source_optimizer_profile": source_optimizer_profile,
        "source_commit": source_commit,
        "status": "sealed",
        "thresholds": {
            "maximum_heldout_loss_regression_fraction": (
                MAXIMUM_HELDOUT_LOSS_REGRESSION_FRACTION
            ),
            "maximum_peak_memory_bytes": MAXIMUM_PEAK_MEMORY_BYTES,
            "maximum_training_loss_regression_fraction": (
                MAXIMUM_TRAINING_LOSS_REGRESSION_FRACTION
            ),
            "minimum_median_throughput_gain_fraction": (
                MINIMUM_MEDIAN_THROUGHPUT_GAIN_FRACTION
            ),
            "minimum_p10_throughput_gain_fraction": (
                MINIMUM_P10_THROUGHPUT_GAIN_FRACTION
            ),
        },
    }
    plan["plan_sha256"] = _fingerprint(plan)

    authorization: dict[str, Any] = {
        "allowed_run_fingerprints": fingerprints,
        "approved_at": approved.isoformat(),
        "approved_from_step": PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_STEP,
        "approved_from_tokens": PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_TOKENS,
        "approved_through_step": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS,
        "approved_through_tokens": (
            PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS
            * PAID_SEGMENT2_CHECKPOINT_GATE_TOKENS_PER_STEP
        ),
        "deadline_utc": (
            approved
            + timedelta(minutes=PAID_SEGMENT2_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES)
        ).isoformat(),
        "full_4b_training": False,
        "hourly_rate_usd": 31.92,
        "instance_ip": "209.20.157.24",
        "instance_type": "gpu_8x_h100_sxm5",
        "max_cost_usd": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_COST_USD,
        "max_runtime_minutes": PAID_SEGMENT2_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES,
        "owner_acknowledgement": PAID_SEGMENT2_CHECKPOINT_GATE_OWNER_ACKNOWLEDGEMENT,
        "paid_gpu": True,
        "plan_sha256": plan["plan_sha256"],
        "provider": "Lambda Cloud",
        "run_kind": PAID_SEGMENT2_CHECKPOINT_GATE_RUN_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "checkpoint_modification": False,
            "new_instance": False,
            "off_instance_export": False,
            "production_training": False,
            "termination": False,
        },
        "source_checkpoint": {
            key: source_binding[key]
            for key in (
                "manifest_sha256",
                "run_fingerprint",
                "sequence_cursor",
                "step",
                "tokens",
            )
        }
        | {"read_only": True},
        "source_commit": source_commit,
        "status": "owner_approved",
        "strategy": "fsdp_full_shard",
        "world_size": 8,
    }
    authorization["authorization_sha256"] = _fingerprint(authorization)
    return plan, authorization


def write_checkpoint_gate_bundle(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output: str | Path,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(_canonical_json_bytes(plan) + b"\n")
    (output / "authorization.json").write_bytes(
        _canonical_json_bytes(authorization) + b"\n"
    )


def validate_checkpoint_gate_bundle(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("gate") != CHECKPOINT_GATE
        or plan.get("status") != "sealed"
    ):
        raise ValueError("unsupported checkpoint-gate plan")
    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("checkpoint-gate plan hash changed")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization,
        "authorization_sha256",
    ):
        raise ValueError("checkpoint-gate authorization hash changed")
    if authorization.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("checkpoint-gate authorization targets another plan")
    labels = tuple(profile.get("label") for profile in plan.get("profiles", []))
    if labels != PROFILE_ORDER:
        raise ValueError("checkpoint-gate profile order changed")
    if plan.get("source_optimizer_profile") != "legacy" or any(
        profile.get("optimizer_profile") != plan["source_optimizer_profile"]
        for profile in plan["profiles"]
    ):
        raise ValueError("checkpoint-gate optimizer identity changed")
    fingerprints = {
        profile["label"]: profile["run_fingerprint"]
        for profile in plan["profiles"]
    }
    if authorization.get("allowed_run_fingerprints") != fingerprints:
        raise ValueError("checkpoint-gate fingerprints changed")
    if authorization.get("source_checkpoint") != {
        key: plan["source_checkpoint"][key]
        for key in (
            "manifest_sha256",
            "run_fingerprint",
            "sequence_cursor",
            "step",
            "tokens",
        )
    } | {"read_only": True}:
        raise ValueError("checkpoint-gate source binding changed")
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("checkpoint-gate evaluation binding is missing")
    tokenizer_path = Path(str(evaluation.get("tokenizer_path", ""))).resolve()
    tokenizer_root_path = Path(
        str(evaluation.get("tokenizer_root_path", ""))
    ).resolve()
    if tokenizer_root_path != tokenizer_path.parent:
        raise ValueError("checkpoint-gate tokenizer root changed")


def _finite_positive(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def summarize_candidate(
    *,
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    metrics_path: str | Path,
    state_path: str | Path,
    evaluation_report_path: str | Path,
    source_manifest_sha256_after: str,
) -> dict[str, Any]:
    metrics = [
        json.loads(line)
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_step = {metric["step"]: metric for metric in metrics}
    measured_start = (
        PAID_SEGMENT2_CHECKPOINT_GATE_SOURCE_STEP + WARMUP_STEPS + 1
    )
    measured = [
        by_step[step]
        for step in range(measured_start, FIRST_PHASE_STOP_STEP + 1)
    ]
    resume = [
        by_step[step]
        for step in range(
            FIRST_PHASE_STOP_STEP + 1,
            PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS + 1,
        )
    ]
    if len(measured) != MEASURED_STEPS or len(resume) != RESUME_CANARY_STEPS:
        raise ValueError("checkpoint-gate metrics do not cover exact bounded steps")
    if any(
        not math.isfinite(float(metric["loss"]))
        or not math.isfinite(float(metric["tokens_per_second"]))
        for metric in measured + resume
    ):
        raise ValueError("checkpoint-gate metrics contain non-finite values")

    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    expected_source = {
        key: plan["source_checkpoint"][key]
        for key in (
            "manifest_sha256",
            "run_fingerprint",
            "sequence_cursor",
            "step",
            "tokens",
        )
    }
    if state.get("imported_source_checkpoint") != expected_source:
        raise ValueError("candidate did not import the exact checkpoint")
    if (
        state.get("step") != PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS
        or state.get("tokens_seen")
        != PAID_SEGMENT2_CHECKPOINT_GATE_MAX_STEPS
        * PAID_SEGMENT2_CHECKPOINT_GATE_TOKENS_PER_STEP
    ):
        raise ValueError("candidate did not reach the bounded resume step")
    if source_manifest_sha256_after != plan["source_checkpoint"]["manifest_sha256"]:
        raise ValueError("sealed Segment-1 checkpoint changed during verification")

    evaluation = json.loads(
        Path(evaluation_report_path).read_text(encoding="utf-8")
    )
    throughputs = [float(metric["tokens_per_second"]) for metric in measured]
    losses = [float(metric["loss"]) for metric in measured]
    p10_index = max(0, math.ceil(len(throughputs) * 0.1) - 1)
    p10 = sorted(throughputs)[p10_index]
    peak_memory = max(
        int(metric["peak_memory_allocated_bytes"]) for metric in measured + resume
    )
    return {
        "checkpoint_restore_pass": True,
        "heldout_loss": _finite_positive(
            evaluation.get("loss", {}).get("mean"),
            f"{profile['label']} held-out loss",
        ),
        "label": profile["label"],
        "median_tokens_per_second": statistics.median(throughputs),
        "p10_tokens_per_second": p10,
        "peak_memory_allocated_bytes": peak_memory,
        "profile": {
            key: profile[key]
            for key in (
                "gradient_accumulation_steps",
                "gradient_checkpointing",
                "label",
                "optimizer_profile",
                "per_rank_batch_size",
            )
        },
        "resume_canary_pass": all(
            int(metric["step"]) > FIRST_PHASE_STOP_STEP for metric in resume
        ),
        "run_fingerprint": profile["run_fingerprint"],
        "source_checkpoint": expected_source,
        "source_checkpoint_unchanged": True,
        "status": "pass",
        "training_loss": statistics.mean(losses),
    }


def decide_checkpoint_gate(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    observed_runtime_minutes: float,
) -> dict[str, Any]:
    validate_checkpoint_gate_bundle(plan=plan, authorization=authorization)
    if observed_runtime_minutes > PAID_SEGMENT2_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES:
        raise ValueError("checkpoint gate exceeded its runtime cap")
    by_label = {result.get("label"): result for result in results}
    if tuple(by_label) != PROFILE_ORDER or any(
        result.get("status") != "pass" for result in results
    ):
        raise ValueError("checkpoint gate requires two passing exact candidates")
    baseline = by_label["baseline_b1_gc"]
    candidate = by_label["batch2_no_gc"]
    median_gain = (
        _finite_positive(
            candidate["median_tokens_per_second"],
            "candidate median throughput",
        )
        / _finite_positive(
            baseline["median_tokens_per_second"],
            "baseline median throughput",
        )
        - 1
    )
    p10_gain = (
        _finite_positive(
            candidate["p10_tokens_per_second"],
            "candidate p10 throughput",
        )
        / _finite_positive(
            baseline["p10_tokens_per_second"],
            "baseline p10 throughput",
        )
        - 1
    )
    training_loss_regression = (
        _finite_positive(candidate["training_loss"], "candidate training loss")
        / _finite_positive(baseline["training_loss"], "baseline training loss")
        - 1
    )
    heldout_loss_regression = (
        _finite_positive(candidate["heldout_loss"], "candidate held-out loss")
        / _finite_positive(baseline["heldout_loss"], "baseline held-out loss")
        - 1
    )
    thresholds = plan["thresholds"]
    reasons = []
    if median_gain < thresholds["minimum_median_throughput_gain_fraction"]:
        reasons.append("median_gain_below_threshold")
    if p10_gain < thresholds["minimum_p10_throughput_gain_fraction"]:
        reasons.append("p10_gain_below_threshold")
    if (
        training_loss_regression
        > thresholds["maximum_training_loss_regression_fraction"]
    ):
        reasons.append("training_loss_regression")
    if (
        heldout_loss_regression
        > thresholds["maximum_heldout_loss_regression_fraction"]
    ):
        reasons.append("heldout_loss_regression")
    if (
        candidate["peak_memory_allocated_bytes"]
        > thresholds["maximum_peak_memory_bytes"]
    ):
        reasons.append("insufficient_hbm_reserve")
    selected = "baseline_b1_gc" if reasons else "batch2_no_gc"
    report: dict[str, Any] = {
        "authorization_sha256": authorization["authorization_sha256"],
        "comparisons": {
            "heldout_loss_regression_fraction": heldout_loss_regression,
            "median_throughput_gain_fraction": median_gain,
            "p10_throughput_gain_fraction": p10_gain,
            "training_loss_regression_fraction": training_loss_regression,
        },
        "gate": CHECKPOINT_GATE,
        "observed_cost_usd": observed_runtime_minutes / 60 * 31.92,
        "observed_runtime_minutes": observed_runtime_minutes,
        "plan_sha256": plan["plan_sha256"],
        "reasons": reasons,
        "results": results,
        "schema_version": SCHEMA_VERSION,
        "selected_profile": selected,
        "source_checkpoint": plan["source_checkpoint"],
        "status": "pass" if not reasons else "safe_fallback",
    }
    report["report_sha256"] = _fingerprint(report)
    return report
