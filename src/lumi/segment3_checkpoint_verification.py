"""Exact Segment-2 checkpoint gate for the provisional Segment-3 profile."""

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
    PAID_SEGMENT3_CHECKPOINT_GATE_INSTANCE_IP,
    PAID_SEGMENT3_CHECKPOINT_GATE_MAX_COST_USD,
    PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES,
    PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS,
    PAID_SEGMENT3_CHECKPOINT_GATE_OWNER_APPROVAL_TEMPLATE,
    PAID_SEGMENT3_CHECKPOINT_GATE_PROFILES,
    PAID_SEGMENT3_CHECKPOINT_GATE_RUN_KIND,
    PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256,
    PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_RUN_FINGERPRINT,
    PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_SEQUENCE_CURSOR,
    PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_STEP,
    PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_TOKENS,
    PAID_SEGMENT3_CHECKPOINT_GATE_TOKENS_PER_STEP,
    build_distributed_run_identity,
    verify_distributed_checkpoint,
)
from .production_training import (
    FINAL_CORPUS_SHARD_MANIFEST_GATE,
    validate_final_corpus_training_manifest,
)

CHECKPOINT_GATE = PAID_SEGMENT3_CHECKPOINT_GATE_RUN_KIND
SCHEMA_VERSION = 1
SEGMENT2_SOURCE_COMMIT = "3a19884376b7ec07b4d89019c90eab47e5c6217e"
FINAL_CORPUS_MANIFEST_SHA256 = (
    "6de45464af33d44e1261a023ccb7d50f86346a4126bb2fa060530a91213483e9"
)
MODEL_CONFIG_SHA256 = (
    "643e8da652577f802d959740d66a7e6bafc5b065aa45010c2d7e0340ed9509be"
)
FIRST_PHASE_STOP_STEP = 63_020
WARMUP_STEPS = 8
MEASURED_STEPS = 12
RESUME_CANARY_STEPS = (
    PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS - FIRST_PHASE_STOP_STEP
)
CHECKPOINT_STEPS = 20
PROFILE_ORDER = (
    "segment2_baseline_fsdp_b2_fa2",
    "fsdp_b2_fa3_compile",
)
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


def checkpoint_gate_profiles() -> list[dict[str, Any]]:
    common = {
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": False,
        "optimizer_profile": "legacy",
        "per_rank_batch_size": 2,
        "precision": "bf16",
        "strategy": "fsdp_full_shard",
    }
    return [
        {
            **common,
            "attention_implementation": "flash_attention_2",
            "label": "segment2_baseline_fsdp_b2_fa2",
            "torch_compile": False,
        },
        {
            **common,
            "attention_implementation": "flash_attention_3",
            "label": "fsdp_b2_fa3_compile",
            "torch_compile": True,
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
        strategy=profile["strategy"],
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
        attention_implementation=profile["attention_implementation"],
        document_isolation=True,
        optimizer_profile=profile["optimizer_profile"],
        precision=profile["precision"],
        gradient_sync_mode="last_microbatch",
        torch_compile=profile["torch_compile"],
        source_commit=source_commit,
    )


def _validate_source_checkpoint(source_checkpoint: Path) -> dict[str, Any]:
    source_state = verify_distributed_checkpoint(source_checkpoint)
    source_binding = {
        "manifest_sha256": _sha256_file(source_checkpoint / "manifest.sha256"),
        "run_fingerprint": source_state["run_fingerprint"],
        "sequence_cursor": source_state["sequence_cursor"],
        "step": source_state["step"],
        "tokens": source_state["tokens_seen"],
    }
    expected_binding = {
        "manifest_sha256": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256,
        "run_fingerprint": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_RUN_FINGERPRINT,
        "sequence_cursor": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_SEQUENCE_CURSOR,
        "step": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_STEP,
        "tokens": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_TOKENS,
    }
    if source_binding != expected_binding:
        raise ValueError("checkpoint gate source is not the sealed Segment-2 boundary")
    specification = source_state.get("run_specification")
    if not isinstance(specification, Mapping):
        raise ValueError("Segment-2 source run specification is missing")
    if specification.get("source_commit") != SEGMENT2_SOURCE_COMMIT:
        raise ValueError("Segment-2 source commit changed")
    distributed = specification.get("distributed")
    if not isinstance(distributed, Mapping) or any(
        distributed.get(field) != value
        for field, value in {
            "backend": "nccl",
            "per_rank_batch_size": 2,
            "precision": "bf16",
            "strategy": "fsdp_full_shard",
            "torch_compile": False,
            "world_size": 8,
        }.items()
    ):
        raise ValueError("Segment-2 source distributed profile changed")
    optimizer = specification.get("optimizer")
    if not isinstance(optimizer, Mapping) or any(
        optimizer.get(field) != value
        for field, value in {
            "attention_implementation": "flash_attention_2",
            "batch_size": 2,
            "document_isolation": "position_reset_varlen_v0.1",
            "gradient_accumulation_steps": 16,
            "learning_rate": 0.0002,
            "lr_schedule": "cosine",
            "max_grad_norm": 1.0,
            "min_learning_rate_ratio": 0.1,
            "schedule_steps": 76_294,
            "warmup_steps": 1000,
            "weight_decay": 0.1,
        }.items()
    ):
        raise ValueError("Segment-2 source optimizer profile changed")
    return source_binding


def build_checkpoint_gate_plan(
    *,
    source_checkpoint: str | Path,
    manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
    evaluation_plan_path: str | Path,
    source_commit: str,
    instance_boot_utc: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a sealed plan without claiming owner approval."""

    created = datetime.now(UTC) if created_at is None else created_at.astimezone(UTC)
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("checkpoint gate source commit is invalid")
    source_checkpoint = Path(source_checkpoint).resolve()
    manifest_path = Path(manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    config_path = Path(config_path).resolve()
    evaluation_plan_path = Path(evaluation_plan_path).resolve()
    source_binding = _validate_source_checkpoint(source_checkpoint)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("gate") != FINAL_CORPUS_SHARD_MANIFEST_GATE:
        raise ValueError("pre-Segment-3 checkpoint gate requires the final corpus")
    manifest = validate_final_corpus_training_manifest(
        manifest_path,
        shard_root=manifest_path.parent / "shards",
        sequence_length=4096,
        vocab_size=65_536,
        doc_token_id=3,
        verify_hashes=False,
    )
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != FINAL_CORPUS_MANIFEST_SHA256:
        raise ValueError("final corpus manifest changed")
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != MODEL_CONFIG_SHA256:
        raise ValueError("Lumi-4B model config changed")
    if not tokenizer_path.is_file() or tokenizer_path.name != "tokenizer.json":
        raise ValueError("verified tokenizer path is invalid")
    profiles: list[dict[str, Any]] = []
    for raw_profile in checkpoint_gate_profiles():
        identity = _run_identity(
            config_bytes=config_bytes,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            profile=raw_profile,
            source_commit=source_commit,
        )
        profiles.append(
            {
                **raw_profile,
                "run_fingerprint": _fingerprint(identity),
                "run_identity": identity,
            }
        )
    plan: dict[str, Any] = {
        "benchmark": {
            "checkpoint_steps": CHECKPOINT_STEPS,
            "first_phase_stop_step": FIRST_PHASE_STOP_STEP,
            "measured_steps": MEASURED_STEPS,
            "resume_canary_steps": RESUME_CANARY_STEPS,
            "terminal_step": PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
        },
        "budget": {
            "hourly_rate_usd": 31.92,
            "maximum_cost_usd": PAID_SEGMENT3_CHECKPOINT_GATE_MAX_COST_USD,
            "maximum_runtime_minutes": (
                PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES
            ),
        },
        "created_at": created.isoformat(),
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
            "instance_ip": PAID_SEGMENT3_CHECKPOINT_GATE_INSTANCE_IP,
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
        "source_commit": source_commit,
        "source_training_commit": SEGMENT2_SOURCE_COMMIT,
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
    return plan


def build_checkpoint_gate_authorization(
    *,
    plan: Mapping[str, Any],
    approved_at: datetime | None = None,
) -> dict[str, Any]:
    """Create authorization only after the owner supplied the exact approval."""

    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("checkpoint gate plan hash changed")
    approved = datetime.now(UTC) if approved_at is None else approved_at.astimezone(UTC)
    source_commit = str(plan["source_commit"])
    exact_approval = PAID_SEGMENT3_CHECKPOINT_GATE_OWNER_APPROVAL_TEMPLATE.format(
        source_commit=source_commit,
        source_manifest_sha256=PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256,
    )
    fingerprints = {
        profile["label"]: profile["run_fingerprint"]
        for profile in plan["profiles"]
    }
    authorization: dict[str, Any] = {
        "allowed_run_fingerprints": fingerprints,
        "approved_at": approved.isoformat(),
        "approved_from_step": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_STEP,
        "approved_from_tokens": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_TOKENS,
        "approved_through_step": PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS,
        "approved_through_tokens": (
            PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS
            * PAID_SEGMENT3_CHECKPOINT_GATE_TOKENS_PER_STEP
        ),
        "deadline_utc": (
            approved
            + timedelta(minutes=PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES)
        ).isoformat(),
        "exact_owner_approval": exact_approval,
        "exact_owner_approval_sha256": hashlib.sha256(
            exact_approval.encode("utf-8")
        ).hexdigest(),
        "full_4b_training": False,
        "hourly_rate_usd": 31.92,
        "instance_ip": PAID_SEGMENT3_CHECKPOINT_GATE_INSTANCE_IP,
        "instance_type": "gpu_8x_h100_sxm5",
        "max_cost_usd": PAID_SEGMENT3_CHECKPOINT_GATE_MAX_COST_USD,
        "max_runtime_minutes": PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES,
        "paid_gpu": True,
        "plan_sha256": plan["plan_sha256"],
        "production_training": False,
        "provider": "Lambda Cloud",
        "run_kind": PAID_SEGMENT3_CHECKPOINT_GATE_RUN_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "checkpoint_modification": False,
            "new_instance": False,
            "off_instance_export": False,
            "production_training": False,
            "segment_3": False,
            "termination": False,
        },
        "source_checkpoint": {
            key: plan["source_checkpoint"][key]
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
    return authorization


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


def write_checkpoint_gate_plan(
    *,
    plan: Mapping[str, Any],
    output: str | Path,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(_canonical_json_bytes(plan) + b"\n")


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
        raise ValueError("unsupported pre-Segment-3 checkpoint plan")
    if plan.get("plan_sha256") != _fingerprint(plan, "plan_sha256"):
        raise ValueError("pre-Segment-3 checkpoint plan hash changed")
    if authorization.get("authorization_sha256") != _fingerprint(
        authorization, "authorization_sha256"
    ):
        raise ValueError("pre-Segment-3 checkpoint authorization hash changed")
    if authorization.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("authorization targets another checkpoint plan")
    if (
        authorization.get("status") != "owner_approved"
        or authorization.get("run_kind") != CHECKPOINT_GATE
        or authorization.get("full_4b_training") is not False
        or authorization.get("production_training") is not False
    ):
        raise ValueError("pre-Segment-3 checkpoint authorization scope changed")
    if authorization.get("scope") != {
        "checkpoint_modification": False,
        "new_instance": False,
        "off_instance_export": False,
        "production_training": False,
        "segment_3": False,
        "termination": False,
    }:
        raise ValueError("pre-Segment-3 checkpoint authorization scope changed")
    expected_approval = PAID_SEGMENT3_CHECKPOINT_GATE_OWNER_APPROVAL_TEMPLATE.format(
        source_commit=plan["source_commit"],
        source_manifest_sha256=(
            PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256
        ),
    )
    if authorization.get("exact_owner_approval") != expected_approval:
        raise ValueError("pre-Segment-3 checkpoint owner approval changed")
    if authorization.get("exact_owner_approval_sha256") != hashlib.sha256(
        expected_approval.encode("utf-8")
    ).hexdigest():
        raise ValueError("pre-Segment-3 checkpoint owner approval hash changed")
    if (
        authorization.get("max_runtime_minutes")
        != PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES
        or authorization.get("max_cost_usd")
        != PAID_SEGMENT3_CHECKPOINT_GATE_MAX_COST_USD
        or authorization.get("hourly_rate_usd") != 31.92
    ):
        raise ValueError("pre-Segment-3 checkpoint budget changed")
    labels = tuple(profile.get("label") for profile in plan.get("profiles", []))
    if labels != PROFILE_ORDER or set(labels) != PAID_SEGMENT3_CHECKPOINT_GATE_PROFILES:
        raise ValueError("pre-Segment-3 checkpoint profile order changed")
    expected_profiles = checkpoint_gate_profiles()
    for profile, expected in zip(plan["profiles"], expected_profiles, strict=True):
        if any(profile.get(key) != value for key, value in expected.items()):
            raise ValueError("pre-Segment-3 checkpoint profile changed")
    fingerprints = {
        profile["label"]: profile["run_fingerprint"]
        for profile in plan["profiles"]
    }
    if authorization.get("allowed_run_fingerprints") != fingerprints:
        raise ValueError("pre-Segment-3 checkpoint fingerprints changed")
    expected_source = {
        "manifest_sha256": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256,
        "read_only": True,
        "run_fingerprint": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_RUN_FINGERPRINT,
        "sequence_cursor": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_SEQUENCE_CURSOR,
        "step": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_STEP,
        "tokens": PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_TOKENS,
    }
    if authorization.get("source_checkpoint") != expected_source:
        raise ValueError("pre-Segment-3 checkpoint source authorization changed")
    if any(
        plan["source_checkpoint"].get(key) != value
        for key, value in expected_source.items()
    ):
        raise ValueError("pre-Segment-3 checkpoint source plan changed")
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("pre-Segment-3 checkpoint evaluation binding is missing")
    tokenizer_path = Path(str(evaluation.get("tokenizer_path", ""))).resolve()
    if Path(str(evaluation.get("tokenizer_root_path", ""))).resolve() != (
        tokenizer_path.parent
    ):
        raise ValueError("pre-Segment-3 checkpoint tokenizer root changed")


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
    by_step = {int(metric["step"]): metric for metric in metrics}
    measured_start = PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_STEP + WARMUP_STEPS + 1
    measured = [
        by_step[step] for step in range(measured_start, FIRST_PHASE_STOP_STEP + 1)
    ]
    resume = [
        by_step[step]
        for step in range(
            FIRST_PHASE_STOP_STEP + 1,
            PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS + 1,
        )
    ]
    if len(measured) != MEASURED_STEPS or len(resume) != RESUME_CANARY_STEPS:
        raise ValueError("checkpoint gate metrics do not cover exact bounded steps")
    numeric = [
        float(metric[field])
        for metric in measured + resume
        for field in ("gradient_norm", "loss", "tokens_per_second")
    ]
    if any(not math.isfinite(value) or value <= 0 for value in numeric):
        raise ValueError("checkpoint gate metrics contain non-finite values")
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
        raise ValueError("candidate did not import the exact Segment-2 checkpoint")
    if state.get("run_fingerprint") != profile["run_fingerprint"]:
        raise ValueError("candidate run fingerprint changed")
    if (
        state.get("step") != PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS
        or state.get("tokens_seen")
        != PAID_SEGMENT3_CHECKPOINT_GATE_MAX_STEPS
        * PAID_SEGMENT3_CHECKPOINT_GATE_TOKENS_PER_STEP
    ):
        raise ValueError("candidate did not reach the bounded resume step")
    if source_manifest_sha256_after != (
        PAID_SEGMENT3_CHECKPOINT_GATE_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("sealed Segment-2 checkpoint changed during verification")
    evaluation = json.loads(
        Path(evaluation_report_path).read_text(encoding="utf-8")
    )
    rates = [float(metric["tokens_per_second"]) for metric in measured]
    losses = [float(metric["loss"]) for metric in measured]
    gradients = [float(metric["gradient_norm"]) for metric in measured]
    peak_memory = max(
        int(metric.get("peak_memory_reserved_bytes", 0))
        or int(metric["peak_memory_allocated_bytes"])
        for metric in measured + resume
    )
    return {
        "checkpoint_restore_pass": True,
        "heldout_loss": _finite_positive(
            evaluation.get("loss", {}).get("mean"),
            f"{profile['label']} held-out loss",
        ),
        "label": profile["label"],
        "mean_gradient_norm": statistics.mean(gradients),
        "median_tokens_per_second": statistics.median(rates),
        "p10_tokens_per_second": statistics.quantiles(
            rates, n=10, method="inclusive"
        )[0],
        "peak_memory_bytes": peak_memory,
        "profile": {
            key: profile[key]
            for key in (
                "attention_implementation",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
                "label",
                "optimizer_profile",
                "per_rank_batch_size",
                "precision",
                "strategy",
                "torch_compile",
            )
        },
        "resume_canary_pass": True,
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
    fa3_parity: Mapping[str, Any],
    observed_runtime_minutes: float,
) -> dict[str, Any]:
    validate_checkpoint_gate_bundle(plan=plan, authorization=authorization)
    if observed_runtime_minutes > PAID_SEGMENT3_CHECKPOINT_GATE_MAX_RUNTIME_MINUTES:
        raise ValueError("pre-Segment-3 checkpoint gate exceeded runtime cap")
    if fa3_parity.get("status") != "pass":
        raise ValueError("FA3 parity did not pass")
    by_label = {result.get("label"): result for result in results}
    if tuple(by_label) != PROFILE_ORDER or any(
        result.get("status") != "pass" for result in results
    ):
        raise ValueError("checkpoint gate requires two passing exact candidates")
    baseline = by_label[PROFILE_ORDER[0]]
    candidate = by_label[PROFILE_ORDER[1]]
    median_gain = (
        _finite_positive(
            candidate["median_tokens_per_second"], "candidate median throughput"
        )
        / _finite_positive(
            baseline["median_tokens_per_second"], "baseline median throughput"
        )
        - 1
    )
    p10_gain = (
        _finite_positive(candidate["p10_tokens_per_second"], "candidate p10")
        / _finite_positive(baseline["p10_tokens_per_second"], "baseline p10")
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
    reasons: list[str] = []
    if median_gain < thresholds["minimum_median_throughput_gain_fraction"]:
        reasons.append("median_gain_below_threshold")
    if p10_gain < thresholds["minimum_p10_throughput_gain_fraction"]:
        reasons.append("p10_gain_below_threshold")
    if training_loss_regression > thresholds[
        "maximum_training_loss_regression_fraction"
    ]:
        reasons.append("training_loss_regression")
    if heldout_loss_regression > thresholds[
        "maximum_heldout_loss_regression_fraction"
    ]:
        reasons.append("heldout_loss_regression")
    if candidate["peak_memory_bytes"] > thresholds["maximum_peak_memory_bytes"]:
        reasons.append("insufficient_hbm_reserve")
    selected = PROFILE_ORDER[0] if reasons else PROFILE_ORDER[1]
    report: dict[str, Any] = {
        "authorization_sha256": authorization["authorization_sha256"],
        "comparisons": {
            "heldout_loss_regression_fraction": heldout_loss_regression,
            "median_throughput_gain_fraction": median_gain,
            "p10_throughput_gain_fraction": p10_gain,
            "training_loss_regression_fraction": training_loss_regression,
        },
        "fa3_parity": dict(fa3_parity),
        "gate": CHECKPOINT_GATE,
        "observed_cost_usd": observed_runtime_minutes / 60 * 31.92,
        "observed_runtime_minutes": observed_runtime_minutes,
        "plan_sha256": plan["plan_sha256"],
        "production_training_authorized": False,
        "reasons": reasons,
        "results": results,
        "schema_version": SCHEMA_VERSION,
        "selected_profile": selected,
        "source_checkpoint": plan["source_checkpoint"],
        "status": "pass" if not reasons else "safe_fallback",
    }
    report["report_sha256"] = _fingerprint(report)
    return report
