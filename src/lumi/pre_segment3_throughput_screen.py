"""Fail-closed, screening-only optimization gate before Lumi Segment 3."""

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
    PAID_PRE_SEGMENT3_SCREEN_AGGREGATE_TOKEN_CAP,
    PAID_PRE_SEGMENT3_SCREEN_CANDIDATES,
    PAID_PRE_SEGMENT3_SCREEN_INSTANCE_IP,
    PAID_PRE_SEGMENT3_SCREEN_MAX_COST_USD,
    PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES,
    PAID_PRE_SEGMENT3_SCREEN_MAX_STEPS,
    PAID_PRE_SEGMENT3_SCREEN_OWNER_APPROVAL,
    PAID_PRE_SEGMENT3_SCREEN_RUN_KIND,
    PAID_PRE_SEGMENT3_SCREEN_TOKEN_CAP,
    PAID_PRE_SEGMENT3_SCREEN_WORLD_SIZE,
    build_distributed_run_identity,
)
from .evaluation_data import validate_heldout_manifest
from .packed_data import validate_packed_manifest
from .pilot_training import _fingerprint, _write_json_atomic

SCHEMA_VERSION = 1
BASELINE_LABEL = "baseline_fsdp_fa2"
REQUIRED_LABELS = (
    BASELINE_LABEL,
    "ddp_b1_gc_fa2",
    "fsdp_b2_fa2_compile",
    "fsdp_b2_fa3",
    "fsdp_b2_fa3_compile",
)
MEASURED_STEP_START = 5
MEASURED_STEP_END = 16
RESUME_CANARY_STEP = 17


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _validate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(payload)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pre-Segment-3 screen contract")
    if contract.get("status") != "planning_only":
        raise ValueError("pre-Segment-3 contract status changed")
    if contract.get("approval") != {
        "exact_owner_approval": PAID_PRE_SEGMENT3_SCREEN_OWNER_APPROVAL,
        "full_4b_training": False,
        "production_training": False,
    }:
        raise ValueError("pre-Segment-3 owner approval changed")
    benchmark = contract.get("benchmark")
    if benchmark != {
        "global_sequences_per_step": 256,
        "measured_steps_per_candidate": 12,
        "resume_canary_step": RESUME_CANARY_STEP,
        "sequence_length": 4096,
        "warmup_steps_per_candidate": 4,
    }:
        raise ValueError("pre-Segment-3 benchmark geometry changed")
    if contract.get("budget") != {
        "hourly_rate_usd": 31.92,
        "maximum_cost_usd": PAID_PRE_SEGMENT3_SCREEN_MAX_COST_USD,
        "maximum_runtime_minutes": PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES,
    }:
        raise ValueError("pre-Segment-3 budget changed")
    hardware = contract.get("hardware")
    if hardware != {
        "gpu_model_contains": "H100",
        "instance_ip": PAID_PRE_SEGMENT3_SCREEN_INSTANCE_IP,
        "instance_type": "gpu_8x_h100_sxm5",
        "world_size": PAID_PRE_SEGMENT3_SCREEN_WORLD_SIZE,
    }:
        raise ValueError("pre-Segment-3 hardware changed")
    profiles = contract.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != PAID_PRE_SEGMENT3_SCREEN_CANDIDATES:
        raise ValueError("pre-Segment-3 profile count changed")
    if tuple(profile.get("label") for profile in profiles) != REQUIRED_LABELS:
        raise ValueError("pre-Segment-3 profile labels changed")
    for profile in profiles:
        required = {
            "attention_implementation",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "label",
            "per_rank_batch_size",
            "precision",
            "strategy",
            "torch_compile",
        }
        if set(profile) != required:
            raise ValueError(f"pre-Segment-3 profile fields changed: {profile.get('label')}")
        if profile["strategy"] not in {"ddp", "fsdp_full_shard"}:
            raise ValueError("pre-Segment-3 profile strategy changed")
        if profile["attention_implementation"] not in {
            "flash_attention_2",
            "flash_attention_3",
        }:
            raise ValueError("pre-Segment-3 attention backend changed")
        global_sequences = (
            PAID_PRE_SEGMENT3_SCREEN_WORLD_SIZE
            * profile["per_rank_batch_size"]
            * profile["gradient_accumulation_steps"]
        )
        if global_sequences != 256:
            raise ValueError("pre-Segment-3 profile changed the global batch")
    return contract


def _tokenizer_inventory(tokenizer_root: Path, heldout: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for expected in heldout["tokenizer"]["files"]:
        path = tokenizer_root / expected["name"]
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"pre-Segment-3 tokenizer changed: {expected['name']}")
        inventory.append(dict(expected))
    return inventory


def build_plan(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    tokenizer_root: str | Path,
    config_path: str | Path,
    source_commit: str,
    instance_boot_utc: str,
) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    tokenizer_root = Path(tokenizer_root).resolve()
    config_path = Path(config_path).resolve()
    contract = _validate_contract(json.loads(contract_path.read_text(encoding="utf-8")))
    manifest = validate_packed_manifest(manifest_path, verify_hashes=True)
    heldout = validate_heldout_manifest(heldout_manifest_path)
    source = contract["source"]
    if manifest["statistics"]["tokens"] != 83_886_080:
        raise ValueError("pre-Segment-3 bounded prefix token count changed")
    if manifest["packing"]["sequence_length"] != 4096:
        raise ValueError("pre-Segment-3 sequence length changed")
    if _sha256(manifest_path) != source["bounded_manifest_sha256"]:
        raise ValueError("pre-Segment-3 bounded manifest changed")
    if manifest["provenance"]["final_manifest_sha256"] != source["corpus_manifest_sha256"]:
        raise ValueError("pre-Segment-3 corpus binding changed")
    if _sha256(config_path) != source["model_config_sha256"]:
        raise ValueError("pre-Segment-3 model config changed")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("pre-Segment-3 source commit is invalid")
    tokenizer_files = _tokenizer_inventory(tokenizer_root, heldout)
    boot = datetime.fromisoformat(instance_boot_utc).astimezone(UTC)
    deadline = boot + timedelta(minutes=PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES)
    manifest_sha256 = _sha256(manifest_path)
    identities: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for profile in contract["profiles"]:
        identity = build_distributed_run_identity(
            config_bytes=config_path.read_bytes(),
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            backend="nccl",
            world_size=PAID_PRE_SEGMENT3_SCREEN_WORLD_SIZE,
            strategy=profile["strategy"],
            per_rank_batch_size=profile["per_rank_batch_size"],
            gradient_accumulation_steps=profile["gradient_accumulation_steps"],
            learning_rate=0.0002,
            weight_decay=0.1,
            max_grad_norm=1.0,
            keep_checkpoints=1,
            seed=7,
            gradient_checkpointing=profile["gradient_checkpointing"],
            lr_schedule="cosine",
            warmup_steps=4,
            schedule_steps=RESUME_CANARY_STEP,
            min_learning_rate_ratio=0.1,
            attention_implementation=profile["attention_implementation"],
            document_isolation=True,
            optimizer_profile="legacy",
            precision=profile["precision"],
            gradient_sync_mode="last_microbatch",
            torch_compile=profile["torch_compile"],
            source_commit=source_commit,
        )
        label = profile["label"]
        identities[label] = identity
        fingerprints[label] = _fingerprint(identity)
    plan: dict[str, Any] = {
        "budget": {
            "deadline_utc": deadline.replace(microsecond=0).isoformat(),
            "hourly_rate_usd": 31.92,
            "instance_boot_utc": boot.replace(microsecond=0).isoformat(),
            "maximum_cost_usd": PAID_PRE_SEGMENT3_SCREEN_MAX_COST_USD,
            "maximum_runtime_minutes": PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES,
        },
        "contract_sha256": _sha256(contract_path),
        "created_at": _utc_now(),
        "data": {
            "heldout_manifest_sha256": _sha256(heldout_manifest_path),
            "manifest_sha256": manifest_sha256,
            "source_prefix_sha256": manifest["provenance"]["source_prefix_sha256"],
            "tokenizer_files": tokenizer_files,
            "training_tokens": manifest["statistics"]["tokens"],
        },
        "instance": contract["hardware"],
        "profiles": contract["profiles"],
        "run_fingerprints": fingerprints,
        "run_identities": identities,
        "run_kind": PAID_PRE_SEGMENT3_SCREEN_RUN_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "running_segment_2_access": False,
            "sealed_segment_1_checkpoint_used": False,
            "screening_only": True,
        },
        "source_commit": source_commit,
        "status": "sealed",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return plan


def build_authorization(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("plan_sha256") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("pre-Segment-3 plan hash changed")
    profiles = {
        profile["label"]: {
            "run_fingerprint": plan["run_fingerprints"][profile["label"]],
            "strategy": profile["strategy"],
        }
        for profile in plan["profiles"]
    }
    authorization: dict[str, Any] = {
        "aggregate_candidate_count": PAID_PRE_SEGMENT3_SCREEN_CANDIDATES,
        "aggregate_token_cap": PAID_PRE_SEGMENT3_SCREEN_AGGREGATE_TOKEN_CAP,
        "allowed_profiles": profiles,
        "allowed_run_fingerprints": plan["run_fingerprints"],
        "allowed_strategies": ["ddp", "fsdp_full_shard"],
        "approved_at": _utc_now(),
        "approved_through_step": PAID_PRE_SEGMENT3_SCREEN_MAX_STEPS,
        "approved_token_cap": PAID_PRE_SEGMENT3_SCREEN_TOKEN_CAP,
        "deadline_utc": plan["budget"]["deadline_utc"],
        "exact_owner_approval": PAID_PRE_SEGMENT3_SCREEN_OWNER_APPROVAL,
        "exact_owner_approval_sha256": hashlib.sha256(
            PAID_PRE_SEGMENT3_SCREEN_OWNER_APPROVAL.encode("utf-8")
        ).hexdigest(),
        "full_4b_training": False,
        "hourly_rate_usd": 31.92,
        "instance_boot_utc": plan["budget"]["instance_boot_utc"],
        "instance_ip": PAID_PRE_SEGMENT3_SCREEN_INSTANCE_IP,
        "instance_type": "gpu_8x_h100_sxm5",
        "max_cost_usd": PAID_PRE_SEGMENT3_SCREEN_MAX_COST_USD,
        "max_runtime_minutes": PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES,
        "paid_gpu": True,
        "plan_sha256": plan["plan_sha256"],
        "production_training": False,
        "provider": "Lambda Cloud",
        "run_kind": PAID_PRE_SEGMENT3_SCREEN_RUN_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "inputs": ["bounded_training_prefix", "heldout_data", "tokenizer"],
            "running_segment_2": {
                "access": False,
                "instance_ip": "192.222.52.81",
                "modify": False,
            },
            "sealed_segment_1_checkpoint": {"allowed": True, "used": False},
        },
        "source_commit": plan["source_commit"],
        "status": "owner_approved",
        "strategy": "profile_bound",
        "world_size": PAID_PRE_SEGMENT3_SCREEN_WORLD_SIZE,
    }
    authorization["authorization_sha256"] = _fingerprint(authorization)
    return authorization


def validate_files(
    *, plan_path: str | Path, authorization_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("pre-Segment-3 plan hash changed")
    if authorization.get("authorization_sha256") != _fingerprint(
        {key: value for key, value in authorization.items() if key != "authorization_sha256"}
    ):
        raise ValueError("pre-Segment-3 authorization hash changed")
    if authorization.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("pre-Segment-3 authorization targets another plan")
    return plan, authorization


def summarize_candidate(
    *, profile: Mapping[str, Any], metrics_path: str | Path, state_path: str | Path
) -> dict[str, Any]:
    metrics = [
        json.loads(line)
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    measured = [
        item
        for item in metrics
        if MEASURED_STEP_START <= item.get("step", -1) <= MEASURED_STEP_END
    ]
    if len(measured) != MEASURED_STEP_END - MEASURED_STEP_START + 1:
        raise ValueError(f"candidate {profile['label']} metrics are incomplete")
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if state.get("step") != RESUME_CANARY_STEP:
        raise ValueError(f"candidate {profile['label']} resume canary failed")
    rates = [float(item["tokens_per_second"]) for item in measured]
    losses = [float(item["loss"]) for item in measured]
    gradients = [float(item["gradient_norm"]) for item in measured]
    if any(not math.isfinite(value) or value <= 0 for value in [*rates, *losses, *gradients]):
        raise ValueError(f"candidate {profile['label']} has non-finite metrics")
    return {
        "label": profile["label"],
        "measured_steps": len(measured),
        "metrics": {
            "mean_gradient_norm": statistics.mean(gradients),
            "median_tokens_per_second": statistics.median(rates),
            "p10_tokens_per_second": statistics.quantiles(
                rates, n=10, method="inclusive"
            )[0],
            "peak_memory_bytes": max(int(item["peak_memory_reserved_bytes"]) for item in measured),
            "training_loss": statistics.mean(losses),
        },
        "non_finite_count": 0,
        "profile": dict(profile),
        "resume_canary_pass": True,
        "run_fingerprint": state["run_fingerprint"],
        "status": "pass",
        "warmup_steps": MEASURED_STEP_START - 1,
    }


def decide(
    *, contract: Mapping[str, Any], results: list[Mapping[str, Any]], runtime_minutes: float
) -> dict[str, Any]:
    validated = _validate_contract(contract)
    cost = runtime_minutes / 60 * 31.92
    if runtime_minutes > PAID_PRE_SEGMENT3_SCREEN_MAX_RUNTIME_MINUTES:
        raise ValueError("pre-Segment-3 runtime cap exceeded")
    if cost > PAID_PRE_SEGMENT3_SCREEN_MAX_COST_USD:
        raise ValueError("pre-Segment-3 cost cap exceeded")
    by_label = {str(item["label"]): item for item in results}
    baseline = by_label.get(BASELINE_LABEL)
    if baseline is None or baseline.get("status") != "pass":
        raise ValueError("pre-Segment-3 baseline failed")
    thresholds = validated["decision"]
    baseline_metrics = baseline["metrics"]
    if baseline_metrics["p10_tokens_per_second"] < thresholds[
        "minimum_baseline_p10_tokens_per_second"
    ]:
        raise ValueError("pre-Segment-3 baseline throughput floor failed")
    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    for label in REQUIRED_LABELS[1:]:
        candidate = by_label.get(label, {"status": "not_run"})
        if candidate.get("status") != "pass":
            comparisons[label] = {
                "eligible": False,
                "reasons": [candidate.get("status", "failed")],
            }
            continue
        metrics = candidate["metrics"]
        gains = {
            "p10_throughput_gain_fraction": (
                metrics["p10_tokens_per_second"]
                / baseline_metrics["p10_tokens_per_second"]
                - 1
            ),
            "throughput_gain_fraction": (
                metrics["median_tokens_per_second"]
                / baseline_metrics["median_tokens_per_second"]
                - 1
            ),
            "training_loss_regression_fraction": (
                metrics["training_loss"] / baseline_metrics["training_loss"] - 1
            ),
        }
        reasons: list[str] = []
        if gains["throughput_gain_fraction"] < thresholds[
            "minimum_throughput_gain_fraction"
        ]:
            reasons.append("median_gain_below_threshold")
        if gains["p10_throughput_gain_fraction"] < thresholds[
            "minimum_p10_throughput_gain_fraction"
        ]:
            reasons.append("p10_gain_below_threshold")
        if gains["training_loss_regression_fraction"] > thresholds[
            "maximum_training_loss_regression_fraction"
        ]:
            reasons.append("training_loss_regression")
        if metrics["peak_memory_bytes"] > thresholds["maximum_peak_memory_bytes"]:
            reasons.append("insufficient_hbm_reserve")
        comparisons[label] = {**gains, "eligible": not reasons, "reasons": reasons}
        if not reasons:
            eligible.append(label)
    winner = (
        max(eligible, key=lambda label: by_label[label]["metrics"]["median_tokens_per_second"])
        if eligible
        else BASELINE_LABEL
    )
    decision: dict[str, Any] = {
        "comparisons_to_baseline": comparisons,
        "exact_segment2_checkpoint_verification_required": True,
        "observed_cost_usd": cost,
        "observed_runtime_minutes": runtime_minutes,
        "production_training_authorized": False,
        "reason": (
            "provisional_material_safe_throughput_gain"
            if winner != BASELINE_LABEL
            else "no_candidate_cleared_fail_closed_screen"
        ),
        "results": results,
        "schema_version": SCHEMA_VERSION,
        "screening_only": True,
        "status": "pass",
        "winner": winner,
    }
    decision["decision_sha256"] = _fingerprint(decision)
    return decision


def write_bundle(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any], output: str | Path
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "plan.json", dict(plan))
    _write_json_atomic(output / "authorization.json", dict(authorization))
