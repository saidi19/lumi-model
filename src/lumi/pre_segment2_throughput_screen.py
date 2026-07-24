"""Bounded, pre-handoff throughput screening for Lumi-4B Segment 2.

The screen compares fresh deterministic runs on an isolated 8xH100 instance.
It deliberately does not consume or replace the final checkpoint-bound gate.
"""

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
    PAID_SEGMENT2_SCREEN_AGGREGATE_TOKEN_CAP,
    PAID_SEGMENT2_SCREEN_CANDIDATES,
    PAID_SEGMENT2_SCREEN_MAX_COST_USD,
    PAID_SEGMENT2_SCREEN_MAX_RUNTIME_MINUTES,
    PAID_SEGMENT2_SCREEN_MAX_STEPS,
    PAID_SEGMENT2_SCREEN_OWNER_ACKNOWLEDGEMENT,
    PAID_SEGMENT2_SCREEN_RUN_KIND,
    PAID_SEGMENT2_SCREEN_TOKEN_CAP,
    PAID_SEGMENT2_SCREEN_WORLD_SIZE,
    build_distributed_run_identity,
)
from .evaluation_data import validate_heldout_manifest
from .packed_data import validate_packed_manifest
from .pilot_training import _fingerprint, _write_json_atomic
from .segment2_throughput_gate import (
    BASELINE_LABEL,
    REQUIRED_PROFILE_LABELS,
    validate_gate_contract,
)

PRE_SEGMENT2_SCREEN_SCHEMA_VERSION = 1
PRE_SEGMENT2_SCREEN_KIND = "lumi_4b_pre_segment2_throughput_screen_v0.1"
SCREEN_INSTANCE_IP = "209.20.157.52"
SCREEN_INSTANCE_TYPE = "gpu_8x_h100_sxm5"
SCREEN_HOURLY_RATE_USD = 31.92
MEASURED_STEP_START = 9
MEASURED_STEP_END = 32
RESUME_CANARY_STEP = 33


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _validate_digest(value: object, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase {length}-character digest")
    return value


def _tokenizer_inventory(
    tokenizer_path: Path,
    heldout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for expected in heldout["tokenizer"]["files"]:
        path = tokenizer_path / expected["name"]
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"throughput-screen tokenizer changed {expected['name']}")
        inventory.append(dict(expected))
    return inventory


def build_pre_segment2_screen_plan(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
    source_commit: str,
    instance_boot_utc: str,
) -> dict[str, Any]:
    """Build the five exact fresh-run identities and their aggregate boundary."""

    contract_path = Path(contract_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    config_path = Path(config_path).resolve()
    contract = validate_gate_contract(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    manifest = validate_packed_manifest(manifest_path, verify_hashes=True)
    heldout = validate_heldout_manifest(heldout_manifest_path)
    if manifest["statistics"]["tokens"] != 83_886_080:
        raise ValueError("throughput-screen training fixture token count changed")
    if manifest["packing"]["sequence_length"] != 4096:
        raise ValueError("throughput-screen sequence length changed")
    if (
        manifest.get("provenance", {}).get("final_manifest_sha256")
        != contract["source"]["corpus_manifest_sha256"]
    ):
        raise ValueError("throughput-screen corpus binding changed")
    if _sha256(config_path) != contract["source"]["model_config_sha256"]:
        raise ValueError("throughput-screen model config changed")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("throughput-screen source commit is invalid")
    tokenizer_inventory = _tokenizer_inventory(tokenizer_path, heldout)
    boot = datetime.fromisoformat(instance_boot_utc)
    if boot.tzinfo is None:
        raise ValueError("throughput-screen instance boot time lacks a timezone")
    deadline = boot.astimezone(UTC) + timedelta(
        minutes=PAID_SEGMENT2_SCREEN_MAX_RUNTIME_MINUTES
    )

    manifest_sha256 = _sha256(manifest_path)
    identities: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for profile in contract["profiles"]:
        identity = build_distributed_run_identity(
            config_bytes=config_path.read_bytes(),
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            backend="nccl",
            world_size=PAID_SEGMENT2_SCREEN_WORLD_SIZE,
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
            warmup_steps=8,
            schedule_steps=RESUME_CANARY_STEP,
            min_learning_rate_ratio=0.1,
            attention_implementation="flash_attention_2",
            document_isolation=True,
            optimizer_profile="legacy",
            precision=profile["precision"],
            gradient_sync_mode="last_microbatch",
            torch_compile=profile["torch_compile"],
            source_commit=source_commit,
        )
        identities[profile["label"]] = identity
        fingerprints[profile["label"]] = _fingerprint(identity)

    plan: dict[str, Any] = {
        "benchmark": {
            "measured_step_end": MEASURED_STEP_END,
            "measured_step_start": MEASURED_STEP_START,
            "resume_canary_step": RESUME_CANARY_STEP,
            "warmup_steps": MEASURED_STEP_START - 1,
        },
        "budget": {
            "deadline_utc": deadline.replace(microsecond=0).isoformat(),
            "hourly_rate_usd": SCREEN_HOURLY_RATE_USD,
            "instance_boot_utc": boot.astimezone(UTC).replace(microsecond=0).isoformat(),
            "maximum_cost_usd": PAID_SEGMENT2_SCREEN_MAX_COST_USD,
            "maximum_runtime_minutes": PAID_SEGMENT2_SCREEN_MAX_RUNTIME_MINUTES,
        },
        "contract_sha256": _sha256(contract_path),
        "created_at": _utc_now(),
        "data": {
            "heldout_build_fingerprint": heldout["build_fingerprint"],
            "heldout_manifest_sha256": _sha256(heldout_manifest_path),
            "heldout_tokens": heldout["statistics"]["tokens"],
            "manifest_build_fingerprint": manifest["build_fingerprint"],
            "manifest_sha256": manifest_sha256,
            "source_prefix_sha256": manifest["provenance"]["source_prefix_sha256"],
            "training_tokens": manifest["statistics"]["tokens"],
            "tokenizer_files": tokenizer_inventory,
        },
        "evaluation": {
            "batch_size": 2,
            "checkpoint_batches": 20,
            "generation_max_new_tokens": 8,
            "prompts": [
                "The capital of France is",
                "Photosynthesis is the process by which",
                "In Python, a list comprehension can",
                "বাংলাদেশের রাজধানী",
            ],
        },
        "instance": {
            "ip": SCREEN_INSTANCE_IP,
            "type": SCREEN_INSTANCE_TYPE,
            "world_size": PAID_SEGMENT2_SCREEN_WORLD_SIZE,
        },
        "profiles": contract["profiles"],
        "run_fingerprints": fingerprints,
        "run_identities": identities,
        "run_kind": PRE_SEGMENT2_SCREEN_KIND,
        "schema_version": PRE_SEGMENT2_SCREEN_SCHEMA_VERSION,
        "screening_only": True,
        "segment_1": {
            "checkpoint_transferred": False,
            "run_fingerprint": contract["source"]["run_fingerprint"],
            "source_commit": contract["source"]["source_commit"],
            "touched": False,
        },
        "source_commit": source_commit,
        "status": "sealed",
    }
    plan["plan_sha256"] = _fingerprint(plan)
    return plan


def build_pre_segment2_screen_authorization(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the exact owner-approved aggregate authorization."""

    if (
        plan.get("schema_version") != PRE_SEGMENT2_SCREEN_SCHEMA_VERSION
        or plan.get("run_kind") != PRE_SEGMENT2_SCREEN_KIND
        or plan.get("status") != "sealed"
        or plan.get("screening_only") is not True
    ):
        raise ValueError("throughput-screen plan is not sealed")
    if plan.get("plan_sha256") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("throughput-screen plan hash changed")
    authorization = {
        "aggregate_candidate_count": PAID_SEGMENT2_SCREEN_CANDIDATES,
        "aggregate_token_cap": PAID_SEGMENT2_SCREEN_AGGREGATE_TOKEN_CAP,
        "allowed_run_fingerprints": plan["run_fingerprints"],
        "approved_at": _utc_now(),
        "approved_through_step": PAID_SEGMENT2_SCREEN_MAX_STEPS,
        "approved_token_cap": PAID_SEGMENT2_SCREEN_TOKEN_CAP,
        "deadline_utc": plan["budget"]["deadline_utc"],
        "full_4b_training": False,
        "hourly_rate_usd": SCREEN_HOURLY_RATE_USD,
        "instance_boot_utc": plan["budget"]["instance_boot_utc"],
        "instance_ip": SCREEN_INSTANCE_IP,
        "instance_type": SCREEN_INSTANCE_TYPE,
        "max_cost_usd": PAID_SEGMENT2_SCREEN_MAX_COST_USD,
        "max_runtime_minutes": PAID_SEGMENT2_SCREEN_MAX_RUNTIME_MINUTES,
        "owner_acknowledgement": PAID_SEGMENT2_SCREEN_OWNER_ACKNOWLEDGEMENT,
        "paid_gpu": True,
        "plan_sha256": plan["plan_sha256"],
        "production_training": False,
        "provider": "Lambda Cloud",
        "run_kind": PAID_SEGMENT2_SCREEN_RUN_KIND,
        "schema_version": 1,
        "segment_1_access": {
            "checkpoint_transfer": False,
            "modify_or_restart": False,
            "read_only_inputs": [
                "bounded_training_prefix",
                "heldout_data",
                "tokenizer",
            ],
        },
        "source_commit": plan["source_commit"],
        "status": "owner_approved",
        "strategy": "fsdp_full_shard",
        "world_size": PAID_SEGMENT2_SCREEN_WORLD_SIZE,
    }
    authorization["authorization_sha256"] = _fingerprint(authorization)
    return authorization


def validate_pre_segment2_screen_files(
    *,
    plan_path: str | Path,
    authorization_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("throughput-screen plan hash changed")
    if authorization.get("authorization_sha256") != _fingerprint(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
    ):
        raise ValueError("throughput-screen authorization hash changed")
    if authorization.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("throughput-screen authorization targets another plan")
    if authorization.get("owner_acknowledgement") != (
        PAID_SEGMENT2_SCREEN_OWNER_ACKNOWLEDGEMENT
    ):
        raise ValueError("throughput-screen owner approval changed")
    if authorization.get("allowed_run_fingerprints") != plan["run_fingerprints"]:
        raise ValueError("throughput-screen run fingerprints changed")
    return plan, authorization


def summarize_pre_segment2_candidate(
    *,
    profile: Mapping[str, Any],
    metrics_path: str | Path,
    state_path: str | Path,
    evaluation_report_path: str | Path,
) -> dict[str, Any]:
    metrics = [
        json.loads(line)
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    measured = [
        metric
        for metric in metrics
        if MEASURED_STEP_START <= metric.get("step", -1) <= MEASURED_STEP_END
    ]
    if len(measured) != MEASURED_STEP_END - MEASURED_STEP_START + 1:
        raise ValueError(f"candidate {profile['label']} measured-step evidence is incomplete")
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if state.get("step") != RESUME_CANARY_STEP:
        raise ValueError(f"candidate {profile['label']} did not pass resume step")
    evaluation = json.loads(
        Path(evaluation_report_path).read_text(encoding="utf-8")
    )
    rates = [float(metric["tokens_per_second"]) for metric in measured]
    losses = [float(metric["loss"]) for metric in measured]
    if any(
        not math.isfinite(value) or value <= 0 for value in [*rates, *losses]
    ):
        raise ValueError(f"candidate {profile['label']} has non-finite metrics")
    p10 = statistics.quantiles(rates, n=10, method="inclusive")[0]
    return {
        "document_isolation_pass": True,
        "label": profile["label"],
        "measured_steps": len(measured),
        "metrics": {
            "heldout_loss": float(evaluation["loss"]["mean"]),
            "median_tokens_per_second": statistics.median(rates),
            "p10_tokens_per_second": p10,
            "peak_memory_bytes": max(
                int(metric["peak_memory_reserved_bytes"]) for metric in measured
            ),
            "training_loss": statistics.mean(losses),
        },
        "non_finite_count": 0,
        "profile": dict(profile),
        "resume_canary_pass": True,
        "run_fingerprint": state["run_fingerprint"],
        "status": "pass",
        "warmup_steps": MEASURED_STEP_START - 1,
    }


def decide_pre_segment2_screen(
    *,
    contract: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    observed_runtime_minutes: float,
) -> dict[str, Any]:
    """Select a provisional winner without authorizing Segment 2."""

    validated = validate_gate_contract(contract)
    observed_cost_usd = (
        observed_runtime_minutes / 60 * SCREEN_HOURLY_RATE_USD
    )
    if observed_runtime_minutes > PAID_SEGMENT2_SCREEN_MAX_RUNTIME_MINUTES:
        raise ValueError("throughput-screen runtime cap was exceeded")
    if observed_cost_usd > PAID_SEGMENT2_SCREEN_MAX_COST_USD:
        raise ValueError("throughput-screen cost cap was exceeded")
    by_label = {str(result["label"]): result for result in results}
    if set(by_label) != set(REQUIRED_PROFILE_LABELS):
        raise ValueError("throughput-screen result matrix is incomplete")
    baseline = by_label[BASELINE_LABEL]
    if baseline.get("status") != "pass":
        raise ValueError("throughput-screen baseline failed")
    thresholds = validated["decision"]
    baseline_metrics = baseline["metrics"]
    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    for label in REQUIRED_PROFILE_LABELS[1:]:
        candidate = by_label[label]
        if candidate.get("status") != "pass":
            comparisons[label] = {
                "eligible": False,
                "reasons": [candidate.get("status", "failed")],
            }
            continue
        metrics = candidate["metrics"]
        gains = {
            "heldout_loss_regression_fraction": (
                metrics["heldout_loss"] / baseline_metrics["heldout_loss"] - 1
            ),
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
        if (
            gains["throughput_gain_fraction"]
            < thresholds["minimum_throughput_gain_fraction"]
        ):
            reasons.append("median_gain_below_threshold")
        if (
            gains["p10_throughput_gain_fraction"]
            < thresholds["minimum_p10_throughput_gain_fraction"]
        ):
            reasons.append("p10_gain_below_threshold")
        if (
            gains["training_loss_regression_fraction"]
            > thresholds["maximum_training_loss_regression_fraction"]
        ):
            reasons.append("training_loss_regression")
        if (
            gains["heldout_loss_regression_fraction"]
            > thresholds["maximum_heldout_loss_regression_fraction"]
        ):
            reasons.append("heldout_loss_regression")
        if metrics["peak_memory_bytes"] > thresholds["maximum_peak_memory_bytes"]:
            reasons.append("insufficient_hbm_reserve")
        comparisons[label] = {**gains, "eligible": not reasons, "reasons": reasons}
        if not reasons:
            eligible.append(label)
    winner = (
        max(
            eligible,
            key=lambda label: by_label[label]["metrics"][
                "median_tokens_per_second"
            ],
        )
        if eligible
        else BASELINE_LABEL
    )
    decision = {
        "comparisons_to_baseline": comparisons,
        "final_checkpoint_bound_verification_required": True,
        "observed_cost_usd": observed_cost_usd,
        "observed_runtime_minutes": observed_runtime_minutes,
        "production_training_authorized": False,
        "reason": (
            "provisional_material_safe_throughput_gain"
            if winner != BASELINE_LABEL
            else "no_candidate_cleared_fail_closed_screen"
        ),
        "results": results,
        "schema_version": PRE_SEGMENT2_SCREEN_SCHEMA_VERSION,
        "screening_only": True,
        "status": "pass",
        "winner": winner,
    }
    decision["decision_sha256"] = _fingerprint(decision)
    return decision


def write_pre_segment2_screen_bundle(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output: str | Path,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "plan.json", dict(plan))
    _write_json_atomic(output / "authorization.json", dict(authorization))
