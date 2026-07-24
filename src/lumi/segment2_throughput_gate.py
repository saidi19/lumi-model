"""Fail-closed decision logic for Lumi-4B Segment-2 throughput tuning.

This module evaluates sealed benchmark evidence only. It cannot authorize or
start paid GPU work, migrate a production checkpoint, or launch Segment 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SEGMENT2_THROUGHPUT_GATE_SCHEMA_VERSION = 1
SEGMENT2_THROUGHPUT_GATE = "lumi_4b_segment2_throughput_gate_v0.1"
BASELINE_LABEL = "baseline_b1_gc"
REQUIRED_PROFILE_LABELS = (
    BASELINE_LABEL,
    "batch2_gc",
    "batch1_no_gc",
    "batch2_no_gc",
    "batch2_no_gc_compile",
)
TERMINAL_CANDIDATE_STATUSES = {"pass", "oom", "compile_error"}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(value: Mapping[str, Any], omitted: str | None = None) -> str:
    material = dict(value)
    if omitted is not None:
        material.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _finite_positive(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _non_negative_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _hex_digest(value: object, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase {length}-character digest")
    return value


def _validate_profile(profile: Mapping[str, Any], *, world_size: int) -> dict[str, Any]:
    expected_fields = {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "label",
        "per_rank_batch_size",
        "precision",
        "strategy",
        "torch_compile",
    }
    if set(profile) != expected_fields:
        raise ValueError("throughput profile fields changed")
    label = profile.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("throughput profile label is invalid")
    per_rank_batch_size = profile.get("per_rank_batch_size")
    accumulation = profile.get("gradient_accumulation_steps")
    if (
        not isinstance(per_rank_batch_size, int)
        or isinstance(per_rank_batch_size, bool)
        or per_rank_batch_size < 1
        or not isinstance(accumulation, int)
        or isinstance(accumulation, bool)
        or accumulation < 1
    ):
        raise ValueError(f"throughput profile {label} has invalid batch geometry")
    if profile.get("strategy") != "fsdp_full_shard":
        raise ValueError(f"throughput profile {label} changed checkpoint sharding")
    if profile.get("precision") != "bf16":
        raise ValueError(f"throughput profile {label} changed training precision")
    if not isinstance(profile.get("gradient_checkpointing"), bool):
        raise ValueError(f"throughput profile {label} has invalid checkpointing flag")
    if not isinstance(profile.get("torch_compile"), bool):
        raise ValueError(f"throughput profile {label} has invalid compile flag")
    if world_size * per_rank_batch_size * accumulation != 256:
        raise ValueError(f"throughput profile {label} changed the global batch")
    return dict(profile)


def validate_gate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable planning-only A/B contract."""

    if (
        contract.get("schema_version") != SEGMENT2_THROUGHPUT_GATE_SCHEMA_VERSION
        or contract.get("status") != "planning_only"
    ):
        raise ValueError("unsupported Segment-2 throughput gate contract")
    if contract.get("approval") != {
        "paid_gpu": False,
        "production_training": False,
    }:
        raise ValueError("Segment-2 throughput contract cannot authorize paid work")

    source = contract.get("source", {})
    if source.get("checkpoint_format") != "dcp-model_rank-local-optimizer-v0.1":
        raise ValueError("Segment-2 source checkpoint format changed")
    _hex_digest(source.get("run_fingerprint"), "Segment-1 run fingerprint")
    _hex_digest(source.get("corpus_manifest_sha256"), "corpus manifest SHA-256")
    _hex_digest(source.get("model_config_sha256"), "model config SHA-256")
    _hex_digest(source.get("source_commit"), "Segment-1 source commit", length=40)
    if source.get("segment_1_stop_step") != 31_500:
        raise ValueError("Segment-1 stop step changed")
    if source.get("segment_1_stop_tokens") != 33_030_144_000:
        raise ValueError("Segment-1 stop token cursor changed")

    hardware = contract.get("hardware", {})
    if hardware != {"gpu_model_contains": "H100", "world_size": 8}:
        raise ValueError("Segment-2 throughput hardware changed")
    benchmark = contract.get("benchmark", {})
    if benchmark != {
        "global_sequences_per_step": 256,
        "measured_steps_per_candidate": 24,
        "sequence_length": 4096,
        "warmup_steps_per_candidate": 8,
    }:
        raise ValueError("Segment-2 benchmark geometry changed")

    budget = contract.get("budget", {})
    if budget != {
        "hourly_rate_usd": 31.92,
        "maximum_cost_usd": 63.84,
        "maximum_runtime_minutes": 120,
    }:
        raise ValueError("Segment-2 throughput budget changed")
    if budget["hourly_rate_usd"] * budget["maximum_runtime_minutes"] / 60 > budget[
        "maximum_cost_usd"
    ]:
        raise ValueError("Segment-2 throughput runtime exceeds its cost cap")

    decision = contract.get("decision", {})
    if decision != {
        "maximum_heldout_loss_regression_fraction": 0.005,
        "maximum_peak_memory_bytes": 76_000_000_000,
        "maximum_training_loss_regression_fraction": 0.01,
        "minimum_baseline_p10_tokens_per_second": 65_000,
        "minimum_p10_throughput_gain_fraction": 0.1,
        "minimum_throughput_gain_fraction": 0.15,
    }:
        raise ValueError("Segment-2 throughput thresholds changed")

    profiles = contract.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Segment-2 throughput profiles are missing")
    validated_profiles = [
        _validate_profile(profile, world_size=hardware["world_size"]) for profile in profiles
    ]
    labels = tuple(profile["label"] for profile in validated_profiles)
    if labels != REQUIRED_PROFILE_LABELS:
        raise ValueError("Segment-2 throughput profile matrix changed")
    expected_baseline = {
        "gradient_accumulation_steps": 32,
        "gradient_checkpointing": True,
        "label": BASELINE_LABEL,
        "per_rank_batch_size": 1,
        "precision": "bf16",
        "strategy": "fsdp_full_shard",
        "torch_compile": False,
    }
    if validated_profiles[0] != expected_baseline:
        raise ValueError("Segment-2 throughput baseline changed")
    return dict(contract)


def _validate_source_binding(
    *,
    source: Mapping[str, Any],
    observed: Mapping[str, Any],
    expected_checkpoint_sha256: str | None,
) -> str:
    exact_fields = (
        "run_fingerprint",
        "source_commit",
        "step",
        "tokens",
    )
    expected = {
        "run_fingerprint": source["run_fingerprint"],
        "source_commit": source["source_commit"],
        "step": source["segment_1_stop_step"],
        "tokens": source["segment_1_stop_tokens"],
    }
    for field in exact_fields:
        if observed.get(field) != expected[field]:
            raise ValueError(f"benchmark source changed {field}")
    checkpoint_sha256 = _hex_digest(
        observed.get("checkpoint_manifest_sha256"),
        "benchmark checkpoint manifest SHA-256",
    )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("candidate benchmarks did not use one checkpoint")
    return checkpoint_sha256


def _validate_passed_candidate(
    *,
    result: Mapping[str, Any],
    profile: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, float]:
    if result.get("profile") != profile:
        raise ValueError(f"candidate {profile['label']} execution profile changed")
    if result.get("checkpoint_restore_pass") is not True:
        raise ValueError(f"candidate {profile['label']} did not restore the checkpoint")
    if result.get("document_isolation_pass") is not True:
        raise ValueError(f"candidate {profile['label']} lost document isolation")
    if result.get("resume_canary_pass") is not True:
        raise ValueError(f"candidate {profile['label']} did not pass checkpoint/resume")
    if result.get("non_finite_count") != 0:
        raise ValueError(f"candidate {profile['label']} reported non-finite metrics")
    benchmark = contract["benchmark"]
    if result.get("warmup_steps") != benchmark["warmup_steps_per_candidate"]:
        raise ValueError(f"candidate {profile['label']} warmup changed")
    if result.get("measured_steps") != benchmark["measured_steps_per_candidate"]:
        raise ValueError(f"candidate {profile['label']} measured-step count changed")

    metrics = result.get("metrics", {})
    return {
        "heldout_loss": _finite_positive(
            metrics.get("heldout_loss"), f"{profile['label']} held-out loss"
        ),
        "median_tokens_per_second": _finite_positive(
            metrics.get("median_tokens_per_second"),
            f"{profile['label']} median throughput",
        ),
        "p10_tokens_per_second": _finite_positive(
            metrics.get("p10_tokens_per_second"),
            f"{profile['label']} p10 throughput",
        ),
        "peak_memory_bytes": _finite_positive(
            metrics.get("peak_memory_bytes"),
            f"{profile['label']} peak memory",
        ),
        "training_loss": _finite_positive(
            metrics.get("training_loss"), f"{profile['label']} training loss"
        ),
    }


def evaluate_segment2_throughput_gate(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose only a materially faster candidate with intact learning semantics."""

    validated = validate_gate_contract(contract)
    if evidence.get("schema_version") != SEGMENT2_THROUGHPUT_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported Segment-2 throughput evidence")
    if evidence.get("status") != "complete":
        raise ValueError("Segment-2 throughput evidence is incomplete")
    runtime = _finite_positive(
        evidence.get("observed_runtime_minutes"), "observed benchmark runtime"
    )
    cost = _non_negative_number(
        evidence.get("observed_cost_usd"), "observed benchmark cost"
    )
    if runtime > validated["budget"]["maximum_runtime_minutes"]:
        raise ValueError("Segment-2 throughput benchmark exceeded its runtime cap")
    if cost > validated["budget"]["maximum_cost_usd"]:
        raise ValueError("Segment-2 throughput benchmark exceeded its cost cap")

    profiles = {profile["label"]: profile for profile in validated["profiles"]}
    results = evidence.get("results")
    if not isinstance(results, list) or len(results) != len(profiles):
        raise ValueError("Segment-2 evidence requires every candidate")
    by_label: dict[str, Mapping[str, Any]] = {}
    checkpoint_sha256: str | None = None
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("Segment-2 candidate evidence is not an object")
        label = result.get("label")
        if label not in profiles or label in by_label:
            raise ValueError("Segment-2 candidate labels are missing or duplicated")
        checkpoint_sha256 = _validate_source_binding(
            source=validated["source"],
            observed=result.get("source_checkpoint", {}),
            expected_checkpoint_sha256=checkpoint_sha256,
        )
        status = result.get("status")
        if status not in TERMINAL_CANDIDATE_STATUSES:
            raise ValueError(f"candidate {label} has a non-terminal status")
        if label == BASELINE_LABEL and status != "pass":
            raise ValueError("Segment-2 throughput baseline did not pass")
        if status != "pass":
            if not str(result.get("error", "")).strip():
                raise ValueError(f"failed candidate {label} has no preserved error")
            if result.get("profile") != profiles[label]:
                raise ValueError(f"failed candidate {label} profile changed")
        by_label[label] = result
    if set(by_label) != set(profiles):
        raise ValueError("Segment-2 evidence did not cover the exact profile matrix")

    summaries: dict[str, dict[str, Any]] = {}
    for label in REQUIRED_PROFILE_LABELS:
        result = by_label[label]
        if result["status"] == "pass":
            summaries[label] = {
                "metrics": _validate_passed_candidate(
                    result=result,
                    profile=profiles[label],
                    contract=validated,
                ),
                "status": "pass",
            }
        else:
            summaries[label] = {
                "error": str(result["error"]),
                "status": result["status"],
            }

    baseline = summaries[BASELINE_LABEL]["metrics"]
    thresholds = validated["decision"]
    if (
        baseline["p10_tokens_per_second"]
        < thresholds["minimum_baseline_p10_tokens_per_second"]
    ):
        raise ValueError("Segment-2 baseline throughput is below the safe production floor")
    if baseline["peak_memory_bytes"] > thresholds["maximum_peak_memory_bytes"]:
        raise ValueError("Segment-2 baseline has insufficient HBM reserve")

    comparisons: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for label in REQUIRED_PROFILE_LABELS[1:]:
        summary = summaries[label]
        if summary["status"] != "pass":
            comparisons[label] = {
                "eligible": False,
                "reason": summary["status"],
            }
            continue
        metrics = summary["metrics"]
        throughput_gain = (
            metrics["median_tokens_per_second"]
            / baseline["median_tokens_per_second"]
            - 1.0
        )
        p10_gain = (
            metrics["p10_tokens_per_second"] / baseline["p10_tokens_per_second"]
            - 1.0
        )
        training_loss_regression = (
            metrics["training_loss"] / baseline["training_loss"] - 1.0
        )
        heldout_loss_regression = (
            metrics["heldout_loss"] / baseline["heldout_loss"] - 1.0
        )
        reasons: list[str] = []
        if throughput_gain < thresholds["minimum_throughput_gain_fraction"]:
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
        if metrics["peak_memory_bytes"] > thresholds["maximum_peak_memory_bytes"]:
            reasons.append("insufficient_hbm_reserve")
        comparisons[label] = {
            "eligible": not reasons,
            "heldout_loss_regression_fraction": heldout_loss_regression,
            "p10_throughput_gain_fraction": p10_gain,
            "reasons": reasons,
            "throughput_gain_fraction": throughput_gain,
            "training_loss_regression_fraction": training_loss_regression,
        }
        if not reasons:
            eligible.append(label)

    if eligible:
        winner = max(
            eligible,
            key=lambda label: summaries[label]["metrics"]["median_tokens_per_second"],
        )
        reason = "material_safe_throughput_gain"
    else:
        winner = BASELINE_LABEL
        reason = "no_candidate_cleared_fail_closed_thresholds"

    baseline_rate = baseline["median_tokens_per_second"]
    selected_rate = summaries[winner]["metrics"]["median_tokens_per_second"]
    segment_2_tokens = 33_030_144_000
    remaining_tokens = 80_000_000_000 - validated["source"]["segment_1_stop_tokens"]
    hourly_rate = validated["budget"]["hourly_rate_usd"]
    decision = {
        "baseline_projected_remaining_cost_usd": (
            remaining_tokens / baseline_rate / 3600 * hourly_rate
        ),
        "baseline_projected_segment_2_cost_usd": (
            segment_2_tokens / baseline_rate / 3600 * hourly_rate
        ),
        "benchmark_cost_usd": cost,
        "benchmark_runtime_minutes": runtime,
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "comparisons_to_baseline": comparisons,
        "contract_sha256": _fingerprint(validated),
        "gate": SEGMENT2_THROUGHPUT_GATE,
        "reason": reason,
        "schema_version": SEGMENT2_THROUGHPUT_GATE_SCHEMA_VERSION,
        "selected_profile": profiles[winner],
        "selected_projected_remaining_cost_usd": (
            remaining_tokens / selected_rate / 3600 * hourly_rate
        ),
        "selected_projected_segment_2_cost_usd": (
            segment_2_tokens / selected_rate / 3600 * hourly_rate
        ),
        "source": validated["source"],
        "status": "pass",
        "summaries": summaries,
        "winner": winner,
    }
    decision["decision_sha256"] = _fingerprint(decision)
    return decision


def validate_gate_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized decision before a continuation plan consumes it."""

    if (
        decision.get("schema_version") != SEGMENT2_THROUGHPUT_GATE_SCHEMA_VERSION
        or decision.get("gate") != SEGMENT2_THROUGHPUT_GATE
        or decision.get("status") != "pass"
    ):
        raise ValueError("unsupported Segment-2 throughput decision")
    if decision.get("decision_sha256") != _fingerprint(
        decision, "decision_sha256"
    ):
        raise ValueError("Segment-2 throughput decision hash changed")
    if decision.get("winner") != decision.get("selected_profile", {}).get("label"):
        raise ValueError("Segment-2 throughput winner and selected profile differ")
    _hex_digest(
        decision.get("checkpoint_manifest_sha256"),
        "Segment-2 decision checkpoint manifest SHA-256",
    )
    return dict(decision)


def evaluate_segment2_throughput_files(
    *,
    contract_path: str | Path,
    evidence_path: str | Path,
) -> dict[str, Any]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    return evaluate_segment2_throughput_gate(contract, evidence)
