"""Fail-closed planning gate for staged Lumi SFT benchmark execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-benchmark-gates.v0.1"
PLAN_STATUS = "benchmark_schedule_frozen_planning_only_no_inference_or_sft_authorized"

EXPECTED_AUTHORIZATION = {
    "data_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "model_inference_authorized": False,
    "paid_gpu_authorized": False,
    "production_release_authorized": False,
    "production_sft_authorized": False,
    "purpose": "freeze_quality_first_sft_benchmark_schedule_only",
}

EXPECTED_GATE_IDS = [
    "base_full_baseline",
    "sealed_dataset_runtime_smoke",
    "learning_candidate_10m",
    "first_release_candidate_50m",
    "quality_checkpoint_100m",
    "quality_checkpoint_250m",
    "quality_ceiling_500m",
    "selected_release_final",
]

EXPECTED_STAGE_TOKENS = [
    0,
    10_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    None,
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_binding(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("benchmark binding path is missing")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"benchmark binding SHA-256 is invalid: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"benchmark binding escapes repository: {relative}")
    if not resolved.is_file() or _file_sha256(resolved) != expected_sha256:
        raise ValueError(f"benchmark binding changed: {relative}")
    return resolved


def load_benchmark_gate_plan(
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Load and verify the non-authorizing benchmark schedule and its bindings."""

    root = repository_root.resolve()
    plan = _read_object(path.resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark-gate plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("benchmark-gate status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("benchmark-gate authorization boundary changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("benchmark-gate bindings are missing")
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise ValueError("benchmark-gate binding is invalid")
        _resolve_binding(root, binding)

    gates = plan.get("gates")
    if not isinstance(gates, list):
        raise ValueError("benchmark gates are missing")
    if [gate.get("gate_id") for gate in gates] != EXPECTED_GATE_IDS:
        raise ValueError("benchmark gate order changed")
    if [gate.get("accepted_train_supervised_tokens") for gate in gates] != (
        EXPECTED_STAGE_TOKENS
    ):
        raise ValueError("benchmark supervised-token stages changed")
    if gates[1].get("kind") == "full_quality_benchmark":
        raise ValueError("runtime smoke must not be classified as a quality benchmark")

    policy = plan.get("decision_policy")
    if not isinstance(policy, dict):
        raise ValueError("benchmark decision policy is missing")
    if policy.get("five_hundred_million_is_a_quality_ceiling_not_a_mandatory_dose") is not True:
        raise ValueError("500M quality-ceiling rule changed")
    if policy.get("runtime_smoke_is_a_quality_benchmark") is not False:
        raise ValueError("runtime-smoke classification changed")
    if policy.get("benchmark_prompts_must_never_enter_training_data") is not True:
        raise ValueError("benchmark contamination boundary changed")
    return plan


def build_benchmark_gate_readiness_report(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Summarize what can run now without retaining prompt or response text."""

    root = repository_root.resolve()
    bindings = plan["bindings"]
    evaluation = _read_object(_resolve_binding(root, bindings["evaluation_plan"]))
    candidate = _read_object(_resolve_binding(root, bindings["candidate_mix_reaudit"]))
    candidate_verification = _read_object(
        _resolve_binding(root, bindings["candidate_mix_reaudit_verification"])
    )
    base = _read_object(_resolve_binding(root, bindings["base_checkpoint"]))

    adapters = evaluation.get("benchmark_adapters")
    if not isinstance(adapters, list):
        raise ValueError("evaluation adapters are missing")
    adapter_build_pending = sorted(
        str(adapter["id"])
        for adapter in adapters
        if adapter.get("status") == "adapter_pending"
    )
    execution_pending = sorted(
        str(adapter["id"])
        for adapter in adapters
        if "execution_pending" in str(adapter.get("status", ""))
    )

    pool = candidate.get("candidate_pool")
    if not isinstance(pool, dict):
        raise ValueError("candidate-pool metrics are missing")
    accepted_tokens = pool.get("accepted_train_supervised_tokens")
    gap = pool.get("gap_to_10m_train_supervised_tokens")
    if not isinstance(accepted_tokens, int) or not isinstance(gap, int):
        raise ValueError("candidate-pool token metrics are invalid")
    if candidate_verification.get("production_eligible") is not False:
        raise ValueError("candidate verification unexpectedly became production eligible")
    if base.get("checkpoint", {}).get("training_tokens") != 80_000_000_000:
        raise ValueError("base checkpoint is not bound to exactly 80B training tokens")

    report: dict[str, Any] = {
        "adapter_build_pending": adapter_build_pending,
        "base_checkpoint": {
            "manifest_sha256": base["checkpoint"]["manifest_sha256"],
            "training_tokens": base["checkpoint"]["training_tokens"],
        },
        "base_full_baseline_ready": not adapter_build_pending,
        "candidate_pool": {
            "accepted_train_supervised_tokens": accepted_tokens,
            "gap_to_10m_train_supervised_tokens": gap,
            "production_eligible": candidate.get("production_eligible"),
        },
        "current_next_action": (
            "finish_frozen_benchmark_adapters_and_continue_rights_clear_candidate_mix"
        ),
        "execution_pending_after_checkpoint_exists": execution_pending,
        "five_hundred_million_mandatory": False,
        "full_gpu_sft_quality_gate_ready": False,
        "gates": [
            {
                "accepted_train_supervised_tokens": gate[
                    "accepted_train_supervised_tokens"
                ],
                "gate_id": gate["gate_id"],
                "kind": gate["kind"],
            }
            for gate in plan["gates"]
        ],
        "model_inference_executed": False,
        "observed_at": plan["observed_at"],
        "paid_gpu_used": False,
        "raw_prompt_or_response_text_retained": False,
        "runtime_smoke_is_quality_benchmark": False,
        "schema_version": "lumi.sft-benchmark-gates-readiness.v0.1",
        "status": "planning_verified_benchmark_adapters_and_dataset_mix_incomplete",
    }
    report["report_sha256"] = _value_sha256(report)
    return report


def verify_benchmark_gate_readiness_report(report: Mapping[str, Any]) -> None:
    """Verify the deterministic, raw-text-free planning report."""

    if report.get("schema_version") != "lumi.sft-benchmark-gates-readiness.v0.1":
        raise ValueError("unsupported benchmark readiness report")
    unsigned = dict(report)
    claimed = unsigned.pop("report_sha256", None)
    if not isinstance(claimed, str) or claimed != _value_sha256(unsigned):
        raise ValueError("benchmark readiness report seal mismatch")
    if report.get("raw_prompt_or_response_text_retained") is not False:
        raise ValueError("benchmark readiness report retained raw text")
    if report.get("paid_gpu_used") is not False:
        raise ValueError("benchmark readiness report unexpectedly used paid GPU")
    if report.get("model_inference_executed") is not False:
        raise ValueError("benchmark readiness report unexpectedly ran inference")
    if report.get("five_hundred_million_mandatory") is not False:
        raise ValueError("benchmark readiness report made 500M mandatory")
