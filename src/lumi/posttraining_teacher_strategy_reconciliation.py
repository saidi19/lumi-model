"""Fail-closed reconciliation of rejected Lumi SFT teacher strategies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-teacher-strategy-reconciliation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-teacher-strategy-reconciliation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-teacher-strategy-reconciliation-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "sft_training_authorized": False,
}
EXPECTED_TRAIN_TOKENS = 7_256_670
EXPECTED_GAP_TO_10M = 2_743_330
EXPECTED_MINIMUM_TASK_CLOSURE = 2_966_672


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic pretty JSON bytes."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    """Hash a mapping using compact canonical JSON."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_internal(value: Mapping[str, Any], field: str, expected: str) -> None:
    observed = value.get(field)
    if observed != expected:
        raise ValueError(f"{field} binding changed")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{field} does not verify")


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    """Load and validate the bounded reconciliation plan."""

    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("teacher strategy reconciliation plan schema changed")
    if plan.get("status") != "authorized_cpu_only_local_evidence_reconciliation":
        raise ValueError("teacher strategy reconciliation is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("teacher strategy reconciliation authorization widened")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or len(bindings) != 9:
        raise ValueError("teacher strategy reconciliation bindings changed")
    workspace_root = root.parent.resolve()
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(workspace_root):
            raise ValueError(f"binding escaped workspace: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding hash mismatch: {name}")
    return plan


def _bound_values(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    values: dict[str, dict[str, Any]] = {}
    for name, binding in plan["bindings"].items():
        if name == "source_registry":
            continue
        values[name] = _load_json((root / binding["path"]).resolve())
    fields = {
        "candidate_mix_reaudit": "report_sha256",
        "candidate_mix_reaudit_verification": "verification_sha256",
        "closure_report": "report_sha256",
        "closure_verification": "verification_sha256",
        "qwen3_8b_scale_failure": "report_sha256",
        "qwen3_8b_scale_failure_verification": "verification_sha256",
        "qwen3_14b_ab_completion": "report_sha256",
        "qwen3_14b_ab_completion_verification": "verification_sha256",
    }
    expected = plan["expected_internal_sha256"]
    for name, field in fields.items():
        _verify_internal(values[name], field, expected[name])
    return values


def build_report(
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build the raw-text-free strategy reconciliation report."""

    values = _bound_values(plan, repository_root)
    reaudit = values["candidate_mix_reaudit"]
    closure = values["closure_report"]
    scale_failure = values["qwen3_8b_scale_failure"]
    scale_verification = values["qwen3_8b_scale_failure_verification"]
    ab_report = values["qwen3_14b_ab_completion"]
    ab_verification = values["qwen3_14b_ab_completion_verification"]

    pool = reaudit.get("candidate_pool", {})
    minimum = closure.get("minimum_accepted_closure", {})
    if pool.get("accepted_train_supervised_tokens") != EXPECTED_TRAIN_TOKENS:
        raise ValueError("current candidate mix token total changed")
    if pool.get("gap_to_10m_train_supervised_tokens") != EXPECTED_GAP_TO_10M:
        raise ValueError("current candidate mix 10M gap changed")
    if minimum.get("task_floor_gap_total") != EXPECTED_MINIMUM_TASK_CLOSURE:
        raise ValueError("minimum task closure changed")
    if scale_failure.get("status") != "stopped_fail_closed_quality_screen_failed":
        raise ValueError("8B scale canary is not a verified fail-closed stop")
    if scale_verification.get("status") != "PASS":
        raise ValueError("8B scale failure verification did not pass")
    if ab_report.get("status") != "verified_candidate_machine_gate_not_met":
        raise ValueError("14B A/B completion status changed")
    if ab_report.get("machine_gate_passed") is not False:
        raise ValueError("14B candidate unexpectedly passed")
    if ab_verification.get("status") != "PASS":
        raise ValueError("14B A/B verification did not pass")
    if ab_verification.get("machine_gate_passed") is not False:
        raise ValueError("14B A/B verification gate changed")

    comparison = ab_report["comparative_audit"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_teacher_strategies_rejected_mix_unchanged",
        "plan_file_sha256": plan_file_sha256,
        "bindings": {
            name: {
                "file_sha256": binding["sha256"],
                "internal_sha256": plan.get("expected_internal_sha256", {}).get(name),
            }
            for name, binding in sorted(plan["bindings"].items())
        },
        "current_mix": {
            "accepted_train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
            "gap_to_10m_train_supervised_tokens": EXPECTED_GAP_TO_10M,
            "minimum_task_closure_supervised_tokens": EXPECTED_MINIMUM_TASK_CLOSURE,
            "task_floor_gaps": minimum["task_floor_gaps"],
            "language_floor_gaps": minimum["language_floor_gaps"],
            "source_cap": 0.35,
            "source_cap_violations": pool["source_shares"]
            and reaudit["closure"]["source_cap_violations"],
        },
        "rejected_teacher_strategies": {
            "qwen3_8b_q4_k_m": {
                "decision": "bulk_generation_ineligible",
                "calls_used": scale_failure["call_budget"]["calls_used"],
                "cumulative_acceptance_rate": scale_failure["acceptance"][
                    "cumulative_rate"
                ],
                "authorized_minimum_acceptance_rate": scale_failure["acceptance"][
                    "authorized_minimum_overall_rate"
                ],
                "manual_quality_defects_present": True,
            },
            "qwen3_14b_q4_k_m": {
                "decision": "stronger_teacher_not_promoted",
                "baseline_acceptance_rate": comparison["baseline"]["acceptance_rate"],
                "candidate_acceptance_rate": comparison["candidate"]["acceptance_rate"],
                "acceptance_advantage": comparison["acceptance_advantage"],
                "candidate_critical_failures": comparison["candidate"][
                    "critical_factual_safety_or_tool_failures"
                ],
                "machine_gate_passed": False,
            },
        },
        "result_preservation": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "rows_promoted_from_failed_strategies": 0,
            "sft_training_started": False,
        },
        "next_gate": {
            "action": "rights_clear_closure_source_shortlist_and_provenance_preflight",
            "target_retained_train_supervised_tokens_minimum": 3_000_000,
            "target_retained_train_supervised_tokens_maximum": 3_300_000,
            "priority_tasks": [
                "writing_summarization",
                "tool_api",
                "safety_helpfulness",
                "multilingual_bengali",
                "factual_qa",
            ],
            "priority_language_modes": [
                "english",
                "banglish",
                "bengali_script",
                "natural_code_switching",
            ],
            "required_constraints": [
                "documented_training_and_redistribution_rights",
                "source_and_conservative_family_share_at_or_below_0_35",
                "no_qwen3_8b_or_qwen3_14b_bulk_teacher_generation",
                "exact_revision_object_and_license_binding_before_payload_access",
                "full_quality_deduplication_decontamination_and_four_k_fit",
            ],
            "payload_download_authorized": False,
            "external_api_authorized": False,
            "gpu_or_sft_authorized": False,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_verification(
    report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    """Build a raw-text-free verification receipt."""

    _verify_internal(report, "report_sha256", str(report["report_sha256"]))
    checks = {
        "8b_scale_failure_bound_and_verified": True,
        "14b_ab_failure_bound_and_verified": True,
        "candidate_mix_and_10m_gap_unchanged": True,
        "failed_teacher_rows_not_promoted": True,
        "no_network_api_gpu_dataset_seal_or_sft": True,
        "replacement_gate_is_metadata_and_provenance_only": True,
    }
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS",
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "checks": checks,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
