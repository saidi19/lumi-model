"""Exact bindings and blocker reporting for the future Lumi SFT execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_assisted_v2 import (
    load_ai_assisted_v2_evidence,
    load_ai_assisted_v2_plan,
)
from .posttraining_authoring_workflow_v2 import load_authoring_workflow_v2_plan
from .posttraining_capabilities import load_basic_capability_plan
from .posttraining_data import load_sft_policy
from .posttraining_first_party_v2 import load_first_party_authoring_v2_plan
from .posttraining_format import load_chat_template
from .posttraining_language import load_language_routing_plan
from .posttraining_readiness import load_final_base_binding
from .posttraining_source_quality import load_source_quality_audit_evidence
from .posttraining_sources import load_sft_source_registry

SFT_EXECUTION_PLAN_SCHEMA_VERSION = "lumi.sft-execution-plan.v0.1"
SFT_READINESS_REPORT_SCHEMA_VERSION = "lumi.sft-execution-readiness.v0.1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_AUTHORIZATION = {
    "dataset_promotion_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "public_release_authorized": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "ai_assisted_candidate_evidence",
        "ai_assisted_pilot",
        "assistant_behavior",
        "basic_capability_gate",
        "chat_template",
        "final_base",
        "first_party_authoring",
        "first_party_authoring_workflow",
        "identity",
        "language_routing_gate",
        "sft_data_contract",
        "source_registry",
        "source_quality_audit",
    }
)
EXPECTED_EXECUTION_ORDER = (
    "complete_ai_assisted_owner_spot_check_without_false_human_authorship_claims",
    "complete_representative_external_source_quality_and_provenance_audits",
    "promote_only_evidence_complete_sources",
    "filter_deduplicate_decontaminate_and_seal_dataset",
    "materialize_hash_bound_encoded_train_and_validation_shards",
    "implement_and_cpu_test_masked_label_sft_runner",
    "run_separately_authorized_two_step_gpu_smoke_and_fresh_resume",
    "run_separately_authorized_equal_token_learning_rate_screen",
    "select_profile_from_validation_quality_and_safety",
    "run_separately_authorized_bounded_full_sft",
    "seal_checkpoint_and_run_frozen_quality_safety_and_basic_capability_gates",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_sft_execution_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the non-authorizing SFT plan and verify every local binding."""

    plan = _load_json(Path(path))
    if plan.get("schema_version") != SFT_EXECUTION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported SFT execution plan schema")
    if plan.get("status") != "planning_only_blocked_before_gpu":
        raise ValueError("SFT execution plan status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("SFT execution authorization boundary changed")
    if tuple(plan.get("execution_order", ())) != EXPECTED_EXECUTION_ORDER:
        raise ValueError("SFT execution order changed")

    root = Path(repository_root)
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("SFT execution binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"SFT execution binding is invalid: {name}")
        relative_path = binding.get("path")
        expected_sha256 = binding.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"SFT execution binding path is invalid: {name}")
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise ValueError(f"SFT execution binding SHA-256 is invalid: {name}")
        if _file_sha256(root / relative_path) != expected_sha256:
            raise ValueError(f"SFT execution binding changed: {name}")

    dataset = plan.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("SFT execution dataset state is missing")
    if dataset.get("completed_human_authored_records") != 0:
        raise ValueError("planning SFT state may not claim authored records")
    if dataset.get("completed_independent_reviews") != 0:
        raise ValueError("planning SFT state may not claim reviews")
    if dataset.get("ai_assisted_candidate_records") != 320:
        raise ValueError("planning SFT state AI-assisted candidate count changed")
    if dataset.get("ai_assisted_machine_validated_records") != 320:
        raise ValueError("planning SFT state machine-validation count changed")
    if dataset.get("ai_assisted_owner_spot_checked_records") != 0:
        raise ValueError("planning SFT state may not claim owner spot checks")
    for field in (
        "ai_assisted_candidate_manifest_sha256",
        "ai_assisted_candidate_source_revision",
    ):
        value = dataset.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"planning SFT state has invalid {field}")
    if dataset.get("production_eligible_sources") != 0:
        raise ValueError("planning SFT state may not claim promoted sources")
    if dataset.get("representative_external_source_audit_complete") is not True:
        raise ValueError("representative external source audit is incomplete")
    audit_sha256 = dataset.get("representative_external_source_audit_report_sha256")
    if not isinstance(audit_sha256, str) or _SHA256.fullmatch(audit_sha256) is None:
        raise ValueError("representative external source audit SHA-256 is invalid")
    if dataset.get("representative_external_source_promotions") != 0:
        raise ValueError("representative external source audit may not promote sources")
    for field in (
        "encoded_training_shards_manifest_sha256",
        "measured_input_tokens",
        "measured_supervised_tokens",
        "sealed_dataset_manifest_sha256",
        "target_epochs",
        "target_optimizer_steps",
        "target_tokens",
    ):
        if dataset.get(field) is not None:
            raise ValueError(f"planning SFT state may not claim {field}")

    training = plan.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("SFT execution training state is missing")
    if training.get("objective") != "full_parameter_supervised_fine_tuning":
        raise ValueError("SFT training objective changed")
    if training.get("sequence_length") != 4096:
        raise ValueError("SFT sequence length changed")
    if training.get("trainer_status") != (
        "single_device_masked_label_cpu_canary_implemented_"
        "distributed_path_not_implemented"
    ):
        raise ValueError("SFT trainer implementation status changed")
    cpu_canary = training.get("cpu_canary")
    if not isinstance(cpu_canary, Mapping) or cpu_canary != {
        "production_seal_bound_run": False,
        "synthetic_assistant_only_loss": "passed",
        "synthetic_fresh_process_resume": "passed",
        "synthetic_steps": 2,
    }:
        raise ValueError("SFT CPU canary status changed")
    loss = training.get("loss")
    if not isinstance(loss, Mapping) or loss.get("ignore_index") != -100:
        raise ValueError("SFT masked-loss contract changed")
    if loss.get("supervised_roles") != ["assistant", "tool"]:
        raise ValueError("SFT supervised roles changed")
    return plan


def build_sft_execution_readiness_report(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate implemented foundations and enumerate every remaining launch blocker."""

    root = Path(repository_root)
    bindings = plan["bindings"]
    base = load_final_base_binding(root / bindings["final_base"]["path"])
    policy = load_sft_policy(root / bindings["sft_data_contract"]["path"])
    template = load_chat_template(root / bindings["chat_template"]["path"])
    registry = load_sft_source_registry(
        root / bindings["source_registry"]["path"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=bindings["source_registry"]["sha256"],
    )
    authoring, _ = load_first_party_authoring_v2_plan(
        root / bindings["first_party_authoring"]["path"],
        repository_root=root,
    )
    authoring_workflow = load_authoring_workflow_v2_plan(
        root / bindings["first_party_authoring_workflow"]["path"],
        repository_root=root,
    )
    ai_assisted_plan, ai_coverage = load_ai_assisted_v2_plan(
        root / bindings["ai_assisted_pilot"]["path"],
        repository_root=root,
    )
    ai_evidence = load_ai_assisted_v2_evidence(
        root / bindings["ai_assisted_candidate_evidence"]["path"]
    )
    if plan["dataset"]["ai_assisted_candidate_source_revision"] != ai_evidence[
        "candidate"
    ]["candidate_source_revision"]:
        raise ValueError("SFT plan AI-assisted source revision changed")
    if plan["dataset"]["ai_assisted_candidate_manifest_sha256"] != ai_evidence[
        "candidate"
    ]["candidate_manifest_sha256"]:
        raise ValueError("SFT plan AI-assisted candidate manifest changed")
    if plan["dataset"]["ai_assisted_candidate_records"] != ai_evidence["candidate"][
        "records"
    ]:
        raise ValueError("SFT plan AI-assisted candidate count changed")
    ai_source = registry.by_id.get("lumi_ai_assisted_sft_v0_2")
    if ai_source is None or ai_source.revision != ai_evidence["candidate"][
        "candidate_source_revision"
    ]:
        raise ValueError("SFT source registry AI-assisted revision changed")
    capabilities = load_basic_capability_plan(
        root / bindings["basic_capability_gate"]["path"]
    )
    language_routing = load_language_routing_plan(
        root / bindings["language_routing_gate"]["path"]
    )
    source_quality_audit = load_source_quality_audit_evidence(
        root / bindings["source_quality_audit"]["path"]
    )
    if source_quality_audit["report_sha256"] != plan["dataset"][
        "representative_external_source_audit_report_sha256"
    ]:
        raise ValueError("SFT plan source quality audit report changed")
    production_sources = [source for source in registry.sources if source.production_eligible]

    blockers = [
        "ai_assisted_records:dataset_level_owner_spot_check_incomplete",
        "sources:representative_external_audit_failed_closed_no_promotions",
        "sources:adapter_quality_and_provenance_remediation_required",
        "dataset:ai_assisted_pilot_is_not_the_complete_target_mix",
        "sources:zero_production_eligible",
        "dataset:target_mix_and_token_count_unmeasured",
        "dataset:production_seal_missing",
        "dataset:encoded_training_shards_not_materialized",
        "trainer:distributed_assistant_only_label_path_not_implemented",
        "trainer:production_seal_bound_cpu_loss_and_resume_canary_not_run",
        "gpu:two_step_smoke_and_fresh_resume_not_run",
        "gpu:learning_rate_screen_not_run",
        "authorization:paid_sft_not_approved",
        "evaluation:post_sft_quality_and_safety_not_run",
        "evaluation:language_routing_not_run",
    ]
    report = {
        "base": {
            "checkpoint_manifest_sha256": base["checkpoint"]["manifest_sha256"],
            "ready": True,
            "training_tokens": base["checkpoint"]["training_tokens"],
        },
        "blockers": blockers,
        "capability_gate": {
            "arithmetic_cases": capabilities["arithmetic"]["total_cases"],
            "ready_for_future_execution": True,
        },
        "cpu_canary": {
            "production_seal_bound": False,
            "synthetic_assistant_only_loss": True,
            "synthetic_fresh_process_resume": True,
        },
        "chat_and_loss_contract": {
            "assistant_only_labels": policy.supervised_roles == frozenset({"assistant", "tool"}),
            "context_window": template.context_window,
            "ready": True,
        },
        "first_party_authoring": {
            "completed_records": plan["dataset"]["completed_human_authored_records"],
            "planned_records": sum(authoring["totals"].values()),
            "template_ready": True,
        },
        "ai_assisted_pilot": {
            "candidate_records": ai_evidence["candidate"]["records"],
            "machine_validated_records": ai_evidence["candidate"][
                "automated_review_passed_records"
            ],
            "owner_sample_records": ai_assisted_plan["owner_gate"]["sample_records"],
            "owner_spot_check_complete": ai_evidence["owner_gate"]["complete"],
            "planned_records": sum(ai_coverage["totals"].values()),
            "production_eligible": ai_evidence["production_eligible"],
            "source_revision": ai_evidence["candidate"]["candidate_source_revision"],
        },
        "first_party_authoring_workflow": {
            "author_slots": len(authoring_workflow["workflow"]["author_slots"]),
            "human_identity_assignment_complete": False,
            "packet_count": authoring_workflow["workflow"]["packet_count"],
            "reviewer_slots": len(authoring_workflow["workflow"]["reviewer_slots"]),
            "rows_per_packet": authoring_workflow["workflow"]["rows_per_packet"],
            "ready": True,
        },
        "language_routing_gate": {
            "core_modes": language_routing["model_behavior_contract"][
                "release_core_modes"
            ],
            "metadata_cases": language_routing["metadata_case_matrix"]["expected_cases"],
            "model_inference_executed": False,
            "ready_for_future_execution": True,
        },
        "source_quality_audit": {
            "complete": source_quality_audit["audit_complete"],
            "outcome": source_quality_audit["representative_audit_outcome"],
            "production_promotions": source_quality_audit["production_promotions"],
            "report_sha256": source_quality_audit["report_sha256"],
            "sparse_tasks": source_quality_audit["candidate_coverage_gap"][
                "sparse_below_one_quarter_of_target_share"
            ],
            "total_blocker_findings": source_quality_audit[
                "total_production_blocker_findings"
            ],
            "zero_coverage_tasks": source_quality_audit["candidate_coverage_gap"][
                "zero_coverage_tasks"
            ],
        },
        "next_gate": (
            "complete_owner_spot_check_remediate_source_mix_and_build_distributed_masked_label_trainer"
        ),
        "paid_gpu_authorized": False,
        "production_eligible_source_count": len(production_sources),
        "ready_for_gpu": False,
        "schema_version": SFT_READINESS_REPORT_SCHEMA_VERSION,
        "source_registry_sha256": registry.registry_sha256,
    }
    report["report_sha256"] = _sha256(report)
    return report
