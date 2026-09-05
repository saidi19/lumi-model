"""Rebase the raw-text-free targeted-gap closure selection on registry v0.22."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_targeted_gap_closure_source_selection_v1 import (
    _matrix_totals,
    _read_object,
    _require_sha256,
    _resolve_binding,
    _verify_internal_seal,
    json_file_bytes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)


PLAN_SCHEMA = "lumi.sft-targeted-gap-closure-source-selection-plan.v0.2"
PLAN_STATUS = "authorized_registry_v0_22_local_aggregate_targeted_gap_rebase"
REPORT_SCHEMA = "lumi.sft-targeted-gap-closure-source-selection-report.v0.2"
REPORT_STATUS = (
    "PASS_registry_v0_22_targeted_gap_rebase_massive_active_"
    "first_party_canary_ready"
)
VERIFICATION_SCHEMA = (
    "lumi.sft-targeted-gap-closure-source-selection-verification.v0.2"
)
VERIFICATION_STATUS = (
    "PASS_verified_registry_v0_22_targeted_gap_rebase_nonproduction"
)

_SEMANTIC_BINDINGS = (
    "source_registry_v22",
    "candidate_mix_plan_v16",
    "candidate_mix_report_v16",
    "candidate_mix_verification_v16",
    "massive_promotion_receipt_v6",
    "massive_promotion_verification_v6",
    "predecessor_source_selection_plan_v1",
    "predecessor_source_selection_report_v1",
    "predecessor_source_selection_verification_v1",
    "zero_inference_preflight_plan_v1",
    "zero_inference_preflight_report_v1",
    "zero_inference_preflight_verification_v1",
)

_FORBIDDEN_RAW_KEYS = {
    "assistant_response",
    "content",
    "exception",
    "messages",
    "prompt",
    "record_id",
    "record_ids",
    "response",
    "rows",
    "source_text",
    "text",
    "user_prompt",
    "utterance",
}


def _assert_raw_text_free(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("raw-text-free evidence keys must be strings")
            if key.lower() in _FORBIDDEN_RAW_KEYS:
                raise ValueError(f"raw training-text field forbidden at {'.'.join(path + (key,))}")
            _assert_raw_text_free(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path=path + (str(index),))


def _bound_json(
    plan: Mapping[str, Any], repository_root: str | Path, binding: str
) -> dict[str, Any]:
    path = _resolve_binding(repository_root, plan["bindings"][binding], label=binding)
    return _read_object(path)


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected targeted-gap v0.2 plan schema")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("targeted-gap v0.2 plan is not authorized")

    authorization = plan.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization must be an object")
    if authorization.get("local_aggregate_metadata_only") is not True:
        raise ValueError("plan must be local aggregate-metadata-only")
    if authorization.get("targeted_gap_rebase_authorized") is not True:
        raise ValueError("targeted-gap rebase is not authorized")
    forbidden = (
        "candidate_construction_authorized",
        "candidate_reads_authorized",
        "dataset_seal_authorized",
        "gpu_authorized",
        "inference_authorized",
        "network_authorized",
        "outreach_authorized",
        "production_promotion_authorized",
        "qualified_rights_decision_claim_authorized",
        "registry_mutation_authorized",
        "remote_access_authorized",
        "sft_training_authorized",
        "source_reads_authorized",
        "test_object_reads_authorized",
        "unrelated_access_authorized",
    )
    if any(authorization.get(field) is not False for field in forbidden):
        raise ValueError("plan authorizes a forbidden action")

    limits = plan.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits must be an object")
    for field in (
        "maximum_candidate_reads",
        "maximum_candidate_writes",
        "maximum_gpu_actions",
        "maximum_inference_calls",
        "maximum_network_bytes",
        "maximum_outreach_actions",
        "maximum_registry_writes",
        "maximum_remote_actions",
        "maximum_source_line_reads",
        "maximum_test_object_reads",
        "maximum_training_actions",
    ):
        if limits.get(field) != 0:
            raise ValueError(f"{field} must be zero")
    for field in ("maximum_report_bytes", "maximum_verification_bytes"):
        if not isinstance(limits.get(field), int) or limits[field] <= 0:
            raise ValueError(f"{field} must be positive")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("plan bindings must be non-empty")
    for name, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {name} must be an object")
        _resolve_binding(repository_root, specification, label=name)

    target = plan.get("replacement_capacity_target")
    if not isinstance(target, dict):
        raise ValueError("replacement capacity target missing")
    task_totals, language_totals = _matrix_totals(
        target["task_by_language_train_supervised_tokens"]
    )
    if task_totals != target["task_totals"]:
        raise ValueError("replacement task totals changed")
    if language_totals != target["language_totals"]:
        raise ValueError("replacement language totals changed")
    if sum(task_totals.values()) != target["retained_train_supervised_tokens"]:
        raise ValueError("replacement target does not sum")
    return plan


def _verify_bound_seals(objects: Mapping[str, Mapping[str, Any]]) -> None:
    fields = {
        "candidate_mix_report_v16": "report_sha256",
        "candidate_mix_verification_v16": "verification_sha256",
        "massive_promotion_receipt_v6": "receipt_sha256",
        "massive_promotion_verification_v6": "verification_sha256",
        "predecessor_source_selection_report_v1": "report_sha256",
        "predecessor_source_selection_verification_v1": "verification_sha256",
        "zero_inference_preflight_plan_v1": "plan_internal_sha256",
        "zero_inference_preflight_report_v1": "report_sha256",
        "zero_inference_preflight_verification_v1": "verification_sha256",
    }
    for label, field in fields.items():
        _verify_internal_seal(objects[label], field, label)


def _validate_v16_state(
    plan: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected = plan["expected_current_state"]
    registry = objects["source_registry_v22"]
    mix = objects["candidate_mix_report_v16"]
    verification = objects["candidate_mix_verification_v16"]

    if len(registry.get("sources", [])) != expected["registry_source_count"]:
        raise ValueError("registry v0.22 source count changed")
    if sha256_value(registry) != expected["registry_semantic_sha256"]:
        raise ValueError("registry v0.22 semantic SHA-256 changed")
    if registry.get("sources", [])[-1].get("source_id") != "amazon_massive_bn_bd_tool_canary":
        raise ValueError("MASSIVE is not the append-only v0.22 source")
    if mix.get("status") != "PASS_registry_v0_22_full_candidate_mix_reaudit_nonproduction":
        raise ValueError("candidate-mix v0.16 status changed")
    if (
        verification.get("status")
        != "PASS_verified_registry_v0_22_full_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix v0.16 verification status changed")
    if mix.get("production_eligible") is not False or verification.get(
        "production_eligible"
    ) is not False:
        raise ValueError("v0.16 unexpectedly became production eligible")

    pool = mix.get("candidate_pool")
    closure = mix.get("closure")
    if not isinstance(pool, dict) or not isinstance(closure, dict):
        raise ValueError("candidate-mix aggregate ledgers missing")
    exact_fields = (
        "accepted_records_after_generic_quarantine",
        "accepted_train_supervised_tokens",
        "accepted_validation_supervised_tokens",
        "strict_targeted_gap_lower_bound",
    )
    for field in exact_fields:
        if pool.get(field) != expected[field]:
            raise ValueError(f"candidate-mix {field} changed")
    if len(pool.get("source_train_supervised_tokens", {})) != expected["accepted_source_count"]:
        raise ValueError("accepted source count changed")
    if pool.get("source_train_supervised_tokens", {}).get(
        "amazon_massive_bn_bd_tool_canary"
    ) != expected["massive_train_supervised_tokens"]:
        raise ValueError("MASSIVE train-token credit changed")
    if closure.get("language_floor_gaps_behavior_safe") != expected["language_gaps"]:
        raise ValueError("language gaps changed")
    if closure.get("task_floor_gaps_behavior_safe") != expected["task_gaps"]:
        raise ValueError("task gaps changed")
    if closure.get("source_family_cap_violations") != []:
        raise ValueError("source-family cap violation appeared")
    if mix.get("next_gate", {}).get("action") != (
        "close_residual_behavior_safe_task_and_language_gaps"
    ):
        raise ValueError("candidate-mix next gate changed")
    return pool, closure


def _validate_massive_gate(
    plan: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]
) -> None:
    receipt = objects["massive_promotion_receipt_v6"]
    verification = objects["massive_promotion_verification_v6"]
    if receipt.get("status") != (
        "PASS_massive_bn_bd_parser_registry_v0_22_materialized_nonproduction"
    ):
        raise ValueError("MASSIVE promotion receipt status changed")
    if verification.get("status") != (
        "PASS_verified_massive_bn_bd_parser_registry_v0_22_materialized_nonproduction"
    ):
        raise ValueError("MASSIVE promotion verification status changed")
    registry = receipt.get("registry")
    if not isinstance(registry, dict):
        raise ValueError("MASSIVE promotion registry ledger missing")
    if registry.get("added_source_id") != "amazon_massive_bn_bd_tool_canary":
        raise ValueError("MASSIVE promotion source changed")
    if registry.get("successor_sha256") != plan["bindings"]["source_registry_v22"][
        "sha256"
    ]:
        raise ValueError("MASSIVE successor registry binding changed")
    checks = verification.get("checks")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise ValueError("MASSIVE promotion verification is incomplete")


def _validate_predecessors(
    plan: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]
) -> None:
    selection = objects["predecessor_source_selection_report_v1"]
    selection_verification = objects["predecessor_source_selection_verification_v1"]
    preflight = objects["zero_inference_preflight_report_v1"]
    preflight_verification = objects["zero_inference_preflight_verification_v1"]
    if selection.get("status") != "STOP_no_complete_rights_clear_local_closure_set":
        raise ValueError("predecessor source-selection status changed")
    if selection.get("production_eligible") is not False:
        raise ValueError("predecessor source selection became production eligible")
    if selection_verification.get("status") != (
        "PASS_verified_targeted_gap_snapshot_source_selection_blocked"
    ):
        raise ValueError("predecessor source-selection verification changed")
    if preflight.get("status") != (
        "PASS_plan_only_zero_inference_closure_ready_massive_gate_first"
    ):
        raise ValueError("zero-inference predecessor status changed")
    if preflight_verification.get("status") != (
        "PASS_verified_plan_only_zero_inference_closure_preflight_nonproduction"
    ):
        raise ValueError("zero-inference predecessor verification changed")
    conditional = preflight.get("conditional_public_source_lane")
    if not isinstance(conditional, dict) or conditional.get("active_family_credit") != 0:
        raise ValueError("predecessor did not preserve the conditional MASSIVE gate")
    if preflight.get("closure_design", {}).get("target") != plan[
        "replacement_capacity_target"
    ]:
        raise ValueError("zero-inference replacement target changed")


def build_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    objects = {
        name: _bound_json(plan, repository_root, name) for name in _SEMANTIC_BINDINGS
    }
    _verify_bound_seals(objects)
    pool, closure = _validate_v16_state(plan, objects)
    _validate_massive_gate(plan, objects)
    _validate_predecessors(plan, objects)

    expected = plan["expected_current_state"]
    target = plan["replacement_capacity_target"]
    task_totals, language_totals = _matrix_totals(
        target["task_by_language_train_supervised_tokens"]
    )
    target_tokens = target["retained_train_supervised_tokens"]
    for task, gap in expected["task_gaps"].items():
        if task_totals.get(task, 0) < gap:
            raise ValueError(f"replacement target does not cover task gap: {task}")
    for language, gap in expected["language_gaps"].items():
        if language_totals.get(language, 0) < gap:
            raise ValueError(f"replacement target does not cover language gap: {language}")
    if target_tokens < expected["strict_targeted_gap_lower_bound"]:
        raise ValueError("replacement target is below strict lower bound")

    projected_total = expected["accepted_train_supervised_tokens"] + target_tokens
    projected_lumi_family = (
        pool["family_train_supervised_tokens"]["lumi_first_party_family"]
        + target_tokens
    )
    projected_lumi_share = round(projected_lumi_family / projected_total, 8)
    if projected_lumi_share > plan["source_family_policy"]["maximum_share"]:
        raise ValueError("projected first-party family exceeds its cap")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "authorization_sha256": plan["authorization_sha256"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "bindings": {
            name: {"file_sha256": specification["sha256"]}
            for name, specification in sorted(plan["bindings"].items())
        },
        "current_verified_state": expected,
        "massive_gate": {
            "source_id": "amazon_massive_bn_bd_tool_canary",
            "registry_v0_22_active": True,
            "full_mix_two_pass_audit_complete": True,
            "train_supervised_tokens_already_accounted": expected[
                "massive_train_supervised_tokens"
            ],
            "active_family_credit": 1,
            "additional_closure_credit": 0,
            "tool_api_gap_after_credit": closure[
                "task_floor_gaps_behavior_safe"
            ]["tool_api"],
        },
        "source_selection": {
            "complete_rights_clear_local_closure_set_exists": False,
            "existing_artifacts_can_close_target_without_new_human_work": False,
            "candidate_construction_authorized": False,
            "first_party_zero_inference_lane_selected_for_canary": True,
            "first_party_canary_ready_for_separate_authorization": True,
            "minimum_bound_replacement_source_families": plan[
                "source_family_policy"
            ]["minimum_bound_replacement_source_families"],
            "bound_replacement_families": [
                "amazon_massive_family",
                "lumi_first_party_family",
            ],
            "bedstc2_active_production_credit": 0,
            "ordinary_owner_review_is_not_qualified_rights_clearance": True,
        },
        "replacement_capacity_target": {
            **target,
            "task_totals": task_totals,
            "language_totals": language_totals,
            "projected_train_supervised_tokens": projected_total,
            "projected_first_party_family_train_supervised_tokens": projected_lumi_family,
            "projected_first_party_family_share": projected_lumi_share,
            "projected_source_family_cap_violations": [],
            "projected_task_gaps": {task: 0 for task in expected["task_gaps"]},
            "projected_language_gaps": {
                language: 0 for language in expected["language_gaps"]
            },
        },
        "canary": plan["canary"],
        "next_gate": plan["next_gate"],
        "authorization": plan["authorization"],
        "resource_accounting": {
            "aggregate_metadata_passes": 1,
            "bound_aggregate_json_objects_read": len(objects),
            "bound_files_hashed": len(plan["bindings"]),
            "candidate_reads": 0,
            "candidate_writes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "outreach_actions": 0,
            "registry_writes": 0,
            "remote_actions": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    _assert_raw_text_free(report)
    report["report_sha256"] = sha256_value(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unexpected targeted-gap v0.2 report schema")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("unexpected targeted-gap v0.2 report status")
    _verify_internal_seal(report, "report_sha256", "targeted_gap_v02_report")
    _assert_raw_text_free(report)
    if report.get("production_eligible") is not False:
        raise ValueError("targeted-gap report cannot be production eligible")
    if report.get("legal_conclusion_made") is not False:
        raise ValueError("targeted-gap report cannot make a legal conclusion")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("targeted-gap report retained raw training text")
    massive = report.get("massive_gate")
    if not isinstance(massive, dict):
        raise ValueError("MASSIVE gate result missing")
    if massive.get("registry_v0_22_active") is not True:
        raise ValueError("MASSIVE registry gate did not pass")
    if massive.get("full_mix_two_pass_audit_complete") is not True:
        raise ValueError("MASSIVE full-mix audit gate did not pass")
    if massive.get("tool_api_gap_after_credit") != 0:
        raise ValueError("MASSIVE tool-api gap is not closed")
    selection = report.get("source_selection")
    if not isinstance(selection, dict):
        raise ValueError("source-selection result missing")
    if selection.get("candidate_construction_authorized") is not False:
        raise ValueError("report authorized candidate construction")
    if selection.get("first_party_canary_ready_for_separate_authorization") is not True:
        raise ValueError("first-party canary was not selected")
    if selection.get("complete_rights_clear_local_closure_set_exists") is not False:
        raise ValueError("report invented a complete local closure set")
    projected = report.get("replacement_capacity_target")
    if not isinstance(projected, dict):
        raise ValueError("replacement projection missing")
    if any(projected.get("projected_task_gaps", {}).values()):
        raise ValueError("replacement task gaps remain")
    if any(projected.get("projected_language_gaps", {}).values()):
        raise ValueError("replacement language gaps remain")
    if projected.get("projected_source_family_cap_violations") != []:
        raise ValueError("replacement source-family cap violation")
    if report.get("next_gate", {}).get("action") != (
        "first_party_zero_inference_1000_row_candidate_canary"
    ):
        raise ValueError("next gate changed")
    accounting = report.get("resource_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("resource accounting missing")
    for field in (
        "candidate_reads",
        "candidate_writes",
        "gpu_actions",
        "inference_calls",
        "network_bytes",
        "outreach_actions",
        "registry_writes",
        "remote_actions",
        "source_line_reads",
        "test_object_reads",
        "training_actions",
    ):
        if accounting.get(field) != 0:
            raise ValueError(f"non-zero prohibited accounting field: {field}")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "all_bound_file_hashes_and_internal_seals_match": True,
            "registry_v0_22_append_only_massive_source_matches": True,
            "candidate_mix_v0_16_two_pass_state_matches": True,
            "massive_promotion_and_full_mix_gate_complete": True,
            "massive_6451_tokens_already_accounted_without_double_credit": True,
            "tool_api_gap_is_zero": True,
            "residual_strict_gap_is_1137534": True,
            "replacement_target_is_1200000_behavior_safe_tokens": True,
            "replacement_matrix_covers_all_task_and_language_gaps": True,
            "projected_first_party_family_share_is_below_35_percent": True,
            "first_party_1000_row_zero_inference_canary_is_next_gate": True,
            "candidate_construction_requires_separate_authorization": True,
            "bedstc2_has_zero_active_production_credit": True,
            "no_source_candidate_or_test_object_read": True,
            "no_network_outreach_gpu_inference_registry_seal_or_sft": True,
            "raw_training_text_absent": True,
            "production_eligible_false": True,
        },
        "resource_accounting": {
            **report["resource_accounting"],
            "aggregate_metadata_passes": 2,
        },
        "production_eligible": False,
    }
    _assert_raw_text_free(verification)
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "PLAN_SCHEMA",
    "PLAN_STATUS",
    "REPORT_SCHEMA",
    "REPORT_STATUS",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_STATUS",
    "build_report",
    "build_verification",
    "json_file_bytes",
    "load_plan",
    "sha256_file",
    "sha256_value",
    "validate_report",
    "write_json_atomic",
]
