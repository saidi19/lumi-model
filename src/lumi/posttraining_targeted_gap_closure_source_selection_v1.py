"""Build a raw-text-free v0.21 targeted-gap source-selection preflight."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "lumi.sft-targeted-gap-closure-source-selection-plan.v0.1"
PLAN_STATUS = "authorized_local_aggregate_only_targeted_gap_source_selection"
REPORT_SCHEMA = "lumi.sft-targeted-gap-closure-source-selection-report.v0.1"
REPORT_STATUS = "STOP_no_complete_rights_clear_local_closure_set"
VERIFICATION_SCHEMA = (
    "lumi.sft-targeted-gap-closure-source-selection-verification.v0.1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(json_file_bytes(value))
    temporary.replace(destination)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = _require_sha256(value.get(field), f"{label}.{field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{label} internal seal changed")


def _resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing binding {label}: {path}")
    if sha256_file(path) != _require_sha256(specification.get("sha256"), f"{label}.sha256"):
        raise ValueError(f"{label} file SHA-256 changed")
    return path


def _bound_json(
    plan: Mapping[str, Any], repository_root: str | Path, binding: str
) -> dict[str, Any]:
    path = _resolve_binding(repository_root, plan["bindings"][binding], label=binding)
    return _read_object(path)


def _matrix_totals(
    matrix: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, int], dict[str, int]]:
    task_totals: dict[str, int] = {}
    language_totals: dict[str, int] = {}
    for task, languages in matrix.items():
        if not isinstance(languages, dict) or not languages:
            raise ValueError(f"allocation row {task} must be a non-empty object")
        task_total = 0
        for language, tokens in languages.items():
            if not isinstance(tokens, int) or tokens < 0:
                raise ValueError(f"allocation {task}/{language} must be non-negative")
            task_total += tokens
            language_totals[language] = language_totals.get(language, 0) + tokens
        task_totals[task] = task_total
    return task_totals, language_totals


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected targeted-gap plan schema")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("targeted-gap plan is not authorized")

    authorization = plan.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization must be an object")
    if authorization.get("local_aggregate_metadata_only") is not True:
        raise ValueError("plan must be local aggregate-metadata-only")
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
        "sft_training_authorized",
        "source_reads_authorized",
        "test_object_reads_authorized",
        "unrelated_access_authorized",
    )
    if any(authorization.get(name) is not False for name in forbidden):
        raise ValueError("plan authorizes a forbidden action")

    limits = plan.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits must be an object")
    for field in (
        "maximum_candidate_reads",
        "maximum_gpu_actions",
        "maximum_inference_calls",
        "maximum_network_bytes",
        "maximum_source_line_reads",
        "maximum_test_object_reads",
    ):
        if limits.get(field) != 0:
            raise ValueError(f"{field} must be zero")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("plan bindings must be non-empty")
    for name, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {name} must be an object")
        _resolve_binding(repository_root, specification, label=name)
    return plan


def _verify_bound_seals(objects: Mapping[str, Mapping[str, Any]]) -> None:
    fields = {
        "candidate_mix_report": "report_sha256",
        "candidate_mix_verification": "verification_sha256",
        "bedstc2_curation_report": "report_sha256",
        "bedstc2_curation_verification": "verification_sha256",
        "bedstc2_owner_receipt": "receipt_sha256",
        "bedstc2_owner_verification": "verification_sha256",
        "bedstc2_factual_rights_report": "report_sha256",
        "bedstc2_factual_rights_verification": "verification_sha256",
        "bedstc2_qualified_handoff": "packet_sha256",
        "bedstc2_qualified_handoff_verification": "verification_sha256",
    }
    for label, field in fields.items():
        _verify_internal_seal(objects[label], field, label)


def build_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    objects = {
        name: _bound_json(plan, repository_root, name)
        for name in (
            "source_registry_v21",
            "candidate_mix_report",
            "candidate_mix_verification",
            "bedstc2_curation_report",
            "bedstc2_curation_verification",
            "bedstc2_owner_receipt",
            "bedstc2_owner_verification",
            "bedstc2_factual_rights_report",
            "bedstc2_factual_rights_verification",
            "bedstc2_qualified_handoff",
            "bedstc2_qualified_handoff_verification",
        )
    }
    _verify_bound_seals(objects)

    registry = objects["source_registry_v21"]
    mix = objects["candidate_mix_report"]
    mix_verification = objects["candidate_mix_verification"]
    bed = objects["bedstc2_curation_report"]
    bed_verification = objects["bedstc2_curation_verification"]
    owner = objects["bedstc2_owner_receipt"]
    owner_verification = objects["bedstc2_owner_verification"]
    rights = objects["bedstc2_factual_rights_report"]
    rights_verification = objects["bedstc2_factual_rights_verification"]
    handoff = objects["bedstc2_qualified_handoff"]
    handoff_verification = objects["bedstc2_qualified_handoff_verification"]

    expected = plan["expected_current_state"]
    if len(registry.get("sources", [])) != expected["registry_source_count"]:
        raise ValueError("registry source count changed")
    if sha256_value(registry) != expected["registry_semantic_sha256"]:
        raise ValueError("registry semantic SHA-256 changed")
    if mix.get("status") != "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction":
        raise ValueError("candidate-mix report status changed")
    if (
        mix_verification.get("status")
        != "PASS_verified_registry_v0_21_full_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("candidate-mix verification status changed")
    if mix.get("production_eligible") is not False:
        raise ValueError("candidate mix unexpectedly became production eligible")

    pool = mix.get("candidate_pool")
    closure = mix.get("closure")
    if not isinstance(pool, dict) or not isinstance(closure, dict):
        raise ValueError("candidate-mix aggregate ledgers missing")
    for field in (
        "accepted_records_after_generic_quarantine",
        "accepted_train_supervised_tokens",
        "accepted_validation_supervised_tokens",
        "strict_targeted_gap_lower_bound",
    ):
        if pool.get(field) != expected[field]:
            raise ValueError(f"candidate-mix {field} changed")
    if len(pool.get("source_train_supervised_tokens", {})) != expected["accepted_source_count"]:
        raise ValueError("accepted source count changed")
    if closure.get("language_floor_gaps_behavior_safe") != expected["language_gaps"]:
        raise ValueError("behavior-safe language gaps changed")
    if closure.get("task_floor_gaps_behavior_safe") != expected["task_gaps"]:
        raise ValueError("behavior-safe task gaps changed")
    if closure.get("source_family_cap_violations") != []:
        raise ValueError("source-family cap violation appeared")
    if mix.get("next_gate", {}).get("action") != "close_residual_behavior_safe_task_and_language_gaps":
        raise ValueError("candidate-mix next gate changed")

    if bed.get("status") != "PASS_nonproduction_bedstc2_candidate_owner_review_pending":
        raise ValueError("Be-DSTC2 curation status changed")
    if (
        bed_verification.get("status")
        != "PASS_verified_nonproduction_bedstc2_candidate_owner_review_pending"
    ):
        raise ValueError("Be-DSTC2 curation verification status changed")
    if owner.get("status") != "owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending":
        raise ValueError("Be-DSTC2 owner-review status changed")
    if owner_verification.get("production_eligible") is not False:
        raise ValueError("owner review incorrectly became production eligible")
    if owner.get("owner_review", {}).get("accepted_records") != 32:
        raise ValueError("Be-DSTC2 owner accepted count changed")
    if rights.get("qualified_review_pending") is not True:
        raise ValueError("qualified rights-review pending state changed")
    if rights.get("rights_assessment", {}).get("criteria_unresolved") != 5:
        raise ValueError("unresolved rights criterion count changed")
    if rights_verification.get("production_eligible") is not False:
        raise ValueError("factual rights review incorrectly became production eligible")
    if handoff.get("status") != "READY_local_qualified_review_handoff_rights_unresolved":
        raise ValueError("qualified-review handoff status changed")
    if handoff.get("boundaries", {}).get("qualified_review_completed") is not False:
        raise ValueError("qualified review unexpectedly completed")
    if handoff_verification.get("boundaries", {}).get("production_eligible") is not False:
        raise ValueError("handoff verification incorrectly became production eligible")

    conditional = plan["conditional_bedstc2"]
    bed_tokens = bed.get("candidate", {}).get("train_supervised_tokens")
    if bed_tokens != conditional["train_supervised_tokens"]:
        raise ValueError("Be-DSTC2 train-token count changed")
    new_metrics = bed.get("audit", {}).get("new_source_metrics", {})
    if new_metrics.get("train_by_language_mode") != {"banglish": bed_tokens}:
        raise ValueError("Be-DSTC2 language allocation changed")
    if new_metrics.get("train_by_task") != {"conversation": bed_tokens}:
        raise ValueError("Be-DSTC2 task allocation changed")
    conditional_projection = bed.get("audit", {}).get("task_language_family", {})
    if conditional_projection.get("language_floor_gaps_behavior_safe") != conditional[
        "language_gaps_after_clearance"
    ]:
        raise ValueError("Be-DSTC2 conditional language projection changed")
    if conditional_projection.get("task_floor_gaps_behavior_safe") != conditional[
        "task_gaps_after_clearance"
    ]:
        raise ValueError("Be-DSTC2 conditional task projection changed")
    if conditional_projection.get("strict_targeted_gap_lower_bound") != conditional[
        "strict_gap_after_clearance"
    ]:
        raise ValueError("Be-DSTC2 conditional strict gap changed")

    capacity = plan["replacement_capacity_target"]
    matrix = capacity["task_by_language_train_supervised_tokens"]
    task_totals, language_totals = _matrix_totals(matrix)
    target_tokens = capacity["target_train_supervised_tokens"]
    if sum(task_totals.values()) != target_tokens or sum(language_totals.values()) != target_tokens:
        raise ValueError("replacement-capacity matrix does not sum to target")
    if task_totals != capacity["task_totals"]:
        raise ValueError("replacement-capacity task totals changed")
    if language_totals != capacity["language_totals"]:
        raise ValueError("replacement-capacity language totals changed")
    for task, gap in expected["task_gaps"].items():
        if task_totals.get(task, 0) < gap:
            raise ValueError(f"replacement target does not cover task gap: {task}")
    for language, gap in expected["language_gaps"].items():
        if language_totals.get(language, 0) < gap:
            raise ValueError(f"replacement target does not cover language gap: {language}")
    if target_tokens < expected["strict_targeted_gap_lower_bound"]:
        raise ValueError("replacement target is below strict lower bound")

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
        "source_selection": {
            "active_selected_sources": [],
            "complete_rights_clear_local_closure_set_exists": False,
            "accepted_registry_sources_already_accounted": expected["accepted_source_count"],
            "registry_sources_not_in_active_mix": expected["registry_source_count"]
            - expected["accepted_source_count"],
            "unconditional_remaining_strict_gap": expected[
                "strict_targeted_gap_lower_bound"
            ],
            "conditional_candidate": {
                "source_id": conditional["source_id"],
                "train_supervised_tokens": bed_tokens,
                "active_production_credit": 0,
                "owner_sample_accepted": 32,
                "qualified_rights_clearance_present": False,
                "required_clear_outcome": "CLEAR_EXACT_SCOPE",
                "language_gaps_after_clearance": conditional[
                    "language_gaps_after_clearance"
                ],
                "task_gaps_after_clearance": conditional["task_gaps_after_clearance"],
                "strict_gap_after_clearance": conditional["strict_gap_after_clearance"],
            },
            "ordinary_owner_review_is_not_qualified_rights_clearance": True,
            "stop_before_candidate_construction": True,
        },
        "replacement_capacity_target": {
            **capacity,
            "task_totals": task_totals,
            "language_totals": language_totals,
            "projected_train_supervised_tokens_without_bedstc2": expected[
                "accepted_train_supervised_tokens"
            ]
            + target_tokens,
            "projected_train_supervised_tokens_if_bedstc2_clears": expected[
                "accepted_train_supervised_tokens"
            ]
            + bed_tokens
            + target_tokens,
            "source_family_cap_must_be_recomputed_after_exact_source_selection": True,
        },
        "next_gate": plan["next_gate"],
        "authorization": plan["authorization"],
        "resource_accounting": {
            "bound_aggregate_metadata_files_read": len(objects),
            "bound_files_hashed": len(plan["bindings"]),
            "candidate_reads": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "outreach_actions": 0,
            "private_write_bytes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != REPORT_STATUS:
        raise ValueError("unexpected targeted-gap report identity")
    _verify_internal_seal(report, "report_sha256", "targeted_gap_report")
    if report.get("production_eligible") is not False:
        raise ValueError("targeted-gap report cannot be production eligible")
    if report.get("legal_conclusion_made") is not False:
        raise ValueError("targeted-gap report cannot make a legal conclusion")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("targeted-gap report retained raw training text")
    selection = report.get("source_selection")
    if not isinstance(selection, dict):
        raise ValueError("source-selection result missing")
    if selection.get("active_selected_sources") != []:
        raise ValueError("preflight selected an active source")
    if selection.get("complete_rights_clear_local_closure_set_exists") is not False:
        raise ValueError("preflight incorrectly claims a complete local closure set")
    conditional = selection.get("conditional_candidate")
    if not isinstance(conditional, dict) or conditional.get("active_production_credit") != 0:
        raise ValueError("conditional candidate received active production credit")
    accounting = report.get("resource_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("resource accounting missing")
    for field in (
        "candidate_reads",
        "gpu_actions",
        "inference_calls",
        "network_bytes",
        "outreach_actions",
        "private_write_bytes",
        "source_line_reads",
        "test_object_reads",
    ):
        if accounting.get(field) != 0:
            raise ValueError(f"non-zero prohibited accounting field: {field}")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS_verified_targeted_gap_snapshot_source_selection_blocked",
        "observed_at": plan["observed_at"],
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "all_bound_file_hashes_and_internal_seals_match": True,
            "registry_v0_21_and_candidate_mix_v0_15_snapshot_match": True,
            "unconditional_strict_gap_is_1137534": True,
            "bedstc2_active_production_credit_is_zero": True,
            "bedstc2_owner_review_not_treated_as_qualified_clearance": True,
            "bedstc2_conditional_strict_gap_is_1084387": True,
            "replacement_capacity_matrix_covers_all_current_task_and_language_gaps": True,
            "no_complete_rights_clear_local_closure_set_selected": True,
            "no_source_candidate_or_test_object_read": True,
            "no_network_outreach_gpu_inference_registry_promotion_seal_or_sft": True,
            "raw_training_text_absent": True,
            "production_eligible_false": True,
        },
        "resource_accounting": report["resource_accounting"],
        "production_eligible": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
