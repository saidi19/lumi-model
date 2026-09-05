"""Plan-only preflight for a Lumi first-party zero-inference SFT closure lane."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "lumi.sft-first-party-zero-inference-closure-preflight-plan.v0.1"
PLAN_STATUS = "authorized_plan_only_first_party_zero_inference_closure_preflight"
REPORT_SCHEMA = "lumi.sft-first-party-zero-inference-closure-preflight-report.v0.1"
REPORT_STATUS = "PASS_plan_only_zero_inference_closure_ready_massive_gate_first"
VERIFICATION_SCHEMA = (
    "lumi.sft-first-party-zero-inference-closure-preflight-verification.v0.1"
)
VERIFICATION_STATUS = (
    "PASS_verified_plan_only_zero_inference_closure_preflight_nonproduction"
)

SOURCE_SELECTION_STATUS = "STOP_no_complete_rights_clear_local_closure_set"
SOURCE_SELECTION_VERIFICATION_STATUS = (
    "PASS_verified_targeted_gap_snapshot_source_selection_blocked"
)
PUBLIC_DISCOVERY_STATUS = "PASS_massive_train_only_ready_jontro_hold_nonproduction"
PUBLIC_DISCOVERY_VERIFICATION_STATUS = (
    "PASS_verified_public_gap_discovery_nonproduction"
)
MASSIVE_ACQUISITION_PREFLIGHT_STATUS = "PREPARED_not_executable"

EXPECTED_LANGUAGE_TOTALS = {
    "banglish": 800_000,
    "natural_code_switching": 400_000,
}
EXPECTED_TASK_TOTALS = {
    "conversation": 150_000,
    "factual_qa": 150_000,
    "multilingual_bengali": 150_000,
    "safety_helpfulness": 300_000,
    "tool_api": 150_000,
    "writing_summarization": 300_000,
}
EXPECTED_TASK_LANGUAGE_MATRIX = {
    "conversation": {"banglish": 100_000, "natural_code_switching": 50_000},
    "factual_qa": {"banglish": 100_000, "natural_code_switching": 50_000},
    "multilingual_bengali": {
        "banglish": 100_000,
        "natural_code_switching": 50_000,
    },
    "safety_helpfulness": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
    "tool_api": {"banglish": 100_000, "natural_code_switching": 50_000},
    "writing_summarization": {
        "banglish": 200_000,
        "natural_code_switching": 100_000,
    },
}
EXPECTED_LANE_TOTALS = {
    "deterministic_grounded": 300_000,
    "human_authored": 900_000,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_AUTHORIZATION_FIELDS = (
    "candidate_construction_authorized",
    "candidate_reads_authorized",
    "dataset_seal_authorized",
    "gpu_authorized",
    "inference_authorized",
    "network_authorized",
    "outreach_authorized",
    "production_promotion_authorized",
    "qualified_legal_conclusion_authorized",
    "registry_mutation_authorized",
    "remote_access_authorized",
    "sft_training_authorized",
    "source_reads_authorized",
    "test_object_reads_authorized",
    "unrelated_access_authorized",
)
_ZERO_LIMIT_FIELDS = (
    "maximum_candidate_reads",
    "maximum_gpu_actions",
    "maximum_inference_calls",
    "maximum_network_bytes",
    "maximum_outreach_actions",
    "maximum_remote_actions",
    "maximum_source_line_reads",
    "maximum_test_object_reads",
)
_REQUIRED_BINDINGS = {
    "assistant_behavior_spec",
    "benchmark_signatures",
    "chat_template",
    "data_quality",
    "deterministic_completion_report",
    "deterministic_completion_verification",
    "human_authoring_blank_audit",
    "human_authoring_plan_v02",
    "human_authoring_workflow_v02",
    "implementation",
    "massive_acquisition_preflight",
    "public_source_discovery_report",
    "public_source_discovery_verification",
    "runbook",
    "runner",
    "scale_quality_failure_report",
    "scale_quality_failure_verification",
    "source_registry_v21",
    "source_selection_report",
    "source_selection_verification",
    "test",
    "verifier",
}


def sha256_value(value: Any) -> str:
    """Return the SHA-256 of canonical compact JSON."""

    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a local file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a JSON object in the repository evidence format."""

    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write one new JSON artifact."""

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
    expected = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected:
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
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise ValueError(f"allocation {task}/{language} must be non-negative")
            task_total += tokens
            language_totals[language] = language_totals.get(language, 0) + tokens
        task_totals[task] = task_total
    return task_totals, language_totals


def _validate_target(plan: Mapping[str, Any]) -> None:
    target = plan.get("target")
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    matrix = target.get("task_by_language_train_supervised_tokens")
    if matrix != EXPECTED_TASK_LANGUAGE_MATRIX:
        raise ValueError("task-language closure matrix changed")
    task_totals, language_totals = _matrix_totals(matrix)
    if task_totals != EXPECTED_TASK_TOTALS:
        raise ValueError("task totals changed")
    if language_totals != EXPECTED_LANGUAGE_TOTALS:
        raise ValueError("language totals changed")
    if target.get("task_totals") != task_totals:
        raise ValueError("declared task totals do not match the matrix")
    if target.get("language_totals") != language_totals:
        raise ValueError("declared language totals do not match the matrix")
    if target.get("retained_train_supervised_tokens") != 1_200_000:
        raise ValueError("retained train-token target changed")

    lanes = plan.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(EXPECTED_LANE_TOTALS):
        raise ValueError("closure lanes changed")
    lane_totals: dict[str, int] = {}
    aggregate: dict[str, dict[str, int]] = {}
    for lane_name, expected_total in EXPECTED_LANE_TOTALS.items():
        lane = lanes.get(lane_name)
        if not isinstance(lane, dict):
            raise ValueError(f"lane {lane_name} must be an object")
        lane_matrix = lane.get("task_by_language_train_supervised_tokens")
        if not isinstance(lane_matrix, dict):
            raise ValueError(f"lane {lane_name} matrix is missing")
        lane_tasks, _ = _matrix_totals(lane_matrix)
        lane_total = sum(lane_tasks.values())
        if (
            lane_total != expected_total
            or lane.get("retained_train_supervised_tokens") != lane_total
        ):
            raise ValueError(f"lane {lane_name} total changed")
        lane_totals[lane_name] = lane_total
        for task, language_map in lane_matrix.items():
            for language, tokens in language_map.items():
                aggregate.setdefault(task, {})[language] = (
                    aggregate.setdefault(task, {}).get(language, 0) + tokens
                )
    if aggregate != EXPECTED_TASK_LANGUAGE_MATRIX:
        raise ValueError("closure lanes do not reproduce the target matrix")
    if lane_totals != EXPECTED_LANE_TOTALS:
        raise ValueError("closure lane totals changed")


def _validate_canary(plan: Mapping[str, Any]) -> None:
    canary = plan.get("canary")
    if not isinstance(canary, dict):
        raise ValueError("canary contract must be an object")
    expected = {
        "candidate_records": 1000,
        "train_records": 800,
        "validation_records": 200,
        "target_retained_train_supervised_tokens": 80_000,
        "minimum_retained_train_supervised_tokens": 60_000,
        "maximum_retained_train_supervised_tokens": 100_000,
        "minimum_overall_acceptance_rate": 0.95,
        "minimum_per_task_language_cell_acceptance_rate": 0.90,
        "critical_failures_maximum": 0,
        "final_row_independent_human_review_rate": 1.0,
        "candidate_construction_authorized": False,
    }
    if any(canary.get(field) != value for field, value in expected.items()):
        raise ValueError("canary contract changed")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    """Load and validate the bounded, non-authorizing plan."""

    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected zero-inference closure plan schema")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("zero-inference closure plan status changed")
    _verify_internal_seal(plan, "plan_internal_sha256", "zero_inference_plan")

    authorization = plan.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization must be an object")
    if authorization.get("local_plan_only") is not True:
        raise ValueError("plan must be local and plan-only")
    if authorization.get("raw_text_free_aggregate_evidence_authorized") is not True:
        raise ValueError("raw-text-free aggregate evidence must be authorized")
    if any(authorization.get(field) is not False for field in _FORBIDDEN_AUTHORIZATION_FIELDS):
        raise ValueError("plan authorizes a forbidden action")

    limits = plan.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits must be an object")
    if any(limits.get(field) != 0 for field in _ZERO_LIMIT_FIELDS):
        raise ValueError("plan permits a prohibited read or action")
    for field in ("maximum_report_bytes", "maximum_verification_bytes"):
        value = limits.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != _REQUIRED_BINDINGS:
        raise ValueError("plan binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {name} must be an object")
        _resolve_binding(repository_root, specification, label=name)

    _validate_target(plan)
    _validate_canary(plan)

    family_policy = plan.get("source_family_policy")
    if not isinstance(family_policy, dict):
        raise ValueError("source-family policy must be an object")
    required_family_policy = {
        "conservative_internal_family_id": "lumi_first_party_family",
        "count_internal_lanes_as_distinct_source_families": False,
        "minimum_new_source_families_in_bound_replacement_plan": 2,
        "fallback_owner_decision_required_if_massive_fails": True,
        "owner_decision_required_before_massive_gate": False,
        "source_family_share_maximum": 0.35,
    }
    if any(family_policy.get(field) != value for field, value in required_family_policy.items()):
        raise ValueError("source-family policy changed")

    review = plan.get("review_contract")
    if not isinstance(review, dict):
        raise ValueError("review contract must be an object")
    required_true = (
        "author_and_reviewer_must_differ_per_row",
        "completed_contributor_rights_declarations_required",
        "deterministic_rows_must_not_claim_human_per_row_authorship",
        "final_row_independent_human_review_required",
        "human_lane_model_assistance_prohibited",
        "owner_dataset_signoff_required",
        "qualified_contributor_rights_review_required_before_candidate_credit",
    )
    if any(review.get(field) is not True for field in required_true):
        raise ValueError("review contract weakened")
    if review.get("final_row_review_rate") != 1.0:
        raise ValueError("final row review rate must remain 100 percent")

    return plan


def _validate_public_source_lane(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    discovery = _bound_json(plan, repository_root, "public_source_discovery_report")
    verification = _bound_json(
        plan, repository_root, "public_source_discovery_verification"
    )
    acquisition = _bound_json(plan, repository_root, "massive_acquisition_preflight")
    _verify_internal_seal(discovery, "report_sha256", "public_source_discovery_report")
    _verify_internal_seal(
        verification,
        "verification_sha256",
        "public_source_discovery_verification",
    )
    if discovery.get("status") != PUBLIC_DISCOVERY_STATUS:
        raise ValueError("public-source discovery status changed")
    if verification.get("status") != PUBLIC_DISCOVERY_VERIFICATION_STATUS:
        raise ValueError("public-source discovery verification changed")
    if discovery.get("production_eligible") is not False:
        raise ValueError("public-source discovery became production eligible")
    if verification.get("production_eligible") is not False:
        raise ValueError("public-source verification became production eligible")
    public_checks = verification.get("checks")
    if (
        not isinstance(public_checks, dict)
        or not public_checks
        or not all(public_checks.values())
    ):
        raise ValueError("public-source verification checks changed")
    summary = discovery.get("decision_summary")
    if not isinstance(summary, dict):
        raise ValueError("public-source discovery summary missing")
    if summary.get("new_public_families_ready_for_payload_acquisition") != 1:
        raise ValueError("conditional public-family count changed")
    if summary.get("massive_may_seed_tool_api_only_after_all_later_gates") is not True:
        raise ValueError("MASSIVE conditional role changed")
    if summary.get("public_source_closes_full_language_gap") is not False:
        raise ValueError("public source unexpectedly closes language gap")
    if summary.get("public_source_closes_factual_qa_gap") is not False:
        raise ValueError("public source unexpectedly closes factual gap")
    if acquisition.get("status") != MASSIVE_ACQUISITION_PREFLIGHT_STATUS:
        raise ValueError("MASSIVE acquisition preflight status changed")
    if acquisition.get("authorization", {}).get("execution_authorized") is not False:
        raise ValueError("MASSIVE acquisition unexpectedly executable")
    source = acquisition.get("bound_source")
    if not isinstance(source, dict) or source.get("object_sha256") != (
        "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d"
    ):
        raise ValueError("MASSIVE exact object binding changed")
    return {
        "conditional_family_id": "amazon_massive_family",
        "conditional_source_id": "amazon_massive_bn_bd_tool_canary",
        "active_family_credit": 0,
        "active_task_or_language_credit": 0,
        "acquisition_execution_authorized": False,
        "can_satisfy_second_new_family_only_after_all_later_gates": True,
        "fallback_owner_decision_required_if_massive_fails": True,
    }


def _validate_bound_source_selection(
    plan: Mapping[str, Any], repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _bound_json(plan, repository_root, "source_selection_report")
    verification = _bound_json(plan, repository_root, "source_selection_verification")
    _verify_internal_seal(report, "report_sha256", "source_selection_report")
    _verify_internal_seal(
        verification,
        "verification_sha256",
        "source_selection_verification",
    )
    if report.get("status") != SOURCE_SELECTION_STATUS:
        raise ValueError("bound source-selection status changed")
    if verification.get("status") != SOURCE_SELECTION_VERIFICATION_STATUS:
        raise ValueError("bound source-selection verification status changed")
    if report.get("production_eligible") is not False:
        raise ValueError("source selection unexpectedly became production eligible")
    if verification.get("production_eligible") is not False:
        raise ValueError("source-selection verification became production eligible")
    checks = verification.get("checks")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise ValueError("source-selection verification check changed")
    return report, verification


def _validate_precedents(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    authoring_plan = _bound_json(plan, repository_root, "human_authoring_plan_v02")
    authoring_workflow = _bound_json(plan, repository_root, "human_authoring_workflow_v02")
    blank_audit = _bound_json(plan, repository_root, "human_authoring_blank_audit")
    scale_failure = _bound_json(plan, repository_root, "scale_quality_failure_report")
    scale_verification = _bound_json(
        plan, repository_root, "scale_quality_failure_verification"
    )
    deterministic = _bound_json(plan, repository_root, "deterministic_completion_report")
    deterministic_verification = _bound_json(
        plan, repository_root, "deterministic_completion_verification"
    )
    registry = _bound_json(plan, repository_root, "source_registry_v21")

    if authoring_plan.get("status") != (
        "bounded_message_native_blank_template_ready_no_authored_records"
    ):
        raise ValueError("human authoring plan status changed")
    if authoring_plan.get("totals") != {"train_records": 240, "validation_records": 80}:
        raise ValueError("human authoring pilot bounds changed")
    if authoring_workflow.get("status") != (
        "message_native_blank_human_packets_ready_identity_assignment_pending"
    ):
        raise ValueError("human authoring workflow status changed")
    if blank_audit.get("authoring_complete") is not False:
        raise ValueError("human authoring unexpectedly became complete")
    if blank_audit.get("status_counts") != {"accepted": 0, "blank": 320, "invalid": 0}:
        raise ValueError("human authoring blank-state count changed")

    if scale_failure.get("status") != "stopped_fail_closed_quality_screen_failed":
        raise ValueError("scale quality failure status changed")
    acceptance = scale_failure.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("scale quality acceptance ledger missing")
    if acceptance.get("cumulative_rate") != 0.6:
        raise ValueError("scale quality cumulative rate changed")
    if acceptance.get("amended_screen_rate") != 0.45:
        raise ValueError("scale quality amended-screen rate changed")
    if scale_verification.get("status") != "PASS":
        raise ValueError("scale quality failure verification changed")

    if deterministic.get("production_eligible") is not False:
        raise ValueError("deterministic precedent became production eligible")
    completion = deterministic.get("deterministic_completion")
    generation = deterministic.get("generation")
    if not isinstance(completion, dict) or not isinstance(generation, dict):
        raise ValueError("deterministic completion ledger missing")
    if completion.get("additional_records_retained") != 15:
        raise ValueError("deterministic precedent retained-count changed")
    if completion.get("local_inference_calls") != 0 or generation.get("calls") != 0:
        raise ValueError("deterministic precedent inference accounting changed")
    if deterministic_verification.get("status") != (
        "verified_private_candidate_pending_owner_review"
    ):
        raise ValueError("deterministic precedent verification status changed")
    if deterministic_verification.get("deterministic_template_records_verified") != 15:
        raise ValueError("deterministic precedent verification count changed")
    if deterministic_verification.get("local_inference_calls_verified") != 0:
        raise ValueError("deterministic precedent verification inference changed")

    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry sources missing")
    human_entries = [
        entry for entry in sources if entry.get("source_id") == "lumi_owned_sft_v0_1"
    ]
    if len(human_entries) != 1:
        raise ValueError("planned human-authored source entry changed")
    human_entry = human_entries[0]
    if (
        human_entry.get("status") != "planned"
        or human_entry.get("production_eligible") is not False
    ):
        raise ValueError("planned human-authored source state changed")
    if "first_party_records_and_author_declarations_not_created" not in human_entry.get(
        "blockers", []
    ):
        raise ValueError("planned human-authored source blocker changed")

    return {
        "deterministic_precedent_local_inference_calls": 0,
        "deterministic_precedent_records": 15,
        "human_authoring_accepted_records": 0,
        "human_authoring_blank_records": 320,
        "human_authoring_identity_assignment_pending": True,
        "scale_amended_screen_acceptance_rate": 0.45,
        "scale_cumulative_acceptance_rate": 0.6,
        "scale_generation_strategy_rejected": True,
    }


def build_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    """Build a raw-text-free plan assessment without constructing candidates."""

    source_selection, _ = _validate_bound_source_selection(plan, repository_root)
    public_source_lane = _validate_public_source_lane(plan, repository_root)
    precedent = _validate_precedents(plan, repository_root)
    current = source_selection.get("current_verified_state")
    replacement = source_selection.get("replacement_capacity_target")
    if not isinstance(current, dict) or not isinstance(replacement, dict):
        raise ValueError("bound source-selection aggregate state missing")
    if current.get("strict_targeted_gap_lower_bound") != 1_137_534:
        raise ValueError("strict targeted gap changed")
    if current.get("language_gaps") != {
        "banglish": 772_114,
        "bengali_script": 0,
        "english": 0,
        "natural_code_switching": 365_420,
    }:
        raise ValueError("bound behavior-safe language gaps changed")
    if replacement.get("target_train_supervised_tokens") != 1_200_000:
        raise ValueError("bound replacement target changed")
    if replacement.get("language_totals") != EXPECTED_LANGUAGE_TOTALS:
        raise ValueError("bound replacement language totals changed")
    if replacement.get("task_totals") != EXPECTED_TASK_TOTALS:
        raise ValueError("bound replacement task totals changed")
    if replacement.get("task_by_language_train_supervised_tokens") != (
        EXPECTED_TASK_LANGUAGE_MATRIX
    ):
        raise ValueError("bound replacement task-language matrix changed")
    if replacement.get("minimum_new_source_families") != 2:
        raise ValueError("bound source-family diversity target changed")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "plan_internal_sha256": plan["plan_internal_sha256"],
        "bindings": {
            name: {"file_sha256": specification["sha256"]}
            for name, specification in sorted(plan["bindings"].items())
        },
        "current_verified_state": current,
        "bound_source_selection": {
            "report_internal_sha256": source_selection["report_sha256"],
            "status": source_selection["status"],
            "production_eligible": False,
            "minimum_new_source_families": replacement["minimum_new_source_families"],
        },
        "precedent_assessment": precedent,
        "conditional_public_source_lane": public_source_lane,
        "closure_design": {
            "target": plan["target"],
            "lanes": plan["lanes"],
            "maximum_candidate_train_rows": plan["scale_bounds"][
                "maximum_candidate_train_rows"
            ],
            "maximum_candidate_validation_rows": plan["scale_bounds"][
                "maximum_candidate_validation_rows"
            ],
            "candidate_construction_authorized": False,
            "feasible_only_after_human_rights_review_and_canary": True,
            "existing_artifacts_can_close_target_without_new_human_work": False,
        },
        "review_contract": plan["review_contract"],
        "source_family_policy": plan["source_family_policy"],
        "canary": plan["canary"],
        "quality_gates": plan["quality_gates"],
        "next_gate": plan["next_gate"],
        "authorization": plan["authorization"],
        "resource_accounting": {
            "bound_aggregate_json_objects_read": 12,
            "bound_files_hashed": len(plan["bindings"]),
            "candidate_reads": 0,
            "candidate_writes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "outreach_actions": 0,
            "registry_mutations": 0,
            "remote_actions": 0,
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
    """Validate one materialized plan-only report."""

    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != REPORT_STATUS:
        raise ValueError("unexpected zero-inference preflight report identity")
    _verify_internal_seal(report, "report_sha256", "zero_inference_preflight_report")
    for field in ("production_eligible", "legal_conclusion_made", "raw_training_text_retained"):
        if report.get(field) is not False:
            raise ValueError(f"report boundary changed: {field}")
    closure = report.get("closure_design")
    if not isinstance(closure, dict):
        raise ValueError("closure design missing")
    if closure.get("candidate_construction_authorized") is not False:
        raise ValueError("report authorizes candidate construction")
    if closure.get("existing_artifacts_can_close_target_without_new_human_work") is not False:
        raise ValueError("report overstates existing first-party capacity")
    family_policy = report.get("source_family_policy")
    if (
        not isinstance(family_policy, dict)
        or family_policy.get("fallback_owner_decision_required_if_massive_fails")
        is not True
    ):
        raise ValueError("source-family fallback decision disappeared")
    if family_policy.get("owner_decision_required_before_massive_gate") is not False:
        raise ValueError("owner decision incorrectly blocks the MASSIVE gate")
    if family_policy.get("count_internal_lanes_as_distinct_source_families") is not False:
        raise ValueError("internal lanes were incorrectly counted as distinct families")
    public_lane = report.get("conditional_public_source_lane")
    if not isinstance(public_lane, dict):
        raise ValueError("conditional public-source lane missing")
    if public_lane.get("active_family_credit") != 0:
        raise ValueError("MASSIVE received premature family credit")
    if public_lane.get("can_satisfy_second_new_family_only_after_all_later_gates") is not True:
        raise ValueError("conditional MASSIVE family path disappeared")
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
        "registry_mutations",
        "remote_actions",
        "source_line_reads",
        "test_object_reads",
    ):
        if accounting.get(field) != 0:
            raise ValueError(f"non-zero prohibited accounting field: {field}")


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    """Build independent structural verification for the plan-only report."""

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
            "bound_source_selection_report_and_verification_match": True,
            "current_strict_gap_is_1137534": True,
            "target_is_800k_banglish_and_400k_natural_code_switching": True,
            "human_lane_is_900k_and_deterministic_grounded_lane_is_300k": True,
            "human_authoring_precedent_is_zero_of_320_complete": True,
            "zero_inference_deterministic_precedent_is_only_15_records": True,
            "failed_teacher_scale_strategy_not_reused": True,
            "internal_lanes_not_misrepresented_as_two_source_families": True,
            "massive_is_conditional_second_family_with_zero_active_credit": True,
            "massive_gate_precedes_first_party_candidate_canary": True,
            "owner_family_decision_is_fallback_only_if_massive_fails": True,
            "full_independent_human_row_review_required": True,
            "no_candidate_source_or_test_object_read": True,
            "no_network_outreach_gpu_inference_registry_promotion_seal_or_sft": True,
            "raw_training_text_absent": True,
            "production_eligible_false": True,
        },
        "resource_accounting": report["resource_accounting"],
        "legal_conclusion_made": False,
        "production_eligible": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    """Validate the independent structural verification and its boundaries."""

    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != VERIFICATION_STATUS
    ):
        raise ValueError("unexpected zero-inference verification identity")
    _verify_internal_seal(
        verification,
        "verification_sha256",
        "zero_inference_preflight_verification",
    )
    for field in ("production_eligible", "legal_conclusion_made"):
        if verification.get(field) is not False:
            raise ValueError(f"verification boundary changed: {field}")
    checks = verification.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("zero-inference verification checks are incomplete")
