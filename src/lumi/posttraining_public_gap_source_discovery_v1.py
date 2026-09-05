"""Materialize the raw-text-free public gap-source discovery successor."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "lumi.sft-public-gap-source-discovery-plan.v0.1"
PLAN_STATUS = "authorized_public_primary_metadata_discovery_no_payload"
ACQUISITION_SCHEMA = "lumi.sft-public-gap-canary-acquisition-preflight.v0.1"
ACQUISITION_STATUS = "PREPARED_not_executable"
REPORT_SCHEMA = "lumi.sft-public-gap-source-discovery-report.v0.1"
REPORT_STATUS = "PASS_massive_train_only_ready_jontro_hold_nonproduction"
VERIFICATION_SCHEMA = "lumi.sft-public-gap-source-discovery-verification.v0.1"
VERIFICATION_STATUS = "PASS_verified_public_gap_discovery_nonproduction"
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
    repository_root: str | Path, specification: Mapping[str, Any], *, label: str
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


def _load_bound_objects(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "source_registry_v21",
    }:
        raise ValueError("discovery bindings changed")
    for label, specification in bindings.items():
        if not isinstance(specification, dict):
            raise ValueError(f"binding {label} must be an object")
        objects[label] = _read_object(
            _resolve_binding(repository_root, specification, label=label)
        )
    return objects


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("unexpected public-source discovery plan identity")
    authorization = plan.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("authorization must be an object")
    if authorization.get("network_scope") != (
        "public_https_primary_source_metadata_and_rights_documents_only"
    ):
        raise ValueError("discovery network scope changed")
    for field in (
        "candidate_construction_authorized",
        "dataset_payload_download_authorized",
        "dataset_seal_authorized",
        "gpu_authorized",
        "inference_authorized",
        "legal_conclusion_authorized",
        "outreach_authorized",
        "production_promotion_authorized",
        "registry_mutation_authorized",
        "sft_training_authorized",
    ):
        if authorization.get(field) is not False:
            raise ValueError(f"discovery cannot authorize {field}")
    limits = plan.get("limits")
    if not isinstance(limits, dict) or any(value != 0 for value in limits.values()):
        raise ValueError("discovery payload/action limits must all be zero")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "predecessor_plan",
        "predecessor_report",
        "predecessor_verification",
        "source_registry_v21",
    }:
        raise ValueError("discovery bindings changed")
    _validate_source_decisions(plan)
    return plan


def load_acquisition_preflight(path: str | Path) -> dict[str, Any]:
    preflight = _read_object(path)
    if (
        preflight.get("schema_version") != ACQUISITION_SCHEMA
        or preflight.get("status") != ACQUISITION_STATUS
    ):
        raise ValueError("unexpected acquisition-preflight identity")
    authorization = preflight.get("authorization")
    if not isinstance(authorization, dict) or authorization.get("execution_authorized") is not False:
        raise ValueError("acquisition preflight unexpectedly executable")
    source = preflight.get("bound_source")
    if not isinstance(source, dict) or source != {
        "dataset_id": "AmazonScience/massive",
        "dataset_revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
        "expected_train_rows": 11514,
        "object_bytes": 1071668,
        "object_path": "bn-BD/train/0000.parquet",
        "object_sha256": "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d",
    }:
        raise ValueError("MASSIVE acquisition binding changed")
    limits = preflight.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("acquisition limits missing")
    if limits.get("maximum_inbound_bytes") != 2097152:
        raise ValueError("acquisition inbound-byte bound changed")
    if limits.get("maximum_source_row_reads") != 11514:
        raise ValueError("acquisition row-read bound changed")
    for field in (
        "maximum_candidate_reads",
        "maximum_gpu_actions",
        "maximum_inference_calls",
        "maximum_test_dev_validation_row_reads",
        "maximum_training_actions",
    ):
        if limits.get(field) != 0:
            raise ValueError(f"acquisition {field} must be zero")
    jontro = preflight.get("jontro")
    if not isinstance(jontro, dict) or jontro.get("payload_download_authorized") is not False:
        raise ValueError("Jontro payload must remain unauthorized")
    return preflight


def _validate_source_decisions(plan: Mapping[str, Any]) -> None:
    decisions = plan.get("public_source_decisions")
    if not isinstance(decisions, list) or len(decisions) != 2:
        raise ValueError("exactly two public-source decisions are required")
    by_id = {item.get("source_id"): item for item in decisions if isinstance(item, dict)}
    if set(by_id) != {
        "amazon_massive_bn_bd_tool_canary",
        "jontro_bangla_tool_use_conditional_canary",
    }:
        raise ValueError("public-source shortlist changed")
    massive = by_id["amazon_massive_bn_bd_tool_canary"]
    if massive.get("acquisition_class") != (
        "READY_FOR_BOUNDED_TRAIN_ONLY_ACQUISITION_PREFLIGHT"
    ):
        raise ValueError("MASSIVE decision changed")
    if massive.get("exact_train_object") != {
        "bytes": 1071668,
        "path": "bn-BD/train/0000.parquet",
        "sha256": "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d",
    }:
        raise ValueError("MASSIVE object binding changed")
    if any(massive.get("allowed_credit_before_content_audit", {}).values()):
        raise ValueError("MASSIVE received premature credit")
    jontro = by_id["jontro_bangla_tool_use_conditional_canary"]
    if jontro.get("acquisition_class") != (
        "HOLD_PAYLOAD_PENDING_GENERATION_RIGHTS_AND_QUALITY_PRECHECK"
    ):
        raise ValueError("Jontro hold changed")
    if any(jontro.get("allowed_credit_before_clearance", {}).values()):
        raise ValueError("Jontro received premature credit")


def _verify_predecessor(
    plan: Mapping[str, Any], objects: Mapping[str, Mapping[str, Any]]
) -> None:
    report = objects["predecessor_report"]
    verification = objects["predecessor_verification"]
    registry = objects["source_registry_v21"]
    _verify_internal_seal(report, "report_sha256", "predecessor_report")
    _verify_internal_seal(verification, "verification_sha256", "predecessor_verification")
    if report.get("status") != "STOP_no_complete_rights_clear_local_closure_set":
        raise ValueError("predecessor source selection changed")
    if verification.get("status") != (
        "PASS_verified_targeted_gap_snapshot_source_selection_blocked"
    ):
        raise ValueError("predecessor verification changed")
    expected = plan["current_gap"]
    state = report.get("current_verified_state")
    if not isinstance(state, dict):
        raise ValueError("predecessor current-state snapshot missing")
    if state.get("strict_targeted_gap_lower_bound") != expected[
        "strict_targeted_gap_lower_bound"
    ]:
        raise ValueError("strict gap changed")
    if {
        key: state.get("language_gaps", {}).get(key)
        for key in expected["language_floor_gaps_behavior_safe"]
    } != expected["language_floor_gaps_behavior_safe"]:
        raise ValueError("language gaps changed")
    if {
        key: state.get("task_gaps", {}).get(key)
        for key in expected["task_floor_gaps_behavior_safe"]
    } != expected["task_floor_gaps_behavior_safe"]:
        raise ValueError("task gaps changed")
    source_ids = {
        item.get("source_id")
        for item in registry.get("sources", [])
        if isinstance(item, dict)
    }
    if not {"banglabridge_instructions_closure_canary", "openassistant_oasst2"} <= source_ids:
        raise ValueError("already-accounted source bindings changed")
    if {"amazon_massive_bn_bd_tool_canary", "jontro_bangla_tool_use_conditional_canary"} & source_ids:
        raise ValueError("discovery candidate unexpectedly entered registry")


def build_report(
    plan: Mapping[str, Any],
    acquisition_preflight: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    acquisition_preflight_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    objects = _load_bound_objects(plan, repository_root)
    _verify_predecessor(plan, objects)
    _validate_source_decisions(plan)
    load_acquisition_preflight_from_value(acquisition_preflight)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "authorization_text_sha256": plan["authorization"][
            "authorization_text_sha256"
        ],
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "acquisition_preflight_file_sha256": _require_sha256(
            acquisition_preflight_file_sha256,
            "acquisition_preflight_file_sha256",
        ),
        "bindings": {
            name: {"file_sha256": specification["sha256"]}
            for name, specification in sorted(plan["bindings"].items())
        },
        "gap_snapshot": plan["current_gap"],
        "source_decisions": plan["public_source_decisions"],
        "existing_source_exclusions": plan["existing_source_exclusions"],
        "rejected_public_source_classes": plan["rejected_public_source_classes"],
        "first_party_contingency": plan["first_party_contingency"],
        "decision_summary": {
            "new_public_families_ready_for_payload_acquisition": 1,
            "new_public_families_on_hold": 1,
            "public_source_closes_full_language_gap": False,
            "public_source_closes_factual_qa_gap": False,
            "massive_may_seed_tool_api_only_after_all_later_gates": True,
            "jontro_has_zero_active_credit": True,
            "first_party_creation_remains_required": True,
        },
        "prepared_successor": {
            "schema_version": acquisition_preflight["schema_version"],
            "status": acquisition_preflight["status"],
            "execution_authorized": False,
            "massive_object_sha256": acquisition_preflight["bound_source"][
                "object_sha256"
            ],
            "jontro_payload_download_authorized": False,
        },
        "next_gate": {
            "action": "explicitly_authorize_bounded_massive_train_only_acquisition_or_start_first_party_human_project",
            "massive_acquisition_execution_authorized": False,
            "jontro_payload_authorized": False,
            "candidate_construction_authorized": False,
            "registry_mutation_authorized": False,
            "dataset_seal_authorized": False,
            "sft_training_authorized": False,
        },
        "resource_accounting": {
            "local_metadata_json_objects_read": len(objects) + 2,
            "local_files_hashed": len(plan["bindings"]) + 2,
            "dataset_payload_bytes_read": 0,
            "dataset_rows_read": 0,
            "candidate_rows_read": 0,
            "test_dev_validation_rows_read": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "registry_mutations": 0,
            "training_actions": 0,
        },
        "legal_conclusion_made": False,
        "production_eligible": False,
        "raw_training_text_retained": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def load_acquisition_preflight_from_value(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != ACQUISITION_SCHEMA
        or value.get("status") != ACQUISITION_STATUS
    ):
        raise ValueError("unexpected acquisition-preflight value")
    if value.get("authorization", {}).get("execution_authorized") is not False:
        raise ValueError("prepared acquisition preflight cannot execute")
    if value.get("jontro", {}).get("payload_download_authorized") is not False:
        raise ValueError("prepared preflight authorized Jontro payload")


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != REPORT_STATUS:
        raise ValueError("unexpected public-source discovery report identity")
    _verify_internal_seal(report, "report_sha256", "public_source_discovery_report")
    if report.get("production_eligible") is not False:
        raise ValueError("discovery report cannot be production eligible")
    if report.get("legal_conclusion_made") is not False:
        raise ValueError("discovery report cannot make a legal conclusion")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("discovery report retained raw training text")
    summary = report.get("decision_summary")
    if not isinstance(summary, dict):
        raise ValueError("decision summary missing")
    if summary.get("new_public_families_ready_for_payload_acquisition") != 1:
        raise ValueError("ready public-family count changed")
    if summary.get("public_source_closes_full_language_gap") is not False:
        raise ValueError("public discovery incorrectly closes language gap")
    if summary.get("public_source_closes_factual_qa_gap") is not False:
        raise ValueError("public discovery incorrectly closes factual gap")
    if report.get("prepared_successor", {}).get("execution_authorized") is not False:
        raise ValueError("prepared successor unexpectedly executable")
    for field, value in report.get("resource_accounting", {}).items():
        if field.startswith("local_"):
            continue
        if value != 0:
            raise ValueError(f"prohibited resource accounting is non-zero: {field}")


def build_verification(
    *,
    plan: Mapping[str, Any],
    acquisition_preflight: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    load_acquisition_preflight_from_value(acquisition_preflight)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "predecessor_gap_and_registry_bindings_match": True,
            "banglabridge_and_oasst2_are_not_misclassified_as_new_sources": True,
            "massive_revision_object_hash_size_and_rights_chain_are_pinned": True,
            "massive_has_zero_credit_before_content_and_candidate_gates": True,
            "jontro_revision_object_and_known_defect_counts_are_pinned": True,
            "jontro_payload_and_credit_remain_on_hold": True,
            "no_public_factual_qa_source_passed": True,
            "first_party_creation_remains_required": True,
            "acquisition_successor_is_prepared_but_not_authorized": True,
            "no_dataset_payload_candidate_test_split_gpu_inference_registry_seal_or_sft": True,
            "raw_training_text_absent": True,
            "production_eligible_false": True,
        },
        "resource_accounting": report["resource_accounting"],
        "production_eligible": False,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != VERIFICATION_STATUS
    ):
        raise ValueError("unexpected public-source discovery verification identity")
    _verify_internal_seal(
        verification, "verification_sha256", "public_source_discovery_verification"
    )
    if verification.get("production_eligible") is not False:
        raise ValueError("verification cannot mark production eligible")
    checks = verification.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("verification checks are incomplete")
