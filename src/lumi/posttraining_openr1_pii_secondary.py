"""Line-scoped, raw-text-free secondary PII validation for OpenR1 math traces."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_data as sft_data
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_openr1_pii import _CONTACT_KEYWORD, _MATH_CONTEXT

PLAN_SCHEMA_VERSION = "lumi.sft-openr1-pii-secondary-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-openr1-pii-secondary-report.v0.1"
EXPECTED_AUTHORIZATION = {
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "global_pii_filter_override_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "secondary_local_diagnostic_authorized": True,
    "purpose": "existing_pinned_openr1_slice_line_scoped_pii_reconciliation_only",
}
EXPECTED_SOURCE = {
    "dataset_id": "open-r1/OpenR1-Math-220k",
    "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "source_id": "openr1_math_220k",
}
EXPECTED_LIMITS = {
    "expected_cross_message_join_artifact_records": 1,
    "expected_global_phone_detector_records": 6,
    "expected_line_scoped_math_false_positive_records": 5,
    "expected_primary_unresolved_records": 2,
    "expected_secondary_unresolved_records": 0,
    "expected_selected_records": 18,
    "maximum_raw_rows": 24,
    "maximum_runtime_minutes": 30,
}
EXPECTED_POLICY = {
    "contact_keyword_scope": "same_line_as_phone_match",
    "cross_message_join_is_not_a_source_text_match": True,
    "global_detector_remains_unchanged": True,
    "math_context_required_for_line_scoped_false_positive": True,
    "other_sensitive_detector_hits_allowed": False,
    "required_match_role": "assistant",
}
EXPECTED_OUTPUT = {
    "contains_raw_training_text": False,
    "global_filter_modified": False,
    "production_promotion_permitted": False,
    "report_path": "evidence/lumi-4b-sft-openr1-pii-secondary-v0.1.json",
    "selected_records_persisted": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "diagnostic_implementation",
        "diagnostic_runner",
        "primary_pii_evidence",
        "raw_rows",
        "source_filter_implementation",
        "verified_trace_audit_evidence",
        "verified_trace_implementation",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_secondary_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("OpenR1 secondary PII binding path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("OpenR1 secondary PII binding escaped the workspace")
    return path


def load_secondary_pii_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 secondary PII plan")
    if plan.get("status") != "authorized_local_secondary_diagnostic_no_override":
        raise ValueError("OpenR1 secondary PII plan status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OpenR1 secondary PII plan date changed")
    for field, expected in (
        ("authorization", EXPECTED_AUTHORIZATION),
        ("source", EXPECTED_SOURCE),
        ("limits", EXPECTED_LIMITS),
        ("policy", EXPECTED_POLICY),
        ("output", EXPECTED_OUTPUT),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"OpenR1 secondary PII {field} changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenR1 secondary PII binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenR1 secondary PII binding invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_secondary_binding(repository_root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"OpenR1 secondary PII binding changed: {name}")
    return plan


def _other_detector_hit(text: str) -> bool:
    return any(
        detector.search(text) is not None
        for detector in (
            sft_data._AWS_ACCESS_KEY,
            sft_data._EMAIL,
            sft_data._PRIVATE_KEY,
            sft_data._TOKEN_ASSIGNMENT,
        )
    )


def diagnose_line_scoped_phone_patterns(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify global phone hits using message and line boundaries without exposing text."""

    classifications: Counter[str] = Counter()
    classified_ids: dict[str, list[str]] = {
        "cross_message_join_artifact": [],
        "line_scoped_math_numeric_false_positive": [],
        "unresolved": [],
    }
    global_phone_records = 0
    line_phone_matches = 0
    phone_matches_by_role: Counter[str] = Counter()
    phone_matches_in_math_context = 0
    same_line_contact_keyword_records = 0
    other_detector_records = 0

    for record in records:
        record_id = record.get("record_id")
        messages = record.get("messages")
        if not isinstance(record_id, str) or not isinstance(messages, list):
            raise ValueError("OpenR1 secondary PII record shape changed")
        if any(
            not isinstance(message, Mapping)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("OpenR1 secondary PII message shape changed")
        combined = "\n".join(str(message["content"]) for message in messages)
        if sft_data._PHONE.search(combined) is None:
            continue
        global_phone_records += 1
        other_hit = _other_detector_hit(combined)
        if other_hit:
            other_detector_records += 1

        line_hits: list[tuple[str, bool, bool, int]] = []
        for message in messages:
            role = str(message["role"])
            for line in str(message["content"]).splitlines():
                matches = list(sft_data._PHONE.finditer(line))
                if not matches:
                    continue
                count = len(matches)
                math_context = _MATH_CONTEXT.search(line) is not None
                contact_keyword = _CONTACT_KEYWORD.search(line) is not None
                line_hits.append((role, math_context, contact_keyword, count))
                line_phone_matches += count
                phone_matches_by_role[role] += count
                if math_context:
                    phone_matches_in_math_context += count

        if not line_hits:
            classification = "cross_message_join_artifact"
        else:
            same_line_contact = any(hit[2] for hit in line_hits)
            if same_line_contact:
                same_line_contact_keyword_records += 1
            safe_math_shape = (
                not other_hit
                and not same_line_contact
                and all(
                    role == "assistant" and math_context
                    for role, math_context, _, _ in line_hits
                )
            )
            classification = (
                "line_scoped_math_numeric_false_positive" if safe_math_shape else "unresolved"
            )
        classifications[classification] += 1
        classified_ids[classification].append(record_id)

    metrics = {
        "classification_counts": dict(sorted(classifications.items())),
        "classification_record_ids_sha256": {
            name: sha256_value(sorted(record_ids))
            for name, record_ids in sorted(classified_ids.items())
        },
        "cross_message_join_artifact_records": classifications[
            "cross_message_join_artifact"
        ],
        "detector_signatures": {
            "phone_regex_sha256": hashlib.sha256(
                sft_data._PHONE.pattern.encode("utf-8")
            ).hexdigest(),
            "secondary_policy_sha256": sha256_value(EXPECTED_POLICY),
        },
        "global_phone_detector_records": global_phone_records,
        "line_phone_match_count": line_phone_matches,
        "line_scoped_math_numeric_false_positive_records": classifications[
            "line_scoped_math_numeric_false_positive"
        ],
        "other_sensitive_detector_records": other_detector_records,
        "phone_matches_by_role": dict(sorted(phone_matches_by_role.items())),
        "phone_matches_in_math_context": phone_matches_in_math_context,
        "raw_text_retained": False,
        "records": len(records),
        "same_line_contact_keyword_records": same_line_contact_keyword_records,
        "unresolved_records": classifications["unresolved"],
    }
    if sum(classifications.values()) != global_phone_records:
        raise ValueError("OpenR1 secondary PII classifications do not reconcile")
    return metrics


def build_secondary_pii_report(
    *,
    metrics: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    primary_pii_receipt_sha256: str,
    verified_trace_receipt_sha256: str,
) -> dict[str, Any]:
    for field, value in (
        ("plan_file_sha256", plan_file_sha256),
        ("primary_pii_receipt_sha256", primary_pii_receipt_sha256),
        ("verified_trace_receipt_sha256", verified_trace_receipt_sha256),
    ):
        _require_sha256(value, field)
    if metrics.get("raw_text_retained") is not False or metrics.get("unresolved_records") != 0:
        raise ValueError("OpenR1 secondary PII result did not resolve the bounded slice")
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "primary_pii_evidence_file_sha256": plan["bindings"]["primary_pii_evidence"][
                "sha256"
            ],
            "primary_pii_receipt_sha256": primary_pii_receipt_sha256,
            "raw_rows_sha256": plan["bindings"]["raw_rows"]["sha256"],
            "source_filter_implementation_sha256": plan["bindings"][
                "source_filter_implementation"
            ]["sha256"],
            "verified_trace_evidence_file_sha256": plan["bindings"][
                "verified_trace_audit_evidence"
            ]["sha256"],
            "verified_trace_receipt_sha256": verified_trace_receipt_sha256,
        },
        "decision": "bounded_slice_false_positives_resolved_full_set_gate_still_required",
        "diagnostic_metrics": dict(metrics),
        "global_pii_filter_modified": False,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": [
            "full_dataset_secondary_pii_validation_not_run",
            "source_specific_exception_policy_not_materialized",
            "full_dataset_4096_token_fit_not_measured",
            "underlying_problem_source_rights_and_attribution_not_resolved_per_row",
            "record_level_quality_review_not_complete",
            "attribution_manifest_not_built",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "local_secondary_pii_diagnostic_passed_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_secondary_pii_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 secondary PII report")
    seal = report.get("report_sha256")
    if not isinstance(seal, str) or sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    ) != seal:
        raise ValueError("OpenR1 secondary PII report seal changed")
    if (
        report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("status") != "local_secondary_pii_diagnostic_passed_nonproduction"
        or report.get("global_pii_filter_modified") is not False
        or report.get("production_eligible") is not False
        or report.get("production_promotions") != 0
        or report.get("raw_text_retained") is not False
        or report.get("diagnostic_metrics", {}).get("unresolved_records") != 0
    ):
        raise ValueError("OpenR1 secondary PII report boundary changed")
