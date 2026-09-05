"""Raw-text-free diagnosis of OpenR1 math/phone-pattern collisions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_data as sft_data

OPENR1_PII_PLAN_SCHEMA_VERSION = "lumi.sft-openr1-pii-diagnostic-plan.v0.1"
OPENR1_PII_REPORT_SCHEMA_VERSION = "lumi.sft-openr1-pii-diagnostic-report.v0.1"
EXPECTED_AUTHORIZATION = {
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "global_pii_filter_override_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_pinned_openr1_slice_raw_text_free_pii_diagnostic_only",
}
EXPECTED_SOURCE = {
    "dataset_id": "open-r1/OpenR1-Math-220k",
    "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "source_id": "openr1_math_220k",
}
EXPECTED_LIMITS = {
    "expected_selected_records": 18,
    "expected_source_filter_pii_rejections": 6,
    "maximum_raw_rows": 24,
}
EXPECTED_OUTPUT = {
    "contains_raw_training_text": False,
    "global_filter_modified": False,
    "production_promotion_permitted": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "diagnostic_implementation",
        "diagnostic_runner",
        "raw_rows",
        "source_filter_implementation",
        "verified_trace_audit_evidence",
        "verified_trace_implementation",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTACT_KEYWORD = re.compile(
    r"\b(?:call|contact|fax|mobile|phone|sms|tel\.?|telephone|whatsapp)\b",
    re.IGNORECASE,
)
_MATH_CONTEXT = re.compile(r"[=+*/^_{}\\]|[$≈≤≥∑√]|\d\s*[-–]\s*\d")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def load_openr1_pii_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact local-only PII diagnostic plan and verify every binding."""

    root = Path(repository_root).resolve()
    workspace = root.parent
    plan = _load_json(path)
    if plan.get("schema_version") != OPENR1_PII_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 PII diagnostic plan")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenR1 PII diagnostic authorization changed")
    if plan.get("source") != EXPECTED_SOURCE:
        raise ValueError("OpenR1 PII diagnostic source changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenR1 PII diagnostic limits changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("OpenR1 PII diagnostic output boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenR1 PII diagnostic binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenR1 PII binding must be an object: {name}")
        relative_path = binding.get("path")
        expected_sha256 = _require_sha256(
            binding.get("sha256"),
            f"bindings.{name}.sha256",
        )
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"OpenR1 PII binding path is invalid: {name}")
        resolved = (root / relative_path).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"OpenR1 PII binding escapes workspace: {name}")
        if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"OpenR1 PII binding changed: {name}")
    return plan


def diagnose_openr1_pii_patterns(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify detector hits without returning source text or matched substrings."""

    detector_record_counts: Counter[str] = Counter()
    phone_matches_by_role: Counter[str] = Counter()
    phone_match_count = 0
    phone_matches_in_math_context = 0
    phone_matches_outside_math_context = 0
    contact_keyword_records = 0
    likely_math_numeric_records = 0
    unresolved_phone_like_records = 0
    records_with_any_detector = 0

    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        text_parts = [
            message.get("content")
            for message in messages
            if isinstance(message, Mapping) and isinstance(message.get("content"), str)
        ]
        combined = "\n".join(text_parts)
        detector_hits = {
            "aws_access_key": sft_data._AWS_ACCESS_KEY.search(combined) is not None,
            "email": sft_data._EMAIL.search(combined) is not None,
            "phone": sft_data._PHONE.search(combined) is not None,
            "private_key": sft_data._PRIVATE_KEY.search(combined) is not None,
            "token_assignment": sft_data._TOKEN_ASSIGNMENT.search(combined) is not None,
        }
        if any(detector_hits.values()):
            records_with_any_detector += 1
        for detector, hit in detector_hits.items():
            if hit:
                detector_record_counts[detector] += 1
        if not detector_hits["phone"]:
            continue

        has_contact_keyword = _CONTACT_KEYWORD.search(combined) is not None
        if has_contact_keyword:
            contact_keyword_records += 1
        record_phone_matches = 0
        record_math_matches = 0
        record_non_assistant_matches = 0
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            for line in content.splitlines():
                matches = list(sft_data._PHONE.finditer(line))
                if not matches:
                    continue
                count = len(matches)
                record_phone_matches += count
                phone_match_count += count
                phone_matches_by_role[role] += count
                if role != "assistant":
                    record_non_assistant_matches += count
                if _MATH_CONTEXT.search(line) is not None:
                    record_math_matches += count
                    phone_matches_in_math_context += count
                else:
                    phone_matches_outside_math_context += count
        if (
            record_phone_matches > 0
            and record_phone_matches == record_math_matches
            and record_non_assistant_matches == 0
            and not has_contact_keyword
            and not any(hit for name, hit in detector_hits.items() if name != "phone")
        ):
            likely_math_numeric_records += 1
        else:
            unresolved_phone_like_records += 1

    return {
        "contact_keyword_records": contact_keyword_records,
        "detector_record_counts": dict(sorted(detector_record_counts.items())),
        "detector_signatures": {
            "phone_regex_sha256": hashlib.sha256(
                sft_data._PHONE.pattern.encode("utf-8")
            ).hexdigest(),
            "source_filter_implementation": "lumi.posttraining_data._scan_sensitive_content",
        },
        "likely_math_numeric_false_positive_records": likely_math_numeric_records,
        "phone_match_count": phone_match_count,
        "phone_matches_by_role": dict(sorted(phone_matches_by_role.items())),
        "phone_matches_in_math_context": phone_matches_in_math_context,
        "phone_matches_outside_math_context": phone_matches_outside_math_context,
        "records": len(records),
        "records_with_any_sensitive_detector": records_with_any_detector,
        "raw_text_retained": False,
        "review_outcome": (
            "likely_math_numeric_false_positives_not_automatically_overridden"
            if likely_math_numeric_records and unresolved_phone_like_records == 0
            else "phone_like_matches_require_manual_resolution"
        ),
        "unresolved_phone_like_records": unresolved_phone_like_records,
    }


def build_openr1_pii_report(
    *,
    diagnostic_metrics: Mapping[str, Any],
    observed_at: str,
    plan_sha256: str,
    raw_rows_sha256: str,
    source_filter_sha256: str,
    verified_trace_evidence_file_sha256: str,
    verified_trace_evidence_internal_sha256: str,
) -> dict[str, Any]:
    """Seal a non-authorizing, raw-text-free PII diagnostic receipt."""

    for name, digest in {
        "plan_sha256": plan_sha256,
        "raw_rows_sha256": raw_rows_sha256,
        "source_filter_sha256": source_filter_sha256,
        "verified_trace_evidence_file_sha256": verified_trace_evidence_file_sha256,
        "verified_trace_evidence_internal_sha256": (
            verified_trace_evidence_internal_sha256
        ),
    }.items():
        _require_sha256(digest, name)
    if diagnostic_metrics.get("raw_text_retained") is not False:
        raise ValueError("OpenR1 PII diagnostic must remain raw-text-free")

    likely = diagnostic_metrics.get("likely_math_numeric_false_positive_records")
    unresolved = diagnostic_metrics.get("unresolved_phone_like_records")
    if not isinstance(likely, int) or not isinstance(unresolved, int):
        raise ValueError("OpenR1 PII diagnostic counts are invalid")
    report = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "plan_sha256": plan_sha256,
            "raw_rows_sha256": raw_rows_sha256,
            "source_filter_sha256": source_filter_sha256,
            "verified_trace_evidence_file_sha256": (
                verified_trace_evidence_file_sha256
            ),
            "verified_trace_evidence_internal_sha256": (
                verified_trace_evidence_internal_sha256
            ),
        },
        "decision": (
            "math_numeric_phone_regex_collision_confirmed_no_filter_override"
            if likely and unresolved == 0
            else "pii_pattern_requires_further_review_no_filter_override"
        ),
        "diagnostic_metrics": dict(diagnostic_metrics),
        "global_pii_filter_modified": False,
        "observed_at": observed_at,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_blockers": [
            "secondary_pii_validation_or_owner_review_not_complete",
            "global_pii_filter_override_not_authorized",
        ],
        "schema_version": OPENR1_PII_REPORT_SCHEMA_VERSION,
    }
    report["report_sha256"] = sha256_value(report)
    return report
