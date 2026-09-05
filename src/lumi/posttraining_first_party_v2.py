"""Message-native, fail-closed human authoring for Lumi SFT records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import normalize_for_matching
from .posttraining_ai_assisted import expected_candidate_metadata
from .posttraining_capabilities import parse_single_integer_answer
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import (
    canonical_json_bytes,
    file_sha256,
    load_first_party_authoring_plan,
)

PLAN_SCHEMA_VERSION = "lumi.first-party-authoring-plan.v0.2"
ROW_SCHEMA_VERSION = "lumi.first-party-authoring-row.v0.2"
AUDIT_SCHEMA_VERSION = "lumi.first-party-authoring-audit.v0.2"

EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "human_authoring_template_authorized": True,
    "model_generated_training_text_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
EXPECTED_SOURCE = {
    "dataset_id": "Afnan-Technologies-Inc/lumi-owned-sft-v0.2",
    "license_id": "lumi-owned",
    "source_id": "lumi_owned_sft_v0_2",
}
EXPECTED_QUALITY_POLICY = {
    "author_and_reviewer_must_differ": True,
    "capability_answers_must_not_repeat_company_identity": True,
    "context_modifier_grounding_required": True,
    "cross_split_exact_and_normalized_response_overlap_max": 0,
    "human_authorship_without_model_assistance_required": True,
    "minimum_quality_score": 0.85,
    "no_personal_data_or_secrets": True,
    "no_persistent_memory_claims": True,
    "no_proprietary_assistant_transcript_rewriting": True,
    "original_human_authorship_required": True,
    "rights_declaration_required": True,
    "template_family_must_be_split_isolated": True,
    "unique_assistant_response_ratio_by_skill_minimum": 1.0,
}

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=])",
    re.IGNORECASE,
)
_FAMILY_ID = re.compile(r"^[a-z0-9][a-z0-9_:.-]{5,127}$")
_PERSISTENT_MEMORY = re.compile(
    r"(?:\bi(?:'ll| will) remember (?:this|that|it)\b|"
    r"\b(?:saved|stored) (?:this|that|it) (?:to|in) memory\b|"
    r"\balways remember\b|\bmone rakh(?:lam|bo)\b|মনে রাখলাম)",
    re.IGNORECASE,
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_first_party_authoring_v2_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the message-native plan and exact v0.1 coverage metadata binding."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported first-party v0.2 authoring plan")
    if value.get("status") != (
        "bounded_message_native_blank_template_ready_no_authored_records"
    ):
        raise ValueError("first-party v0.2 authoring status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("first-party v0.2 authorization boundary changed")
    if value.get("source_binding") != EXPECTED_SOURCE:
        raise ValueError("first-party v0.2 source binding changed")
    if value.get("quality_policy") != EXPECTED_QUALITY_POLICY:
        raise ValueError("first-party v0.2 quality policy changed")
    if value.get("totals") != {"train_records": 240, "validation_records": 80}:
        raise ValueError("first-party v0.2 record bounds changed")

    binding = value.get("coverage_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("first-party v0.2 coverage binding is missing")
    if binding.get("path") != "configs/lumi-4b-first-party-authoring-pilot-v0.1.json":
        raise ValueError("first-party v0.2 coverage path changed")
    if binding.get("use") != "record_ids_skill_language_split_and_count_contract_only":
        raise ValueError("first-party v0.2 coverage use changed")
    coverage_path = Path(repository_root) / str(binding["path"])
    if file_sha256(coverage_path) != binding.get("sha256"):
        raise ValueError("first-party v0.2 coverage binding changed")
    coverage = load_first_party_authoring_plan(coverage_path)

    contract = value.get("field_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("first-party v0.2 field contract is missing")
    expected_roles = {
        "calculator_tool_routing": ["user", "tool", "tool_result", "assistant"],
        "greeting_and_small_talk": ["user", "assistant"],
        "identity_and_capability_boundaries": ["user", "assistant"],
        "multi_turn_context_and_correction": [
            "user",
            "assistant",
            "user",
            "assistant",
        ],
        "simple_arithmetic_model_answer": ["user", "assistant"],
        "typo_recovery_banglish": ["user", "assistant"],
        "typo_recovery_bengali_script": ["user", "assistant"],
        "typo_recovery_english": ["user", "assistant"],
    }
    if contract.get("role_shapes") != expected_roles:
        raise ValueError("first-party v0.2 message role shapes changed")
    if contract.get("messages_must_start_empty") is not True:
        raise ValueError("first-party v0.2 messages must start empty")
    if contract.get("required_content_fields") != [
        "messages",
        "task_evidence",
        "template_family_id",
    ]:
        raise ValueError("first-party v0.2 content fields changed")
    return value, coverage


def build_blank_authoring_template_v2(
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build exact blank message-native rows without generating training text."""

    metadata = expected_candidate_metadata(coverage)
    rows = []
    for record_id in sorted(metadata):
        item = metadata[record_id]
        rows.append(
            {
                "author_declaration": {
                    "author_id": "",
                    "authored_at": "",
                    "human_written_from_scratch": None,
                    "model_assistance_used": None,
                    "rights_confirmed": None,
                },
                "independent_review": {
                    "accepted": None,
                    "context_grounding_passed": None,
                    "language_naturalness_passed": None,
                    "quality_score": None,
                    "reviewer_id": "",
                    "safety_review_passed": None,
                    "task_correctness_passed": None,
                },
                "language": item["language"],
                "messages": [],
                "record_id": record_id,
                "schema_version": ROW_SCHEMA_VERSION,
                "skill_id": item["skill_id"],
                "split": item["split"],
                "task_category": item["task_category"],
                "task_evidence": {},
                "template_family_id": "",
            }
        )
    if len(rows) != sum(plan["totals"].values()):
        raise AssertionError("first-party v0.2 blank template size changed")
    return rows


def _blank_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("messages") == []
        and row.get("task_evidence") == {}
        and row.get("template_family_id") == ""
        and row.get("author_declaration")
        == {
            "author_id": "",
            "authored_at": "",
            "human_written_from_scratch": None,
            "model_assistance_used": None,
            "rights_confirmed": None,
        }
        and row.get("independent_review")
        == {
            "accepted": None,
            "context_grounding_passed": None,
            "language_naturalness_passed": None,
            "quality_score": None,
            "reviewer_id": "",
            "safety_review_passed": None,
            "task_correctness_passed": None,
        }
    )


def _assistant_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )


def _required_terms(evidence: Mapping[str, Any], assistant_text: str) -> tuple[str, ...]:
    terms = evidence.get("required_response_terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("required_response_terms_missing")
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("required_response_terms_invalid")
    normalized_response = normalize_for_matching(assistant_text)
    for term in terms:
        if normalize_for_matching(term) not in normalized_response:
            raise ValueError("required_response_term_not_grounded")
    return tuple(terms)


def _validate_task(
    row: Mapping[str, Any],
    messages: list[Mapping[str, str]],
) -> None:
    evidence = row.get("task_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("task_evidence_missing")
    skill = str(row["skill_id"])
    assistant = _assistant_text(messages)
    terms = _required_terms(evidence, assistant)
    if _PERSISTENT_MEMORY.search(assistant):
        raise ValueError("persistent_memory_claim")

    if skill == "greeting_and_small_talk":
        if "Afnan Technologies" in assistant:
            raise ValueError("irrelevant_company_identity")
    elif skill.startswith("typo_recovery_"):
        if evidence.get("intent_recovered") is not True:
            raise ValueError("typo_intent_not_recovered")
        if evidence.get("modifier_grounded") is not True or len(terms) < 2:
            raise ValueError("typo_modifier_not_grounded")
    elif skill == "simple_arithmetic_model_answer":
        expected = evidence.get("expected_integer")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("arithmetic_expected_integer_invalid")
        if parse_single_integer_answer(messages[-1]["content"]) != expected:
            raise ValueError("arithmetic_answer_incorrect")
    elif skill == "calculator_tool_routing":
        route = evidence.get("route_kind")
        if route == "success":
            expected = evidence.get("expected_result")
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise ValueError("calculator_expected_result_invalid")
            if str(expected) not in messages[2]["content"] or str(expected) not in assistant:
                raise ValueError("calculator_result_mismatch")
        elif route == "tool_error":
            error_code = _required_string(evidence.get("error_code"), "error_code")
            if error_code not in messages[2]["content"] or error_code not in assistant:
                raise ValueError("calculator_error_mismatch")
            if evidence.get("fabricated_result") is not False:
                raise ValueError("calculator_error_fabricates_result")
        else:
            raise ValueError("calculator_route_kind_invalid")
    elif skill == "multi_turn_context_and_correction":
        if evidence.get("context_retained") is not True:
            raise ValueError("multi_turn_context_not_retained")
        if evidence.get("context_scope_explicit") is not True:
            raise ValueError("multi_turn_scope_not_explicit")
    elif skill == "identity_and_capability_boundaries":
        if evidence.get("verified_claims_only") is not True:
            raise ValueError("identity_claims_not_verified")
        kind = evidence.get("response_kind")
        if kind not in {"identity", "developer", "capability", "boundary"}:
            raise ValueError("identity_response_kind_invalid")
        if kind in {"identity", "developer"}:
            if "Lumi" not in assistant or "Afnan Technologies Inc." not in assistant:
                raise ValueError("canonical_identity_missing")
        elif "Afnan Technologies" in assistant:
            raise ValueError("capability_answer_has_irrelevant_branding")
    else:  # pragma: no cover - coverage is closed
        raise ValueError("unsupported_skill")


def validate_completed_authoring_row_v2(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one genuinely human-written row and independent review evidence."""

    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("row_schema_version")
    for field in ("record_id", "skill_id", "split", "task_category", "language"):
        if row.get(field) != expected.get(field):
            raise ValueError(f"immutable_metadata:{field}")

    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("messages_type")
    messages: list[Mapping[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping) or set(message) != {"content", "role"}:
            raise ValueError("message_fields")
        if not isinstance(message.get("content"), str) or not isinstance(
            message.get("role"), str
        ):
            raise ValueError("message_types")
        messages.append(message)
    validate_message_sequence(messages)
    expected_roles = plan["field_contract"]["role_shapes"][row["skill_id"]]
    if [message["role"] for message in messages] != expected_roles:
        raise ValueError("skill_role_shape")
    for message in messages:
        content = message["content"]
        if _EMAIL.search(content) or _PHONE.search(content) or _SECRET.search(content):
            raise ValueError("pii_or_secret_pattern")

    combined = "\n".join(message["content"] for message in messages)
    has_bengali = _BENGALI.search(combined) is not None
    if row["language"] == "bn" and not has_bengali:
        raise ValueError("bengali_script_missing")
    if row["language"] == "en" and has_bengali:
        raise ValueError("english_contains_bengali_script")

    family = _required_string(row.get("template_family_id"), "template_family_id")
    if _FAMILY_ID.fullmatch(family) is None:
        raise ValueError("template_family_id_invalid")
    if not family.startswith(f"{row['skill_id']}:{row['split']}:"):
        raise ValueError("template_family_not_split_isolated")

    declaration = row.get("author_declaration")
    review = row.get("independent_review")
    if not isinstance(declaration, Mapping) or not isinstance(review, Mapping):
        raise ValueError("human_evidence_missing")
    author_id = _required_string(declaration.get("author_id"), "author_id")
    reviewer_id = _required_string(review.get("reviewer_id"), "reviewer_id")
    _required_string(declaration.get("authored_at"), "authored_at")
    if author_id == reviewer_id:
        raise ValueError("author_reviewer_not_independent")
    if declaration.get("human_written_from_scratch") is not True:
        raise ValueError("human_authorship_not_declared")
    if declaration.get("model_assistance_used") is not False:
        raise ValueError("model_assistance_used_in_human_track")
    if declaration.get("rights_confirmed") is not True:
        raise ValueError("rights_not_confirmed")
    for field in (
        "accepted",
        "context_grounding_passed",
        "language_naturalness_passed",
        "safety_review_passed",
        "task_correctness_passed",
    ):
        if review.get(field) is not True:
            raise ValueError(f"review_failed:{field}")
    score = review.get("quality_score")
    minimum = float(plan["quality_policy"]["minimum_quality_score"])
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not minimum <= float(score) <= 1
    ):
        raise ValueError("quality_score_invalid")

    _validate_task(row, messages)
    return json.loads(json.dumps(row, ensure_ascii=False))


def audit_first_party_authoring_rows_v2(
    rows: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit private human progress without retaining any authored text in the report."""

    expected = expected_candidate_metadata(coverage)
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("first-party v0.2 row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected:
            raise ValueError("first-party v0.2 contains an unknown record ID")
        if record_id in observed:
            raise ValueError("first-party v0.2 contains a duplicate record ID")
        observed[record_id] = row
    if set(observed) != set(expected):
        raise ValueError("first-party v0.2 exact record membership changed")

    accepted_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for record_id in sorted(expected):
        row = observed[record_id]
        if _blank_row(row):
            status_counts["blank"] += 1
            continue
        try:
            accepted_rows.append(
                validate_completed_authoring_row_v2(
                    row,
                    expected=expected[record_id],
                    plan=plan,
                )
            )
        except ValueError as error:
            status_counts["invalid"] += 1
            reason_counts[str(error)] += 1
        else:
            status_counts["accepted"] += 1
    for status in ("accepted", "blank", "invalid"):
        status_counts[status] += 0

    quality_results: dict[str, Any] = {"status": "pending_until_all_rows_accepted"}
    authoring_complete = status_counts["accepted"] == len(expected)
    if authoring_complete:
        by_skill: dict[str, list[dict[str, Any]]] = {}
        for row in accepted_rows:
            by_skill.setdefault(row["skill_id"], []).append(row)
        exact_overlap = 0
        normalized_overlap = 0
        family_overlap = 0
        for skill_rows in by_skill.values():
            responses = [
                normalize_for_matching(_assistant_text(list(row["messages"])))
                for row in skill_rows
            ]
            if len(responses) != len(set(responses)):
                raise ValueError("human_dataset_response_diversity_failed")
            train = [row for row in skill_rows if row["split"] == "train"]
            validation = [row for row in skill_rows if row["split"] == "validation"]
            train_exact = {_assistant_text(list(row["messages"])) for row in train}
            validation_exact = {
                _assistant_text(list(row["messages"])) for row in validation
            }
            exact_overlap += len(train_exact & validation_exact)
            train_normalized = {normalize_for_matching(value) for value in train_exact}
            validation_normalized = {
                normalize_for_matching(value) for value in validation_exact
            }
            normalized_overlap += len(train_normalized & validation_normalized)
            train_families = {row["template_family_id"] for row in train}
            validation_families = {row["template_family_id"] for row in validation}
            family_overlap += len(train_families & validation_families)
        if exact_overlap or normalized_overlap or family_overlap:
            raise ValueError("human_dataset_train_validation_isolation_failed")
        quality_results = {
            "assistant_response_exact_cross_split_overlap": 0,
            "assistant_response_normalized_cross_split_overlap": 0,
            "template_family_cross_split_overlap": 0,
            "unique_assistant_response_ratio_by_skill": 1.0,
        }

    report: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "authoring_complete": authoring_complete,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "plan_sha256": _fingerprint(plan),
        "quality_gate_results": quality_results,
        "raw_training_text_retained": False,
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows_sha256": _fingerprint(rows),
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status_counts": dict(sorted(status_counts.items())),
    }
    report["report_sha256"] = _fingerprint(report)
    return report
