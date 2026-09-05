"""Fail-closed repair gate for the second Lumi AI-assisted SFT candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from difflib import SequenceMatcher
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

PLAN_SCHEMA_VERSION = "lumi.ai-assisted-sft-pilot.v0.2"
ROW_SCHEMA_VERSION = "lumi.ai-assisted-sft-row.v0.2"
AUDIT_SCHEMA_VERSION = "lumi.ai-assisted-sft-audit.v0.2"
MANIFEST_SCHEMA_VERSION = "lumi.ai-assisted-sft-manifest.v0.2"
EVIDENCE_SCHEMA_VERSION = "lumi.ai-assisted-sft-candidate-evidence.v0.2"

EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "external_model_api_spend_authorized": False,
    "human_authorship_claim_authorized": False,
    "model_generated_training_text_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
EXPECTED_QUALITY_GATES = {
    "assistant_response_exact_cross_split_overlap_max": 0,
    "assistant_response_normalized_cross_split_overlap_max": 0,
    "context_required_term_coverage_minimum": 1.0,
    "identity_branding_on_capability_records_max": 0,
    "near_duplicate_cross_split_pairs_max": 0,
    "near_duplicate_exempt_skills": [
        "calculator_tool_routing",
        "simple_arithmetic_model_answer",
    ],
    "near_duplicate_similarity_threshold": 0.9,
    "persistent_memory_claim_records_max": 0,
    "template_family_cross_split_overlap_max": 0,
    "unique_assistant_response_ratio_by_skill_minimum": 1.0,
}
EXPECTED_OWNER_GATE = {
    "dataset_level_signoff_required": True,
    "independent_second_human_required": False,
    "per_row_human_review_required": False,
    "sample_records": 32,
    "sampling": "three_train_language_coverage_and_one_validation_hash_ranked_v0.2",
    "status": "pending",
}
EXPECTED_SOURCE = {
    "dataset_id": "Afnan-Technologies-Inc/lumi-ai-assisted-sft-v0.2",
    "license_id": "lumi-owned",
    "source_id": "lumi_ai_assisted_sft_v0_2",
}

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=])",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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


def load_ai_assisted_v2_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the replacement candidate plan and its rejected-predecessor binding."""

    plan_path = Path(path)
    with plan_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("AI-assisted v0.2 plan must be an object")
    if value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted v0.2 plan schema")
    if value.get("status") != "candidate_regeneration_authorized_owner_spot_check_pending":
        raise ValueError("AI-assisted v0.2 plan status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("AI-assisted v0.2 authorization boundary changed")
    if value.get("quality_gates") != EXPECTED_QUALITY_GATES:
        raise ValueError("AI-assisted v0.2 quality gates changed")
    if value.get("owner_gate") != EXPECTED_OWNER_GATE:
        raise ValueError("AI-assisted v0.2 owner gate changed")
    if value.get("source_binding") != EXPECTED_SOURCE:
        raise ValueError("AI-assisted v0.2 source binding changed")

    root = Path(repository_root)
    coverage = value.get("coverage_binding")
    if not isinstance(coverage, Mapping):
        raise ValueError("AI-assisted v0.2 coverage binding is missing")
    if coverage.get("path") != "configs/lumi-4b-first-party-authoring-pilot-v0.1.json":
        raise ValueError("AI-assisted v0.2 coverage path changed")
    if coverage.get("use") != (
        "record_ids_skill_language_and_split_counts_only_not_human_authorship_policy"
    ):
        raise ValueError("AI-assisted v0.2 coverage use boundary changed")
    coverage_path = root / str(coverage["path"])
    if file_sha256(coverage_path) != coverage.get("sha256"):
        raise ValueError("AI-assisted v0.2 coverage binding changed")
    coverage_plan = load_first_party_authoring_plan(coverage_path)

    predecessor = value.get("predecessor_rejection")
    if not isinstance(predecessor, Mapping):
        raise ValueError("AI-assisted v0.2 predecessor rejection is missing")
    if predecessor.get("candidate_source_revision") != (
        "0456f47ceaa1b462ee1a5f48fb7635e006e7e09295d01ad1f569185a5e04b10e"
    ):
        raise ValueError("AI-assisted v0.2 predecessor revision changed")
    if predecessor.get("required_outcome") != "owner_spot_check_rejected":
        raise ValueError("AI-assisted v0.2 predecessor outcome changed")
    receipt_path = root / _required_string(
        predecessor.get("owner_review_receipt_path"),
        "predecessor_rejection.owner_review_receipt_path",
    )
    if file_sha256(receipt_path) != predecessor.get("owner_review_receipt_sha256"):
        raise ValueError("AI-assisted v0.2 owner rejection receipt changed")
    with receipt_path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    if (
        receipt.get("status") != "owner_spot_check_rejected"
        or receipt.get("production_eligible") is not False
        or receipt.get("raw_training_text_retained") is not False
    ):
        raise ValueError("AI-assisted v0.2 owner rejection receipt is invalid")

    generation = value.get("generation_policy")
    if not isinstance(generation, Mapping):
        raise ValueError("AI-assisted v0.2 generation policy is missing")
    if generation.get("method") != (
        "ai_generated_context_grounded_split_family_isolated_v0.2"
    ):
        raise ValueError("AI-assisted v0.2 generation method changed")
    if generation.get("allowed_generator_ids") != ["openai-codex-current-session"]:
        raise ValueError("AI-assisted v0.2 generator allowlist changed")
    if generation.get("generator_outputs_must_not_be_claimed_human_written") is not True:
        raise ValueError("AI-assisted v0.2 human-authorship boundary changed")
    if generation.get("rights_confirmation_required") is not True:
        raise ValueError("AI-assisted v0.2 rights boundary changed")

    review = value.get("automated_review")
    if not isinstance(review, Mapping):
        raise ValueError("AI-assisted v0.2 automated review is missing")
    score = review.get("minimum_quality_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("AI-assisted v0.2 minimum quality score is invalid")
    if not 0.85 <= float(score) <= 1:
        raise ValueError("AI-assisted v0.2 minimum quality score is outside bounds")
    _required_string(review.get("reviewer_id"), "automated_review.reviewer_id")
    return value, coverage_plan


def _assistant_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )


def _all_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _normalized(value: str) -> str:
    return normalize_for_matching(value)


def _required_terms(evidence: Mapping[str, Any], assistant_text: str) -> tuple[str, ...]:
    raw_terms = evidence.get("required_response_terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("AI-assisted v0.2 required response terms are missing")
    if any(not isinstance(term, str) or not term.strip() for term in raw_terms):
        raise ValueError("AI-assisted v0.2 required response terms are invalid")
    terms = tuple(raw_terms)
    normalized_response = _normalized(assistant_text)
    for term in terms:
        if _normalized(term) not in normalized_response:
            raise ValueError("AI-assisted v0.2 response ignores a required context term")
    return terms


def _validate_task_evidence(
    row: Mapping[str, Any],
    messages: list[Mapping[str, str]],
) -> None:
    evidence = row.get("task_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("AI-assisted v0.2 task evidence is missing")
    skill_id = str(row["skill_id"])
    assistant_text = _assistant_text(messages)
    required_terms = _required_terms(evidence, assistant_text)

    if _PERSISTENT_MEMORY.search(assistant_text):
        raise ValueError("AI-assisted v0.2 response makes a persistent-memory claim")

    if skill_id == "greeting_and_small_talk":
        if "Afnan Technologies" in assistant_text:
            raise ValueError("AI-assisted v0.2 greeting repeats the company identity")
    elif skill_id in {
        "typo_recovery_banglish",
        "typo_recovery_bengali_script",
        "typo_recovery_english",
    }:
        if evidence.get("intent_recovered") is not True:
            raise ValueError("AI-assisted v0.2 typo intent evidence is missing")
        if evidence.get("modifier_grounded") is not True or len(required_terms) < 2:
            raise ValueError("AI-assisted v0.2 typo modifier evidence is incomplete")
    elif skill_id == "simple_arithmetic_model_answer":
        expected = evidence.get("expected_integer")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("AI-assisted v0.2 arithmetic evidence is invalid")
        if parse_single_integer_answer(messages[-1]["content"]) != expected:
            raise ValueError("AI-assisted v0.2 arithmetic answer is incorrect")
    elif skill_id == "calculator_tool_routing":
        if [message["role"] for message in messages] != [
            "user",
            "tool",
            "tool_result",
            "assistant",
        ]:
            raise ValueError("AI-assisted v0.2 calculator trajectory is invalid")
        route_kind = evidence.get("route_kind")
        if route_kind == "success":
            expected = evidence.get("expected_result")
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise ValueError("AI-assisted v0.2 calculator result is invalid")
            expected_text = str(expected)
            if expected_text not in messages[2]["content"] or expected_text not in messages[3][
                "content"
            ]:
                raise ValueError("AI-assisted v0.2 calculator result does not match")
        elif route_kind == "tool_error":
            error_code = _required_string(evidence.get("error_code"), "error_code")
            if error_code not in messages[2]["content"]:
                raise ValueError("AI-assisted v0.2 calculator error evidence does not match")
            if evidence.get("fabricated_result") is not False:
                raise ValueError("AI-assisted v0.2 calculator error fabricates a result")
        else:
            raise ValueError("AI-assisted v0.2 calculator route kind is invalid")
    elif skill_id == "multi_turn_context_and_correction":
        if len(messages) < 4 or evidence.get("context_retained") is not True:
            raise ValueError("AI-assisted v0.2 multi-turn context evidence is missing")
        if evidence.get("context_scope_explicit") is not True:
            raise ValueError("AI-assisted v0.2 context scope evidence is missing")
    elif skill_id == "identity_and_capability_boundaries":
        if evidence.get("verified_claims_only") is not True:
            raise ValueError("AI-assisted v0.2 identity evidence is missing")
        response_kind = evidence.get("response_kind")
        if response_kind not in {"identity", "developer", "capability", "boundary"}:
            raise ValueError("AI-assisted v0.2 identity response kind is invalid")
        if response_kind in {"identity", "developer"}:
            if "Lumi" not in assistant_text or "Afnan Technologies Inc." not in assistant_text:
                raise ValueError("AI-assisted v0.2 canonical identity is missing")
        elif "Afnan Technologies" in assistant_text:
            raise ValueError("AI-assisted v0.2 capability answer adds irrelevant branding")
    else:  # pragma: no cover - coverage is closed by the plan
        raise ValueError(f"unsupported AI-assisted v0.2 skill: {skill_id}")


def validate_ai_assisted_v2_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one private generated row without claiming human review."""

    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted v0.2 row schema")
    for field in ("record_id", "skill_id", "split", "task_category", "language"):
        if row.get(field) != expected.get(field):
            raise ValueError(f"AI-assisted v0.2 immutable metadata changed: {field}")

    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("AI-assisted v0.2 messages must be an array")
    messages: list[Mapping[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping) or set(message) != {"content", "role"}:
            raise ValueError("AI-assisted v0.2 message fields changed")
        if not isinstance(message.get("content"), str) or not isinstance(
            message.get("role"), str
        ):
            raise ValueError("AI-assisted v0.2 message types are invalid")
        messages.append(message)
    validate_message_sequence(messages)

    combined_text = _all_text(messages)
    if _EMAIL.search(combined_text) or _PHONE.search(combined_text) or _SECRET.search(
        combined_text
    ):
        raise ValueError("AI-assisted v0.2 row contains PII or secret-like text")
    language = row["language"]
    contains_bengali = _BENGALI.search(combined_text) is not None
    if language == "bn" and not contains_bengali:
        raise ValueError("AI-assisted v0.2 Bengali row lacks Bengali script")
    if language == "en" and contains_bengali:
        raise ValueError("AI-assisted v0.2 English row contains Bengali script")

    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("AI-assisted v0.2 generation evidence is missing")
    if generation.get("generator_id") not in plan["generation_policy"][
        "allowed_generator_ids"
    ]:
        raise ValueError("AI-assisted v0.2 generator is not allowed")
    if generation.get("method") != plan["generation_policy"]["method"]:
        raise ValueError("AI-assisted v0.2 generation method changed")
    if generation.get("human_written_from_scratch") is not False:
        raise ValueError("AI-assisted v0.2 row may not claim human authorship")
    if generation.get("rights_confirmed") is not True:
        raise ValueError("AI-assisted v0.2 rights evidence is missing")
    _required_string(generation.get("generated_at"), "generation.generated_at")
    family_id = _required_string(
        generation.get("template_family_id"), "generation.template_family_id"
    )
    if _FAMILY_ID.fullmatch(family_id) is None:
        raise ValueError("AI-assisted v0.2 template family ID is invalid")
    if not family_id.startswith(f"{row['skill_id']}:{row['split']}:"):
        raise ValueError("AI-assisted v0.2 template family is not split-isolated")

    review = row.get("automated_review")
    if not isinstance(review, Mapping):
        raise ValueError("AI-assisted v0.2 automated review is missing")
    if review.get("reviewer_id") != plan["automated_review"]["reviewer_id"]:
        raise ValueError("AI-assisted v0.2 automated reviewer changed")
    for field in (
        "context_grounding_passed",
        "identity_tool_claims_passed",
        "language_mode_passed",
        "message_sequence_passed",
        "pii_secret_scan_passed",
        "safety_passed",
        "task_correctness_passed",
    ):
        if review.get(field) is not True:
            raise ValueError(f"AI-assisted v0.2 automated review failed: {field}")
    score = review.get("quality_score")
    minimum = float(plan["automated_review"]["minimum_quality_score"])
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not minimum <= float(score) <= 1
    ):
        raise ValueError("AI-assisted v0.2 quality score is invalid")

    if row.get("owner_review") != {
        "accepted": None,
        "reviewed_at": "",
        "reviewer_id": "",
        "status": "pending",
    }:
        raise ValueError("AI-assisted v0.2 row may not claim owner approval")
    _validate_task_evidence(row, messages)
    return json.loads(json.dumps(row, ensure_ascii=False))


def candidate_source_revision(rows: list[Mapping[str, Any]]) -> str:
    """Return the content revision of exact ordered private v0.2 source rows."""

    return _fingerprint(rows)


def _response(row: Mapping[str, Any]) -> str:
    return _assistant_text(list(row["messages"]))


def audit_ai_assisted_v2_rows(
    rows: list[Mapping[str, Any]],
    *,
    coverage_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit all rows and fail on response leakage or context-insensitive templates."""

    expected_by_id = expected_candidate_metadata(coverage_plan)
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("AI-assisted v0.2 row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_by_id:
            raise ValueError("AI-assisted v0.2 rows contain an unknown record ID")
        if record_id in observed_by_id:
            raise ValueError("AI-assisted v0.2 rows contain a duplicate record ID")
        observed_by_id[record_id] = row
    if set(observed_by_id) != set(expected_by_id):
        raise ValueError("AI-assisted v0.2 rows do not contain the exact planned IDs")

    validated: list[dict[str, Any]] = []
    conversation_hashes: set[str] = set()
    skill_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for record_id in sorted(expected_by_id):
        row = validate_ai_assisted_v2_row(
            observed_by_id[record_id],
            expected=expected_by_id[record_id],
            plan=plan,
        )
        conversation = [
            {"content": _normalized(message["content"]), "role": message["role"]}
            for message in row["messages"]
        ]
        conversation_hash = _fingerprint(conversation)
        if conversation_hash in conversation_hashes:
            raise ValueError("AI-assisted v0.2 contains a normalized duplicate conversation")
        conversation_hashes.add(conversation_hash)
        validated.append(row)
        skill_counts[row["skill_id"]] += 1
        split_counts[row["split"]] += 1
        language_counts[row["language"]] += 1

    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        by_skill.setdefault(row["skill_id"], []).append(row)

    exact_cross_split_overlap = 0
    normalized_cross_split_overlap = 0
    family_cross_split_overlap = 0
    near_duplicate_pairs = 0
    response_metrics: dict[str, Any] = {}
    threshold = float(plan["quality_gates"]["near_duplicate_similarity_threshold"])
    exempt = set(plan["quality_gates"]["near_duplicate_exempt_skills"])
    minimum_ratio = float(
        plan["quality_gates"]["unique_assistant_response_ratio_by_skill_minimum"]
    )

    for skill_id, skill_rows in sorted(by_skill.items()):
        train_rows = [row for row in skill_rows if row["split"] == "train"]
        validation_rows = [row for row in skill_rows if row["split"] == "validation"]
        responses = [_response(row) for row in skill_rows]
        normalized_responses = [_normalized(response) for response in responses]
        unique_ratio = len(set(normalized_responses)) / len(normalized_responses)
        if unique_ratio < minimum_ratio:
            raise ValueError(
                f"AI-assisted v0.2 response diversity failed for {skill_id}: {unique_ratio}"
            )

        train_exact = {_response(row) for row in train_rows}
        validation_exact = {_response(row) for row in validation_rows}
        exact_overlap = len(train_exact & validation_exact)
        exact_cross_split_overlap += exact_overlap

        train_normalized = {_normalized(_response(row)) for row in train_rows}
        validation_normalized = {_normalized(_response(row)) for row in validation_rows}
        normalized_overlap = len(train_normalized & validation_normalized)
        normalized_cross_split_overlap += normalized_overlap

        train_families = {row["generation"]["template_family_id"] for row in train_rows}
        validation_families = {
            row["generation"]["template_family_id"] for row in validation_rows
        }
        family_overlap = len(train_families & validation_families)
        family_cross_split_overlap += family_overlap

        skill_near_pairs = 0
        if skill_id not in exempt:
            for train_response in train_normalized:
                for validation_response in validation_normalized:
                    if train_response == validation_response:
                        continue
                    if SequenceMatcher(
                        None,
                        train_response,
                        validation_response,
                        autojunk=False,
                    ).ratio() >= threshold:
                        skill_near_pairs += 1
        near_duplicate_pairs += skill_near_pairs
        response_metrics[skill_id] = {
            "cross_split_exact_overlap": exact_overlap,
            "cross_split_near_duplicate_pairs": skill_near_pairs,
            "cross_split_normalized_overlap": normalized_overlap,
            "records": len(skill_rows),
            "template_family_cross_split_overlap": family_overlap,
            "unique_assistant_response_ratio": unique_ratio,
        }

    gates = plan["quality_gates"]
    if exact_cross_split_overlap > gates[
        "assistant_response_exact_cross_split_overlap_max"
    ]:
        raise ValueError("AI-assisted v0.2 has exact train-validation response leakage")
    if normalized_cross_split_overlap > gates[
        "assistant_response_normalized_cross_split_overlap_max"
    ]:
        raise ValueError("AI-assisted v0.2 has normalized train-validation response leakage")
    if family_cross_split_overlap > gates["template_family_cross_split_overlap_max"]:
        raise ValueError("AI-assisted v0.2 has train-validation template-family leakage")
    if near_duplicate_pairs > gates["near_duplicate_cross_split_pairs_max"]:
        raise ValueError("AI-assisted v0.2 has near-duplicate train-validation responses")

    report: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "automated_review_passed_records": len(validated),
        "candidate_source_revision": candidate_source_revision(rows),
        "language_counts": dict(sorted(language_counts.items())),
        "normalized_unique_conversations": len(conversation_hashes),
        "owner_review": {
            "complete": False,
            "sample_records": plan["owner_gate"]["sample_records"],
            "status": "pending",
        },
        "plan_sha256": _fingerprint(plan),
        "production_eligible": False,
        "quality_gate_results": {
            "assistant_response_exact_cross_split_overlap": exact_cross_split_overlap,
            "assistant_response_normalized_cross_split_overlap": (
                normalized_cross_split_overlap
            ),
            "context_required_term_coverage": 1.0,
            "identity_branding_on_capability_records": 0,
            "near_duplicate_cross_split_pairs": near_duplicate_pairs,
            "persistent_memory_claim_records": 0,
            "template_family_cross_split_overlap": family_cross_split_overlap,
        },
        "raw_training_text_retained": False,
        "records": len(validated),
        "response_metrics_by_skill": response_metrics,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "skill_counts": dict(sorted(skill_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
    report["report_sha256"] = _fingerprint(report)
    return report


def select_owner_review_sample_v2(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select three language-diverse train rows and one hash-ranked validation row."""

    by_skill: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(str(row["skill_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for skill_id in sorted(by_skill):
        train = sorted(
            (row for row in by_skill[skill_id] if row["split"] == "train"),
            key=lambda row: _fingerprint(row["record_id"]),
        )
        validation = sorted(
            (row for row in by_skill[skill_id] if row["split"] == "validation"),
            key=lambda row: _fingerprint(row["record_id"]),
        )
        if len(train) < 3 or not validation:
            raise ValueError("AI-assisted v0.2 owner sample coverage is incomplete")
        train_selected: list[Mapping[str, Any]] = []
        seen_languages: set[str] = set()
        for row in train:
            if row["language"] not in seen_languages:
                train_selected.append(row)
                seen_languages.add(row["language"])
            if len(train_selected) == 3:
                break
        for row in train:
            if len(train_selected) == 3:
                break
            if row not in train_selected:
                train_selected.append(row)
        selected.extend(
            json.loads(json.dumps(row, ensure_ascii=False))
            for row in (*train_selected, validation[0])
        )
    if len(selected) != 32:
        raise AssertionError("AI-assisted v0.2 owner sample size changed")
    return selected


def to_sft_candidate_records_v2(
    rows: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_revision: str,
) -> list[dict[str, Any]]:
    """Transform validated v0.2 rows into non-production SFT records."""

    source = plan["source_binding"]
    transform_sha256 = _fingerprint(
        {"schema": "lumi.ai-assisted-to-sft.v0.2", "source_revision": source_revision}
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        messages = json.loads(json.dumps(row["messages"], ensure_ascii=False))
        records.append(
            {
                "language": row["language"],
                "messages": messages,
                "provenance": {
                    "consent_basis": "first_party_rights_declaration",
                    "content_sha256": _fingerprint(messages),
                    "generation_method": "ai_generated",
                    "generator_id": row["generation"]["generator_id"],
                    "human_review_status": "owner_spot_check_pending",
                    "transform_sha256": transform_sha256,
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_method": "deterministic_v0.2_gates_owner_pending",
                    "review_status": "accepted",
                    "score": row["automated_review"]["quality_score"],
                    "secret_scan": "passed",
                },
                "record_id": row["record_id"],
                "safety": {
                    "policy_version": "lumi-safety-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": source["dataset_id"],
                    "document_id": row["record_id"],
                    "license_id": source["license_id"],
                    "revision": source_revision,
                    "source_id": source["source_id"],
                    "uri": f"private://{source['dataset_id']}/{row['record_id']}",
                },
                "split": row["split"],
                "task_category": row["task_category"],
            }
        )
    return records


def build_ai_assisted_v2_manifest(
    *,
    audit: Mapping[str, Any],
    candidate_records: list[Mapping[str, Any]],
    owner_sample: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build raw-text-free evidence for a v0.2 candidate awaiting owner review."""

    manifest: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "audit_report_sha256": audit["report_sha256"],
        "candidate_records": len(candidate_records),
        "candidate_sft_jsonl_sha256": _fingerprint(candidate_records),
        "candidate_source_revision": audit["candidate_source_revision"],
        "owner_review_complete": False,
        "owner_sample_records": len(owner_sample),
        "owner_sample_sha256": _fingerprint(owner_sample),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "machine_validated_v0.2_candidate_waiting_for_owner_spot_check",
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest


def load_ai_assisted_v2_evidence(path: str | Path) -> dict[str, Any]:
    """Validate the raw-text-free v0.2 bridge used by readiness planning."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted v0.2 evidence schema")
    if value.get("status") != "machine_validated_v0.2_candidate_waiting_for_owner_spot_check":
        raise ValueError("AI-assisted v0.2 evidence status changed")
    if value.get("production_eligible") is not False:
        raise ValueError("AI-assisted v0.2 evidence may not claim production eligibility")
    if value.get("authorization") != {
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("AI-assisted v0.2 evidence authorization changed")
    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("AI-assisted v0.2 evidence candidate is missing")
    for field in (
        "automated_review_passed_records",
        "normalized_unique_conversations",
        "records",
        "sft_filter_accepted_records",
    ):
        if candidate.get(field) != 320:
            raise ValueError(f"AI-assisted v0.2 evidence count changed: {field}")
    if candidate.get("sft_filter_rejected_records") != 0:
        raise ValueError("AI-assisted v0.2 evidence contains filter rejections")
    quality = candidate.get("quality_gate_results")
    if not isinstance(quality, Mapping) or any(
        quality.get(field) != expected
        for field, expected in {
            "assistant_response_exact_cross_split_overlap": 0,
            "assistant_response_normalized_cross_split_overlap": 0,
            "identity_branding_on_capability_records": 0,
            "near_duplicate_cross_split_pairs": 0,
            "persistent_memory_claim_records": 0,
            "template_family_cross_split_overlap": 0,
        }.items()
    ):
        raise ValueError("AI-assisted v0.2 evidence quality gates are incomplete")
    for field in (
        "candidate_manifest_sha256",
        "candidate_source_revision",
        "candidate_sft_jsonl_sha256",
        "machine_audit_report_sha256",
        "sft_filter_report_sha256",
    ):
        if not isinstance(candidate.get(field), str) or _SHA256.fullmatch(
            candidate[field]
        ) is None:
            raise ValueError(f"AI-assisted v0.2 evidence hash is invalid: {field}")
    owner = value.get("owner_gate")
    if owner != {
        "complete": False,
        "sample_records": 32,
        "sample_sha256": owner.get("sample_sha256") if isinstance(owner, Mapping) else None,
        "status": "pending",
    }:
        raise ValueError("AI-assisted v0.2 owner gate changed")
    if not isinstance(owner.get("sample_sha256"), str) or _SHA256.fullmatch(
        owner["sample_sha256"]
    ) is None:
        raise ValueError("AI-assisted v0.2 owner sample hash is invalid")
    private = value.get("private_artifact")
    if (
        not isinstance(private, Mapping)
        or private.get("raw_training_text_in_this_evidence") is not False
    ):
        raise ValueError("AI-assisted v0.2 private artifact boundary changed")
    _required_string(private.get("path"), "private_artifact.path")
    return value
