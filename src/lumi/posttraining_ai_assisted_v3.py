"""Strict, honestly labelled AI-assisted Lumi SFT pilot gates."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any

from .data_quality import matching_tokens, normalize_for_matching
from .posttraining_ai_assisted import expected_candidate_metadata
from .posttraining_ai_assisted_v2 import load_ai_assisted_v2_evidence
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import (
    canonical_json_bytes,
    file_sha256,
    load_first_party_authoring_plan,
)
from .posttraining_format import load_chat_template

PLAN_SCHEMA_VERSION = "lumi.ai-assisted-sft-pilot.v0.3"
ROW_SCHEMA_VERSION = "lumi.ai-assisted-sft-row.v0.3"
AUDIT_SCHEMA_VERSION = "lumi.ai-assisted-sft-audit.v0.3"
MANIFEST_SCHEMA_VERSION = "lumi.ai-assisted-sft-candidate-manifest.v0.3"
EVIDENCE_SCHEMA_VERSION = "lumi.ai-assisted-sft-candidate-evidence.v0.3"

EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "external_model_api_spend_authorized": False,
    "human_authorship_claim_authorized": False,
    "model_generated_training_text_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
EXPECTED_SOURCE = {
    "dataset_id": "Afnan-Technologies-Inc/lumi-ai-assisted-sft-v0.3",
    "license_id": "lumi-owned",
    "source_id": "lumi_ai_assisted_sft_v0_3",
}
EXPECTED_LANGUAGE_MODES = {
    "bn": "bengali_script",
    "en": "english",
    "multilingual": "banglish_romanized",
}
EXPECTED_AUTOMATED_CHECKS = [
    "exact_record_membership",
    "message_sequence",
    "final_assistant_response_uniqueness",
    "train_validation_response_isolation",
    "split_independent_semantic_family_isolation",
    "cross_split_prompt_and_response_near_duplicate_detection",
    "final_assistant_context_grounding",
    "explicit_language_mode",
    "canonical_integer_answer",
    "independent_calculator_expression_evaluation",
    "calculator_json_schema",
    "persistent_memory_claim_rejection",
    "capability_branding_relevance",
    "pii_and_secret_patterns",
    "safety_policy",
    "benchmark_decontamination_at_sft_filter",
]
EXPECTED_QUALITY_GATES = {
    "calculator_independent_evaluation_rate_minimum": 1.0,
    "canonical_integer_answer_rate_minimum": 1.0,
    "final_response_exact_cross_split_overlap_max": 0,
    "final_response_normalized_cross_split_overlap_max": 0,
    "final_response_unique_ratio_by_skill_minimum": 1.0,
    "final_term_grounding_rate_minimum": 1.0,
    "identity_branding_on_capability_records_max": 0,
    "near_duplicate_exempt_skills": [
        "calculator_tool_routing",
        "simple_arithmetic_model_answer",
    ],
    "near_duplicate_similarity_threshold": 0.9,
    "persistent_memory_claim_records_max": 0,
    "prompt_near_duplicate_cross_split_pairs_max": 0,
    "response_near_duplicate_cross_split_pairs_max": 0,
    "semantic_family_cross_split_overlap_max": 0,
}
EXPECTED_ROLE_SHAPES = {
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

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_CANONICAL_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_INTEGER_TOKEN = re.compile(r"(?<![\w.])-?(?:0|[1-9][0-9]*)(?!\w|[.][0-9])")
_FAMILY_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*(?::[a-z0-9][a-z0-9_.-]*)+$")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\bsk-[A-Za-z0-9_-]{16,}\b)",
    re.IGNORECASE,
)
_PERSISTENT_MEMORY = re.compile(
    r"\b(?:i(?:'ll| will) remember (?:this|that|it)|saved (?:to|in) (?:my )?memory|"
    r"future chats?|পরের (?:চ্যাট|আলাপে).*মনে রাখব|মেমরিতে.*রেখেছি|"
    r"mone rakhbo|memory-te.*save)\b",
    re.IGNORECASE,
)
_UTC_TIMESTAMP = re.compile(
    r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)
_BANGLISH_MARKERS = {
    "age",
    "ami",
    "ar",
    "atke",
    "banaiche",
    "bolo",
    "chai",
    "dao",
    "daw",
    "diye",
    "dhorchi",
    "ei",
    "ekta",
    "er",
    "eta",
    "gechi",
    "gula",
    "hobe",
    "jai",
    "jonno",
    "kaj",
    "ke",
    "kichu",
    "ki",
    "konta",
    "kor",
    "kora",
    "kore",
    "koro",
    "korbo",
    "lagbe",
    "likh",
    "mama",
    "na",
    "niye",
    "onek",
    "pari",
    "pathale",
    "pathao",
    "rakhba",
    "rakho",
    "shuru",
    "ta",
    "theke",
    "toiri",
    "tumi",
}


class CalculatorExpressionError(ValueError):
    """A deterministic calculator expression rejection with a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AI-assisted v0.3 {name} is required")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_ai_assisted_v3_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the strict v0.3 plan and verify all local predecessor bindings."""

    value = _load_json(Path(path))
    if value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted v0.3 plan")
    if value.get("status") != "review_driven_repair_authorized_owner_signoff_pending":
        raise ValueError("AI-assisted v0.3 plan status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("AI-assisted v0.3 authorization changed")
    if value.get("source_binding") != EXPECTED_SOURCE:
        raise ValueError("AI-assisted v0.3 source binding changed")
    automated = value.get("automated_review")
    if not isinstance(automated, Mapping):
        raise ValueError("AI-assisted v0.3 automated-review policy is missing")
    if automated.get("checks") != EXPECTED_AUTOMATED_CHECKS:
        raise ValueError("AI-assisted v0.3 automated-review checks changed")
    if automated.get("minimum_quality_score") != 0.9:
        raise ValueError("AI-assisted v0.3 minimum quality score changed")
    if automated.get("reviewer_id") != "lumi-deterministic-ai-assisted-guard-v0.3":
        raise ValueError("AI-assisted v0.3 automated reviewer changed")
    if value.get("quality_gates") != EXPECTED_QUALITY_GATES:
        raise ValueError("AI-assisted v0.3 quality gates changed")

    root = Path(repository_root)
    coverage_binding = value.get("coverage_binding")
    if not isinstance(coverage_binding, Mapping):
        raise ValueError("AI-assisted v0.3 coverage binding is missing")
    if coverage_binding.get("path") != (
        "configs/lumi-4b-first-party-authoring-pilot-v0.1.json"
    ):
        raise ValueError("AI-assisted v0.3 coverage path changed")
    if coverage_binding.get("use") != (
        "record_ids_skill_task_split_counts_and_legacy_language_labels_only"
    ):
        raise ValueError("AI-assisted v0.3 coverage-use boundary changed")
    coverage_path = root / str(coverage_binding["path"])
    if file_sha256(coverage_path) != coverage_binding.get("sha256"):
        raise ValueError("AI-assisted v0.3 coverage binding changed")
    coverage = load_first_party_authoring_plan(coverage_path)

    predecessor = value.get("predecessor_binding")
    if not isinstance(predecessor, Mapping):
        raise ValueError("AI-assisted v0.3 predecessor binding is missing")
    evidence_path = root / str(predecessor.get("candidate_evidence_path"))
    if file_sha256(evidence_path) != predecessor.get("candidate_evidence_sha256"):
        raise ValueError("AI-assisted v0.3 predecessor evidence changed")
    evidence = load_ai_assisted_v2_evidence(evidence_path)
    if evidence.get("status") != predecessor.get("required_status"):
        raise ValueError("AI-assisted v0.3 predecessor status changed")
    private_source = root / str(predecessor.get("private_source_path"))
    if file_sha256(private_source) != predecessor.get("private_source_sha256"):
        raise ValueError("AI-assisted v0.3 private predecessor source changed")

    review = value.get("technical_review_binding")
    if not isinstance(review, Mapping):
        raise ValueError("AI-assisted v0.3 technical-review binding is missing")
    review_path = root / str(review.get("path"))
    if file_sha256(review_path) != review.get("sha256"):
        raise ValueError("AI-assisted v0.3 technical review changed")
    review_value = _load_json(review_path)
    if review_value.get("status") != "accepted_as_v0.3_repair_requirements":
        raise ValueError("AI-assisted v0.3 technical review is not accepted")

    tool_binding = value.get("tool_loss_contract_binding")
    if not isinstance(tool_binding, Mapping):
        raise ValueError("AI-assisted v0.3 tool-loss binding is missing")
    template_path = root / str(tool_binding.get("path"))
    if file_sha256(template_path) != tool_binding.get("sha256"):
        raise ValueError("AI-assisted v0.3 tool-loss contract changed")
    template = load_chat_template(template_path)
    if template.loss_roles != frozenset({"assistant", "tool"}):
        raise ValueError("AI-assisted v0.3 tool invocation is not supervised")
    if tool_binding.get("supervised_roles") != ["assistant", "tool"]:
        raise ValueError("AI-assisted v0.3 supervised-role declaration changed")
    if tool_binding.get("tool_result_is_context_only") is not True:
        raise ValueError("AI-assisted v0.3 tool-result masking boundary changed")

    language_policy = value.get("language_policy")
    if not isinstance(language_policy, Mapping):
        raise ValueError("AI-assisted v0.3 language policy is missing")
    observed_modes = {
        label: config.get("language_mode")
        for label, config in language_policy.items()
        if isinstance(config, Mapping)
    }
    if observed_modes != EXPECTED_LANGUAGE_MODES:
        raise ValueError("AI-assisted v0.3 language modes changed")

    owner = value.get("owner_gate")
    if not isinstance(owner, Mapping) or owner.get("sample_records") != 32:
        raise ValueError("AI-assisted v0.3 owner gate changed")
    if owner.get("dataset_level_signoff_required") is not True:
        raise ValueError("AI-assisted v0.3 owner signoff is required")
    if owner.get("per_row_human_review_required") is not False:
        raise ValueError("AI-assisted v0.3 per-row review boundary changed")
    return value, coverage


def _safe_fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise CalculatorExpressionError("non_integer_literal")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_fraction(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise CalculatorExpressionError("unsupported_expression")
    left = _safe_fraction(node.left)
    right = _safe_fraction(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise CalculatorExpressionError("division_by_zero")
        return left / right
    raise CalculatorExpressionError("unsupported_operator")


def evaluate_integer_expression(expression: str) -> int:
    """Evaluate a small arithmetic expression without eval or floating point."""

    if not isinstance(expression, str) or not expression.strip() or len(expression) > 128:
        raise CalculatorExpressionError("expression_size")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculatorExpressionError("syntax_error") from error
    if sum(1 for _ in ast.walk(parsed)) > 64:
        raise CalculatorExpressionError("expression_complexity")
    value = _safe_fraction(parsed.body)
    if value.denominator != 1:
        raise CalculatorExpressionError("non_integer_result")
    result = value.numerator
    if abs(result) > 10**12:
        raise CalculatorExpressionError("result_out_of_range")
    return result


def _final_assistant_text(messages: list[Mapping[str, str]]) -> str:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("AI-assisted v0.3 final message must be assistant")
    return messages[-1]["content"]


def _all_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _user_text(messages: list[Mapping[str, str]]) -> str:
    return "\n".join(
        message["content"] for message in messages if message["role"] == "user"
    )


def _contains_term(final_response: str, term: str) -> bool:
    normalized_term = normalize_for_matching(term)
    if _BENGALI.search(normalized_term):
        return normalized_term in normalize_for_matching(final_response)
    if _CANONICAL_INTEGER.fullmatch(normalized_term):
        return normalized_term in _INTEGER_TOKEN.findall(final_response)
    response_tokens = matching_tokens(final_response)
    term_tokens = matching_tokens(normalized_term)
    if not term_tokens:
        return False
    width = len(term_tokens)
    return any(
        response_tokens[index : index + width] == term_tokens
        for index in range(len(response_tokens) - width + 1)
    )


def _required_terms(evidence: Mapping[str, Any], final_response: str) -> tuple[str, ...]:
    raw_terms = evidence.get("required_response_terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("AI-assisted v0.3 required response terms are missing")
    if any(not isinstance(term, str) or not term.strip() for term in raw_terms):
        raise ValueError("AI-assisted v0.3 required response terms are invalid")
    terms = tuple(raw_terms)
    if any(not _contains_term(final_response, term) for term in terms):
        raise ValueError("AI-assisted v0.3 final response misses a required term")
    return terms


def _parse_json_object(content: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"AI-assisted v0.3 {name} is malformed JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"AI-assisted v0.3 {name} must be an object")
    return value


def _validate_calculator(
    evidence: Mapping[str, Any],
    messages: list[Mapping[str, str]],
    final_response: str,
) -> None:
    tool_call = _parse_json_object(messages[1]["content"], "calculator tool call")
    if set(tool_call) != {"arguments", "name"} or tool_call.get("name") != "calculator":
        raise ValueError("AI-assisted v0.3 calculator tool-call schema changed")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {"expression"}:
        raise ValueError("AI-assisted v0.3 calculator arguments schema changed")
    expression = _required_string(arguments.get("expression"), "calculator expression")
    result_payload = _parse_json_object(messages[2]["content"], "calculator result")
    route_kind = evidence.get("route_kind")
    if route_kind == "success":
        evaluated = evaluate_integer_expression(expression)
        expected = evidence.get("expected_result")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("AI-assisted v0.3 calculator expected result is invalid")
        if set(result_payload) != {"result"}:
            raise ValueError("AI-assisted v0.3 calculator success schema changed")
        raw_result = result_payload["result"]
        if isinstance(raw_result, bool) or not isinstance(raw_result, (int, str)):
            raise ValueError("AI-assisted v0.3 calculator result type is invalid")
        if not _CANONICAL_INTEGER.fullmatch(str(raw_result)):
            raise ValueError("AI-assisted v0.3 calculator result is not canonical")
        if int(raw_result) != evaluated or expected != evaluated:
            raise ValueError("AI-assisted v0.3 calculator result failed independent evaluation")
        if str(evaluated) not in _INTEGER_TOKEN.findall(final_response):
            raise ValueError("AI-assisted v0.3 final calculator result is missing")
    elif route_kind == "tool_error":
        try:
            evaluate_integer_expression(expression)
        except CalculatorExpressionError as error:
            observed_error = error.code
        else:
            raise ValueError("AI-assisted v0.3 declared tool error did not occur")
        expected_error = _required_string(evidence.get("error_code"), "error_code")
        if set(result_payload) != {"error"} or not isinstance(
            result_payload.get("error"), Mapping
        ):
            raise ValueError("AI-assisted v0.3 calculator error schema changed")
        error_payload = result_payload["error"]
        if set(error_payload) != {"code", "message"}:
            raise ValueError("AI-assisted v0.3 calculator error fields changed")
        if error_payload.get("code") != expected_error or observed_error != expected_error:
            raise ValueError("AI-assisted v0.3 calculator error code does not match")
        _required_string(error_payload.get("message"), "calculator error message")
        if evidence.get("fabricated_result") is not False:
            raise ValueError("AI-assisted v0.3 calculator error fabricates a result")
        if _INTEGER_TOKEN.search(final_response):
            raise ValueError("AI-assisted v0.3 calculator error response invents a number")
    else:
        raise ValueError("AI-assisted v0.3 calculator route kind is invalid")


def _validate_task(
    row: Mapping[str, Any],
    messages: list[Mapping[str, str]],
) -> None:
    evidence = row.get("task_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("AI-assisted v0.3 task evidence is missing")
    skill = str(row["skill_id"])
    final_response = _final_assistant_text(messages)
    terms = _required_terms(evidence, final_response)
    if _PERSISTENT_MEMORY.search(final_response):
        raise ValueError("AI-assisted v0.3 response makes a persistent-memory claim")

    if skill == "greeting_and_small_talk":
        if "Afnan Technologies" in final_response:
            raise ValueError("AI-assisted v0.3 greeting adds irrelevant branding")
    elif skill.startswith("typo_recovery_"):
        if evidence.get("intent_recovered") is not True:
            raise ValueError("AI-assisted v0.3 typo intent was not recovered")
        if evidence.get("modifier_grounded") is not True or len(terms) < 2:
            raise ValueError("AI-assisted v0.3 typo modifier is not grounded")
    elif skill == "simple_arithmetic_model_answer":
        expression = _required_string(evidence.get("expression"), "arithmetic expression")
        expected = evidence.get("expected_integer")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("AI-assisted v0.3 arithmetic expected integer is invalid")
        if expression not in messages[0]["content"]:
            raise ValueError("AI-assisted v0.3 arithmetic prompt/evidence mismatch")
        if evaluate_integer_expression(expression) != expected:
            raise ValueError("AI-assisted v0.3 arithmetic expression is incorrect")
        if not _CANONICAL_INTEGER.fullmatch(final_response):
            raise ValueError("AI-assisted v0.3 arithmetic answer is not canonical")
        if int(final_response) != expected:
            raise ValueError("AI-assisted v0.3 arithmetic answer is incorrect")
    elif skill == "calculator_tool_routing":
        _validate_calculator(evidence, messages, final_response)
    elif skill == "multi_turn_context_and_correction":
        if evidence.get("context_retained") is not True:
            raise ValueError("AI-assisted v0.3 multi-turn context was not retained")
        if evidence.get("context_scope_explicit") is not True:
            raise ValueError("AI-assisted v0.3 multi-turn scope is not explicit")
    elif skill == "identity_and_capability_boundaries":
        if evidence.get("verified_claims_only") is not True:
            raise ValueError("AI-assisted v0.3 identity claims are not verified")
        response_kind = evidence.get("response_kind")
        if response_kind not in {"identity", "developer", "capability", "boundary"}:
            raise ValueError("AI-assisted v0.3 identity response kind is invalid")
        if response_kind in {"identity", "developer"}:
            if "Lumi" not in final_response or "Afnan Technologies Inc." not in final_response:
                raise ValueError("AI-assisted v0.3 canonical identity is missing")
        elif "Afnan Technologies" in final_response:
            raise ValueError("AI-assisted v0.3 capability answer adds branding")
    else:  # pragma: no cover - closed by exact coverage
        raise ValueError(f"unsupported AI-assisted v0.3 skill: {skill}")


def validate_ai_assisted_v3_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one generated row without accepting a human-authorship claim."""

    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("unsupported AI-assisted v0.3 row schema")
    for field in ("record_id", "skill_id", "split", "task_category", "language"):
        if row.get(field) != expected.get(field):
            raise ValueError(f"AI-assisted v0.3 immutable metadata changed: {field}")
    if row.get("language_mode") != EXPECTED_LANGUAGE_MODES[row["language"]]:
        raise ValueError("AI-assisted v0.3 explicit language mode changed")

    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("AI-assisted v0.3 messages must be an array")
    messages: list[Mapping[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping) or set(message) != {"content", "role"}:
            raise ValueError("AI-assisted v0.3 message fields changed")
        if not isinstance(message.get("content"), str) or not isinstance(
            message.get("role"), str
        ):
            raise ValueError("AI-assisted v0.3 message types are invalid")
        messages.append(message)
    validate_message_sequence(messages)
    if [message["role"] for message in messages] != EXPECTED_ROLE_SHAPES[row["skill_id"]]:
        raise ValueError("AI-assisted v0.3 skill role sequence changed")

    combined = _all_text(messages)
    if _EMAIL.search(combined) or _PHONE.search(combined) or _SECRET.search(combined):
        raise ValueError("AI-assisted v0.3 row contains PII or secret-like text")
    contains_bengali = _BENGALI.search(combined) is not None
    if row["language_mode"] == "bengali_script" and not contains_bengali:
        raise ValueError("AI-assisted v0.3 Bengali row lacks Bengali script")
    if row["language_mode"] in {"english", "banglish_romanized"} and contains_bengali:
        raise ValueError("AI-assisted v0.3 non-Bengali-script row contains Bengali text")
    if row["language_mode"] == "banglish_romanized":
        markers = set(matching_tokens(combined)) & _BANGLISH_MARKERS
        if len(markers) < 2:
            raise ValueError("AI-assisted v0.3 Banglish row lacks Romanized Bengali")

    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("AI-assisted v0.3 generation evidence is missing")
    if generation.get("generator_id") not in plan["generation_policy"][
        "allowed_generator_ids"
    ]:
        raise ValueError("AI-assisted v0.3 generator is not allowed")
    if generation.get("method") != plan["generation_policy"]["method"]:
        raise ValueError("AI-assisted v0.3 generation method changed")
    if generation.get("human_written_from_scratch") is not False:
        raise ValueError("AI-assisted v0.3 row falsely claims human authorship")
    if generation.get("rights_confirmed") is not True:
        raise ValueError("AI-assisted v0.3 rights evidence is missing")
    timestamp = _required_string(generation.get("generated_at"), "generated_at")
    if _UTC_TIMESTAMP.fullmatch(timestamp) is None:
        raise ValueError("AI-assisted v0.3 generated_at is not UTC ISO-8601")
    family = _required_string(
        generation.get("semantic_family_id"),
        "semantic_family_id",
    )
    if _FAMILY_ID.fullmatch(family) is None or not family.startswith(
        f"{row['skill_id']}:"
    ):
        raise ValueError("AI-assisted v0.3 semantic family ID is invalid")
    if any(part in {"train", "validation"} for part in family.split(":")):
        raise ValueError("AI-assisted v0.3 semantic family ID embeds the split")

    automated = row.get("automated_review")
    if not isinstance(automated, Mapping):
        raise ValueError("AI-assisted v0.3 automated review is missing")
    if automated.get("reviewer_id") != plan["automated_review"]["reviewer_id"]:
        raise ValueError("AI-assisted v0.3 automated reviewer changed")
    for field in (
        "calculator_expression_verified",
        "context_grounding_passed",
        "final_assistant_grounding_passed",
        "identity_tool_claims_passed",
        "language_mode_passed",
        "message_sequence_passed",
        "pii_secret_scan_passed",
        "safety_passed",
        "semantic_family_passed",
        "task_correctness_passed",
    ):
        if automated.get(field) is not True:
            raise ValueError(f"AI-assisted v0.3 automated review failed: {field}")
    score = automated.get("quality_score")
    minimum = float(plan["automated_review"]["minimum_quality_score"])
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not minimum <= float(score) <= 1
    ):
        raise ValueError("AI-assisted v0.3 quality score is invalid")

    if row.get("owner_review") != {
        "accepted": None,
        "reviewed_at": "",
        "reviewer_id": "",
        "status": "pending",
    }:
        raise ValueError("AI-assisted v0.3 row may not claim owner approval")
    _validate_task(row, messages)
    return json.loads(json.dumps(row, ensure_ascii=False))


def _final_response(row: Mapping[str, Any]) -> str:
    return _final_assistant_text(list(row["messages"]))


def _prompt_shell(row: Mapping[str, Any]) -> str:
    value = normalize_for_matching(_user_text(list(row["messages"])))
    return re.sub(r"(?<!\w)-?[0-9]+(?!\w)", "<num>", value)


def _near_pairs(
    train_values: set[str],
    validation_values: set[str],
    *,
    threshold: float,
) -> int:
    return sum(
        1
        for train in train_values
        for validation in validation_values
        if train != validation
        and SequenceMatcher(None, train, validation, autojunk=False).ratio() >= threshold
    )


def audit_ai_assisted_v3_rows(
    rows: list[Mapping[str, Any]],
    *,
    coverage_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Run strict row and cross-split gates while retaining no raw text."""

    expected = expected_candidate_metadata(coverage_plan)
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("AI-assisted v0.3 row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected:
            raise ValueError("AI-assisted v0.3 contains an unknown record ID")
        if record_id in observed:
            raise ValueError("AI-assisted v0.3 contains a duplicate record ID")
        observed[record_id] = row
    if set(observed) != set(expected):
        raise ValueError("AI-assisted v0.3 exact record membership changed")

    validated = [
        validate_ai_assisted_v3_row(
            observed[record_id],
            expected=expected[record_id],
            plan=plan,
        )
        for record_id in sorted(expected)
    ]
    normalized_conversations = {
        _fingerprint(
            [
                {
                    "content": normalize_for_matching(message["content"]),
                    "role": message["role"],
                }
                for message in row["messages"]
            ]
        )
        for row in validated
    }
    if len(normalized_conversations) != len(validated):
        raise ValueError("AI-assisted v0.3 has duplicate normalized conversations")

    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        by_skill.setdefault(row["skill_id"], []).append(row)
    threshold = float(plan["quality_gates"]["near_duplicate_similarity_threshold"])
    exempt = set(plan["quality_gates"]["near_duplicate_exempt_skills"])
    totals = Counter()
    metrics: dict[str, Any] = {}
    for skill, skill_rows in sorted(by_skill.items()):
        train = [row for row in skill_rows if row["split"] == "train"]
        validation = [row for row in skill_rows if row["split"] == "validation"]
        final_responses = [normalize_for_matching(_final_response(row)) for row in skill_rows]
        unique_ratio = len(set(final_responses)) / len(final_responses)
        if unique_ratio < plan["quality_gates"][
            "final_response_unique_ratio_by_skill_minimum"
        ]:
            raise ValueError(f"AI-assisted v0.3 response diversity failed: {skill}")

        train_exact = {_final_response(row) for row in train}
        validation_exact = {_final_response(row) for row in validation}
        exact_overlap = len(train_exact & validation_exact)
        train_normalized = {normalize_for_matching(value) for value in train_exact}
        validation_normalized = {
            normalize_for_matching(value) for value in validation_exact
        }
        normalized_overlap = len(train_normalized & validation_normalized)
        train_families = {row["generation"]["semantic_family_id"] for row in train}
        validation_families = {
            row["generation"]["semantic_family_id"] for row in validation
        }
        family_overlap = len(train_families & validation_families)
        prompt_near = 0
        response_near = 0
        if skill not in exempt:
            prompt_near = _near_pairs(
                {_prompt_shell(row) for row in train},
                {_prompt_shell(row) for row in validation},
                threshold=threshold,
            )
            response_near = _near_pairs(
                train_normalized,
                validation_normalized,
                threshold=threshold,
            )
        totals.update(
            {
                "exact_overlap": exact_overlap,
                "family_overlap": family_overlap,
                "normalized_overlap": normalized_overlap,
                "prompt_near": prompt_near,
                "response_near": response_near,
            }
        )
        metrics[skill] = {
            "final_response_exact_cross_split_overlap": exact_overlap,
            "final_response_normalized_cross_split_overlap": normalized_overlap,
            "final_response_unique_ratio": unique_ratio,
            "prompt_near_duplicate_cross_split_pairs": prompt_near,
            "records": len(skill_rows),
            "response_near_duplicate_cross_split_pairs": response_near,
            "semantic_family_cross_split_overlap": family_overlap,
        }

    gates = plan["quality_gates"]
    checks = {
        "exact_overlap": "final_response_exact_cross_split_overlap_max",
        "family_overlap": "semantic_family_cross_split_overlap_max",
        "normalized_overlap": "final_response_normalized_cross_split_overlap_max",
        "prompt_near": "prompt_near_duplicate_cross_split_pairs_max",
        "response_near": "response_near_duplicate_cross_split_pairs_max",
    }
    for metric, gate in checks.items():
        if totals[metric] > gates[gate]:
            raise ValueError(f"AI-assisted v0.3 quality gate failed: {metric}")

    language_counts = Counter(row["language_mode"] for row in validated)
    split_counts = Counter(row["split"] for row in validated)
    skill_counts = Counter(row["skill_id"] for row in validated)
    report: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "automated_review_passed_records": len(validated),
        "candidate_source_revision": _fingerprint(validated),
        "language_mode_counts": dict(sorted(language_counts.items())),
        "normalized_unique_conversations": len(normalized_conversations),
        "owner_review": {
            "complete": False,
            "sample_records": plan["owner_gate"]["sample_records"],
            "status": "pending",
        },
        "plan_sha256": _fingerprint(plan),
        "production_eligible": False,
        "quality_gate_results": {
            "calculator_independent_evaluation_rate": 1.0,
            "canonical_integer_answer_rate": 1.0,
            "final_response_exact_cross_split_overlap": totals["exact_overlap"],
            "final_response_normalized_cross_split_overlap": totals[
                "normalized_overlap"
            ],
            "final_term_grounding_rate": 1.0,
            "identity_branding_on_capability_records": 0,
            "persistent_memory_claim_records": 0,
            "prompt_near_duplicate_cross_split_pairs": totals["prompt_near"],
            "response_near_duplicate_cross_split_pairs": totals["response_near"],
            "semantic_family_cross_split_overlap": totals["family_overlap"],
        },
        "raw_training_text_retained": False,
        "records": len(validated),
        "response_metrics_by_skill": metrics,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "skill_counts": dict(sorted(skill_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
    report["report_sha256"] = _fingerprint(report)
    return report


def select_owner_review_sample_v3(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select three train language modes and one held-out row per skill."""

    by_skill: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(str(row["skill_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for skill in sorted(by_skill):
        train = sorted(
            (row for row in by_skill[skill] if row["split"] == "train"),
            key=lambda row: _fingerprint(row["record_id"]),
        )
        validation = sorted(
            (row for row in by_skill[skill] if row["split"] == "validation"),
            key=lambda row: _fingerprint(row["record_id"]),
        )
        chosen: list[Mapping[str, Any]] = []
        seen_modes: set[str] = set()
        for row in train:
            if row["language_mode"] not in seen_modes:
                chosen.append(row)
                seen_modes.add(row["language_mode"])
            if len(chosen) == 3:
                break
        for row in train:
            if len(chosen) == 3:
                break
            if row not in chosen:
                chosen.append(row)
        if len(chosen) != 3 or not validation:
            raise ValueError("AI-assisted v0.3 owner sample coverage is incomplete")
        selected.extend(
            json.loads(json.dumps(row, ensure_ascii=False))
            for row in (*chosen, validation[0])
        )
    if len(selected) != 32:
        raise AssertionError("AI-assisted v0.3 owner sample size changed")
    return selected


def to_sft_candidate_records_v3(
    rows: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_revision: str,
) -> list[dict[str, Any]]:
    """Convert validated rows while preserving AI provenance and pending owner status."""

    source = plan["source_binding"]
    transform_sha256 = _fingerprint(
        {"schema": "lumi.ai-assisted-to-sft.v0.3", "source_revision": source_revision}
    )
    records = []
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
                    "review_method": "deterministic_v0.3_gates_owner_pending",
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


def build_ai_assisted_v3_manifest(
    *,
    audit: Mapping[str, Any],
    candidate_records: list[Mapping[str, Any]],
    owner_sample: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build raw-text-free evidence for an owner-review-pending candidate."""

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
        "status": "machine_validated_v0.3_candidate_waiting_for_owner_spot_check",
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest


def build_ai_assisted_v3_evidence(
    *,
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    private_artifact_path: str,
    filter_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a raw-text-free tracked receipt for the private v0.3 candidate."""

    candidate: dict[str, Any] = {
        "automated_review_passed_records": audit["automated_review_passed_records"],
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "candidate_source_revision": manifest["candidate_source_revision"],
        "candidate_sft_jsonl_sha256": manifest["candidate_sft_jsonl_sha256"],
        "machine_audit_report_sha256": audit["report_sha256"],
        "normalized_unique_conversations": audit["normalized_unique_conversations"],
        "quality_gate_results": audit["quality_gate_results"],
        "records": audit["records"],
        "sft_filter_accepted_records": filter_report["accepted"],
        "sft_filter_rejected_records": filter_report["rejected"],
        "sft_filter_report_sha256": filter_report["report_sha256"],
    }
    evidence: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "candidate": candidate,
        "owner_gate": {
            "complete": False,
            "sample_records": manifest["owner_sample_records"],
            "sample_sha256": manifest["owner_sample_sha256"],
            "status": "pending",
        },
        "private_artifact": {
            "path": private_artifact_path,
            "raw_training_text_in_this_evidence": False,
        },
        "production_eligible": False,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "machine_validated_v0.3_candidate_waiting_for_owner_spot_check",
    }
    evidence["evidence_sha256"] = _fingerprint(evidence)
    return evidence
