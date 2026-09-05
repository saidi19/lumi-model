"""Fail-closed gates for a private AI-assisted safety and tool SFT pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .data_quality import normalize_for_matching
from .posttraining_ai_assisted_v3 import (
    CalculatorExpressionError,
    evaluate_integer_expression,
)
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import canonical_json_bytes, file_sha256

PLAN_SCHEMA_VERSION = "lumi.safety-tool-sft-pilot-plan.v0.1"
ROW_SCHEMA_VERSION = "lumi.safety-tool-sft-row.v0.1"
AUDIT_SCHEMA_VERSION = "lumi.safety-tool-sft-audit.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.safety-tool-sft-manifest.v0.1"
EVIDENCE_SCHEMA_VERSION = "lumi.safety-tool-sft-evidence.v0.1"

EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "external_model_api_spend_authorized": False,
    "human_authorship_claim_authorized": False,
    "model_generated_training_text_authorized": True,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
EXPECTED_SOURCE = {
    "dataset_id": "Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1",
    "license_id": "lumi-owned",
    "source_id": "lumi_safety_tool_sft_v0_1",
}
EXPECTED_LIMITS = {
    "families_per_skill": 8,
    "languages_per_family": 4,
    "owner_sample_records": 40,
    "records": 320,
    "train_records": 240,
    "validation_records": 80,
}
EXPECTED_BINDINGS = frozenset(
    {
        "assistant_behavior_spec",
        "calculator_implementation",
        "chat_template",
        "missing_mix_proposal",
        "pilot_implementation",
        "pilot_runner",
        "sft_data_contract",
        "sft_filter_implementation",
    }
)
EXPECTED_LANGUAGE_MODES = {
    "banglish": {"dataset_language": "multilingual"},
    "bengali_script": {"dataset_language": "bn"},
    "english": {"dataset_language": "en"},
    "natural_code_switching": {"dataset_language": "multilingual"},
}
EXPECTED_SKILLS = {
    "benign_request_without_overrefusal": {
        "expected_action": "comply_benignly",
        "role_shape": ["user", "assistant"],
        "task_category": "safety_helpfulness",
    },
    "calculator_tool_error": {
        "expected_action": "report_tool_error_without_fabrication",
        "role_shape": ["user", "tool", "tool_result", "assistant"],
        "task_category": "tool_api",
    },
    "calculator_tool_success": {
        "expected_action": "use_and_ground_calculator_result",
        "role_shape": ["user", "tool", "tool_result", "assistant"],
        "task_category": "tool_api",
    },
    "current_information_retrieval_boundary": {
        "expected_action": "state_fresh_source_requirement",
        "role_shape": ["user", "assistant"],
        "task_category": "tool_api",
    },
    "harmful_request_refusal_and_safe_alternative": {
        "expected_action": "refuse_and_redirect",
        "role_shape": ["user", "assistant"],
        "task_category": "safety_helpfulness",
    },
    "high_stakes_uncertainty_and_escalation": {
        "expected_action": "calibrate_and_escalate",
        "role_shape": ["user", "assistant"],
        "task_category": "safety_helpfulness",
    },
    "list_sorter_tool_grounding": {
        "expected_action": "ground_response_in_sorter_result",
        "role_shape": ["user", "tool", "tool_result", "assistant"],
        "task_category": "tool_api",
    },
    "privacy_and_secret_protection": {
        "expected_action": "refuse_disclosure_and_protect_account",
        "role_shape": ["user", "assistant"],
        "task_category": "safety_helpfulness",
    },
    "prompt_injection_resistance": {
        "expected_action": "ignore_injection_and_protect_data",
        "role_shape": ["user", "assistant"],
        "task_category": "safety_helpfulness",
    },
    "tool_capability_honesty": {
        "expected_action": "state_unavailable_tool_and_next_step",
        "role_shape": ["user", "assistant"],
        "task_category": "tool_api",
    },
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BENGALI = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\bsk-[A-Za-z0-9_-]{16,}\b|"
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9_./+=-]{12,})",
    re.IGNORECASE,
)
_TRAINING_MARKER = re.compile(r"<\|(?:doc|endofdoc)\|>")
_REFUSAL = re.compile(
    r"(?:\bi (?:can't|cannot|won't) (?:help|provide|assist)|"
    r"\bami (?:eta )?(?:korte|dite) parbo na|"
    r"আমি (?:এটা |ওটা )?(?:করতে|দিতে) পারব না|"
    r"এতে সাহায্য করতে পারব না)",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"(?:\b(?:can't|cannot) (?:confirm|diagnose|verify)|\bnot enough information\b|"
    r"নিশ্চিত করে বলা যাচ্ছে না|রোগ নির্ণয় করতে পারি না|"
    r"confirm kora jacche na|diagnose korte pari na)",
    re.IGNORECASE,
)
_ESCALATION = re.compile(
    r"(?:\b(?:doctor|lawyer|emergency services|qualified professional)\b|"
    r"ডাক্তার|আইনজীবী|জরুরি সেবা|বিশেষজ্ঞ|"
    r"daktar|lawyer|emergency service|specialist)",
    re.IGNORECASE,
)
_TOOL_BOUNDARY = re.compile(
    r"(?:\b(?:don't|do not|can't|cannot) have (?:live |direct )?(?:access|browsing)|"
    r"\bneed (?:a )?(?:connected|live|current) source\b|"
    r"লাইভ (?:অ্যাক্সেস|তথ্য)|সরাসরি অ্যাক্সেস নেই|"
    r"live access nei|connected source lagbe)",
    re.IGNORECASE,
)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"safety/tool pilot {field} is required")
    return value


def load_safety_tool_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact local-only pilot plan and verify every file binding."""

    root = Path(repository_root).resolve()
    workspace = root.parent
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool pilot plan")
    if plan.get("status") != "candidate_build_authorized_owner_signoff_pending":
        raise ValueError("safety/tool pilot status changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("safety/tool pilot authorization changed")
    if plan.get("source_binding") != EXPECTED_SOURCE:
        raise ValueError("safety/tool pilot source changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("safety/tool pilot limits changed")
    if plan.get("language_modes") != EXPECTED_LANGUAGE_MODES:
        raise ValueError("safety/tool pilot language modes changed")
    if plan.get("skills") != EXPECTED_SKILLS:
        raise ValueError("safety/tool pilot skills changed")
    if plan.get("family_split") != {
        "train_family_indices": [0, 1, 2, 3, 4, 5],
        "validation_family_indices": [6, 7],
    }:
        raise ValueError("safety/tool pilot split policy changed")
    quality = plan.get("quality_gates")
    if quality != {
        "cross_split_near_duplicate_threshold": 0.97,
        "exact_prompt_overlap_max": 0,
        "exact_response_overlap_max": 0,
        "normalized_conversation_unique_ratio_minimum": 1.0,
        "prompt_near_duplicate_pairs_max": 0,
        "response_near_duplicate_pairs_max": 0,
        "semantic_family_cross_split_overlap_max": 0,
    }:
        raise ValueError("safety/tool pilot quality gates changed")
    owner = plan.get("owner_gate")
    if owner != {
        "dataset_level_signoff_required": True,
        "independent_second_human_required": False,
        "per_row_human_review_required": False,
        "sample_records": 40,
        "status": "pending",
    }:
        raise ValueError("safety/tool pilot owner gate changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("safety/tool pilot binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"safety/tool binding must be an object: {name}")
        relative = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"safety/tool binding path is invalid: {name}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"safety/tool binding SHA is invalid: {name}")
        resolved = (root / relative).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"safety/tool binding escapes workspace: {name}")
        if not resolved.is_file() or file_sha256(resolved) != digest:
            raise ValueError(f"safety/tool binding changed: {name}")
    return plan


def expected_safety_tool_records(plan: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Build the exact immutable row membership from the validated plan."""

    language_modes = tuple(sorted(plan["language_modes"]))
    train_families = set(plan["family_split"]["train_family_indices"])
    expected: dict[str, dict[str, str]] = {}
    for skill_id, skill in sorted(plan["skills"].items()):
        for family_index in range(plan["limits"]["families_per_skill"]):
            split = "train" if family_index in train_families else "validation"
            family_id = f"{skill_id}:family-{family_index:02d}"
            for language_mode in language_modes:
                language = plan["language_modes"][language_mode]["dataset_language"]
                record_id = f"{skill_id}-{family_index:02d}-{language_mode}"
                expected[record_id] = {
                    "language": language,
                    "language_mode": language_mode,
                    "record_id": record_id,
                    "semantic_family_id": family_id,
                    "skill_id": skill_id,
                    "split": split,
                    "task_category": skill["task_category"],
                }
    if len(expected) != plan["limits"]["records"]:
        raise AssertionError("safety/tool pilot exact membership changed")
    return expected


def _messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list):
        raise ValueError("safety/tool pilot messages must be an array")
    messages: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, Mapping) or set(message) != {"content", "role"}:
            raise ValueError("safety/tool pilot message fields changed")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            raise ValueError("safety/tool pilot message content is invalid")
        messages.append({"content": content, "role": role})
    validate_message_sequence(messages)
    return messages


def _required_terms(evidence: Mapping[str, Any], response: str) -> list[str]:
    terms = evidence.get("required_response_terms")
    if (
        not isinstance(terms, list)
        or len(terms) < 2
        or any(not isinstance(term, str) or not term.strip() for term in terms)
    ):
        raise ValueError("safety/tool required response terms are invalid")
    normalized = response.casefold()
    if any(term.casefold() not in normalized for term in terms):
        raise ValueError("safety/tool final response misses required grounding")
    return terms


def _validate_language(row: Mapping[str, Any], text: str) -> None:
    mode = row["language_mode"]
    has_bengali = _BENGALI.search(text) is not None
    has_latin = _LATIN.search(text) is not None
    if mode == "english" and has_bengali:
        raise ValueError("English safety/tool row contains Bengali script")
    if mode == "bengali_script" and not has_bengali:
        raise ValueError("Bengali safety/tool row lacks Bengali script")
    if mode == "banglish" and (has_bengali or not has_latin):
        raise ValueError("Banglish safety/tool row is not Romanized Bengali")
    if mode == "natural_code_switching" and not (has_bengali and has_latin):
        raise ValueError("code-switching safety/tool row needs Bengali and Latin script")


def _validate_calculator(
    evidence: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    *,
    expect_error: bool,
) -> None:
    call = json.loads(messages[1]["content"])
    result = json.loads(messages[2]["content"])
    if set(call) != {"arguments", "name"} or call.get("name") != "calculator":
        raise ValueError("safety/tool calculator call schema changed")
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {"expression"}:
        raise ValueError("safety/tool calculator arguments changed")
    expression = _required_string(arguments.get("expression"), "calculator expression")
    if expression != evidence.get("expression"):
        raise ValueError("safety/tool calculator evidence mismatch")
    if expect_error:
        try:
            evaluate_integer_expression(expression)
        except CalculatorExpressionError as error:
            observed_error = error.code
        else:
            raise ValueError("safety/tool calculator error did not reproduce")
        error_payload = result.get("error") if set(result) == {"error"} else None
        if not isinstance(error_payload, Mapping) or set(error_payload) != {
            "code",
            "message",
        }:
            raise ValueError("safety/tool calculator error schema changed")
        if observed_error != evidence.get("error_code") or error_payload.get(
            "code"
        ) != evidence.get("error_code"):
            raise ValueError("safety/tool calculator error code mismatch")
        if evidence.get("fabricated_result") is not False:
            raise ValueError("safety/tool calculator error fabricates a result")
    else:
        expected = evidence.get("expected_integer")
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("safety/tool calculator expected integer is invalid")
        if evaluate_integer_expression(expression) != expected:
            raise ValueError("safety/tool calculator expected integer is wrong")
        if set(result) != {"result"} or result.get("result") != expected:
            raise ValueError("safety/tool calculator result is not independently verified")
        if str(expected) not in messages[-1]["content"]:
            raise ValueError("safety/tool calculator answer omits verified result")


def _validate_list_sorter(
    evidence: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
) -> None:
    call = json.loads(messages[1]["content"])
    result = json.loads(messages[2]["content"])
    if set(call) != {"arguments", "name"} or call.get("name") != "list_sorter":
        raise ValueError("safety/tool sorter call schema changed")
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {"items"}:
        raise ValueError("safety/tool sorter arguments changed")
    items = arguments.get("items")
    if (
        not isinstance(items, list)
        or len(items) < 3
        or any(not isinstance(item, str) or not item for item in items)
    ):
        raise ValueError("safety/tool sorter items are invalid")
    expected = sorted(items, key=str.casefold)
    if result != {"sorted_items": expected}:
        raise ValueError("safety/tool sorter result is not recomputable")
    if evidence.get("expected_sorted_items") != expected:
        raise ValueError("safety/tool sorter evidence mismatch")
    if any(item not in messages[-1]["content"] for item in expected):
        raise ValueError("safety/tool sorter response omits a verified item")


def _validate_task(
    row: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
) -> None:
    skill_id = str(row["skill_id"])
    skill = EXPECTED_SKILLS[skill_id]
    evidence = row.get("task_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("safety/tool task evidence is missing")
    if evidence.get("expected_action") != skill["expected_action"]:
        raise ValueError("safety/tool expected action changed")
    response = messages[-1]["content"]
    _required_terms(evidence, response)

    refusal_required = skill_id in {
        "harmful_request_refusal_and_safe_alternative",
        "privacy_and_secret_protection",
        "prompt_injection_resistance",
    }
    if evidence.get("refusal_required") is not refusal_required:
        raise ValueError("safety/tool refusal evidence changed")
    if evidence.get("response_refused") is not refusal_required:
        raise ValueError("safety/tool refusal outcome changed")
    if refusal_required and _REFUSAL.search(response) is None:
        raise ValueError("safety/tool required refusal is missing")
    if skill_id == "benign_request_without_overrefusal" and _REFUSAL.search(response):
        raise ValueError("safety/tool benign request was over-refused")
    if skill_id == "high_stakes_uncertainty_and_escalation":
        if _UNCERTAINTY.search(response) is None or _ESCALATION.search(response) is None:
            raise ValueError("safety/tool high-stakes calibration is incomplete")
    if skill_id in {
        "harmful_request_refusal_and_safe_alternative",
        "privacy_and_secret_protection",
        "prompt_injection_resistance",
    } and evidence.get("safe_alternative_present") is not True:
        raise ValueError("safety/tool safe alternative is missing")

    if skill_id in {
        "tool_capability_honesty",
        "current_information_retrieval_boundary",
    }:
        if evidence.get("tool_available") is not False:
            raise ValueError("safety/tool unavailable-tool evidence changed")
        if evidence.get("fabricated_access") is not False:
            raise ValueError("safety/tool response fabricates tool access")
        if _TOOL_BOUNDARY.search(response) is None:
            raise ValueError("safety/tool capability boundary is missing")
    elif skill_id == "calculator_tool_success":
        _validate_calculator(evidence, messages, expect_error=False)
    elif skill_id == "calculator_tool_error":
        _validate_calculator(evidence, messages, expect_error=True)
    elif skill_id == "list_sorter_tool_grounding":
        _validate_list_sorter(evidence, messages)


def validate_safety_tool_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, str],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one private row while forbidding human-authorship claims."""

    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("unsupported safety/tool pilot row schema")
    for field in (
        "language",
        "language_mode",
        "record_id",
        "skill_id",
        "split",
        "task_category",
    ):
        if row.get(field) != expected.get(field):
            raise ValueError(f"safety/tool immutable metadata changed: {field}")
    messages = _messages(row)
    if [message["role"] for message in messages] != plan["skills"][row["skill_id"]][
        "role_shape"
    ]:
        raise ValueError("safety/tool role shape changed")
    combined = "\n".join(message["content"] for message in messages)
    if (
        _EMAIL.search(combined)
        or _PHONE.search(combined)
        or _SECRET.search(combined)
        or _TRAINING_MARKER.search(combined)
    ):
        raise ValueError("safety/tool row contains PII, secret, or training marker")
    _validate_language(row, combined)

    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("safety/tool generation evidence is missing")
    if generation != {
        "generated_at": "2026-08-06T00:00:00Z",
        "generator_id": "openai-codex-current-session",
        "human_written_from_scratch": False,
        "method": "ai_generated_local_candidate_owner_review_pending_v0.1",
        "rights_confirmed": True,
        "semantic_family_id": expected["semantic_family_id"],
    }:
        raise ValueError("safety/tool generation evidence changed")
    automated = row.get("automated_review")
    if automated != {
        "content_grounding_passed": True,
        "language_mode_passed": True,
        "message_sequence_passed": True,
        "pii_secret_scan_passed": True,
        "quality_score": 0.97,
        "reviewer_id": "lumi-safety-tool-deterministic-guard-v0.1",
        "safety_policy_passed": True,
        "task_correctness_passed": True,
        "tool_result_recomputed": True,
    }:
        raise ValueError("safety/tool automated review changed")
    if row.get("owner_review") != {
        "accepted": None,
        "reviewed_at": "",
        "reviewer_id": "",
        "status": "pending",
    }:
        raise ValueError("safety/tool row may not claim owner approval")
    _validate_task(row, messages)
    return json.loads(json.dumps(row, ensure_ascii=False))


def _near_pairs(train: set[str], validation: set[str], threshold: float) -> int:
    return sum(
        1
        for left in train
        for right in validation
        if left != right
        and SequenceMatcher(None, left, right, autojunk=False).ratio() >= threshold
    )


def audit_safety_tool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact membership, split isolation, and raw-free aggregate gates."""

    expected = expected_safety_tool_records(plan)
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("safety/tool row must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected:
            raise ValueError("safety/tool pilot contains an unknown record ID")
        if record_id in observed:
            raise ValueError("safety/tool pilot contains a duplicate record ID")
        observed[record_id] = row
    if set(observed) != set(expected):
        raise ValueError("safety/tool pilot exact record membership changed")
    validated = [
        validate_safety_tool_row(observed[key], expected=expected[key], plan=plan)
        for key in sorted(expected)
    ]

    normalized_conversations = {
        sha256_value(
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
    train = [row for row in validated if row["split"] == "train"]
    validation = [row for row in validated if row["split"] == "validation"]
    train_prompts = {
        normalize_for_matching(row["messages"][0]["content"]) for row in train
    }
    validation_prompts = {
        normalize_for_matching(row["messages"][0]["content"]) for row in validation
    }
    train_responses = {
        normalize_for_matching(row["messages"][-1]["content"]) for row in train
    }
    validation_responses = {
        normalize_for_matching(row["messages"][-1]["content"]) for row in validation
    }
    train_families = {row["generation"]["semantic_family_id"] for row in train}
    validation_families = {
        row["generation"]["semantic_family_id"] for row in validation
    }
    threshold = plan["quality_gates"]["cross_split_near_duplicate_threshold"]
    gates = {
        "exact_prompt_overlap": len(train_prompts & validation_prompts),
        "exact_response_overlap": len(train_responses & validation_responses),
        "normalized_conversation_unique_ratio": len(normalized_conversations)
        / len(validated),
        "prompt_near_duplicate_pairs": _near_pairs(
            train_prompts,
            validation_prompts,
            threshold,
        ),
        "response_near_duplicate_pairs": _near_pairs(
            train_responses,
            validation_responses,
            threshold,
        ),
        "semantic_family_cross_split_overlap": len(
            train_families & validation_families
        ),
    }
    expected_gates = plan["quality_gates"]
    if gates["exact_prompt_overlap"] > expected_gates["exact_prompt_overlap_max"]:
        raise ValueError("safety/tool exact prompt split leakage")
    if gates["exact_response_overlap"] > expected_gates["exact_response_overlap_max"]:
        raise ValueError("safety/tool exact response split leakage")
    if gates["normalized_conversation_unique_ratio"] < expected_gates[
        "normalized_conversation_unique_ratio_minimum"
    ]:
        raise ValueError("safety/tool normalized conversation duplication")
    if gates["prompt_near_duplicate_pairs"] > expected_gates[
        "prompt_near_duplicate_pairs_max"
    ]:
        raise ValueError("safety/tool prompt near-duplicate split leakage")
    if gates["response_near_duplicate_pairs"] > expected_gates[
        "response_near_duplicate_pairs_max"
    ]:
        raise ValueError("safety/tool response near-duplicate split leakage")
    if gates["semantic_family_cross_split_overlap"] > expected_gates[
        "semantic_family_cross_split_overlap_max"
    ]:
        raise ValueError("safety/tool semantic family split leakage")

    split_counts = Counter(row["split"] for row in validated)
    if split_counts != Counter(
        {
            "train": plan["limits"]["train_records"],
            "validation": plan["limits"]["validation_records"],
        }
    ):
        raise ValueError("safety/tool split counts changed")
    report: dict[str, Any] = {
        "authorization": {
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "candidate_source_revision": sha256_value(validated),
        "language_mode_counts": dict(
            sorted(Counter(row["language_mode"] for row in validated).items())
        ),
        "owner_review": {
            "complete": False,
            "sample_records": plan["limits"]["owner_sample_records"],
            "status": "pending",
        },
        "plan_sha256": sha256_value(plan),
        "production_eligible": False,
        "production_promotions": 0,
        "quality_gate_results": gates,
        "raw_training_text_retained": False,
        "records": len(validated),
        "schema_version": AUDIT_SCHEMA_VERSION,
        "skill_counts": dict(
            sorted(Counter(row["skill_id"] for row in validated).items())
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(
            sorted(Counter(row["task_category"] for row in validated).items())
        ),
    }
    report["report_sha256"] = sha256_value(report)
    return report


def select_owner_sample(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one hash-ranked validation row per skill and language mode."""

    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("split") == "validation":
            key = (str(row["skill_id"]), str(row["language_mode"]))
            buckets.setdefault(key, []).append(row)
    selected = []
    for key in sorted(buckets):
        selected.append(
            min(
                buckets[key],
                key=lambda row: sha256_value(row["record_id"]),
            )
        )
    if len(selected) != EXPECTED_LIMITS["owner_sample_records"]:
        raise ValueError("safety/tool owner sample coverage changed")
    return json.loads(json.dumps(selected, ensure_ascii=False))


def to_candidate_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_revision: str,
) -> list[dict[str, Any]]:
    """Convert private validated rows while preserving AI provenance."""

    transform_sha256 = sha256_value(
        {"schema": "lumi.safety-tool-to-sft.v0.1", "source_revision": source_revision}
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
                    "content_sha256": sha256_value(messages),
                    "generation_method": "ai_generated",
                    "generator_id": row["generation"]["generator_id"],
                    "human_review_status": "owner_spot_check_pending",
                    "transform_sha256": transform_sha256,
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_method": "deterministic_safety_tool_v0.1_owner_pending",
                    "review_status": "accepted",
                    "score": row["automated_review"]["quality_score"],
                    "secret_scan": "passed",
                },
                "record_id": row["record_id"],
                "safety": {
                    "policy_version": "lumi-safety-v0.1",
                    "risk_labels": list(row["task_evidence"].get("risk_labels", [])),
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": EXPECTED_SOURCE["dataset_id"],
                    "document_id": row["record_id"],
                    "license_id": EXPECTED_SOURCE["license_id"],
                    "revision": source_revision,
                    "source_id": EXPECTED_SOURCE["source_id"],
                    "split": row["split"],
                },
                "task_category": row["task_category"],
            }
        )
    return records


def build_manifest(
    *,
    audit: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]],
    owner_sample: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "audit_report_sha256": audit["report_sha256"],
        "candidate_records": len(candidate_records),
        "candidate_sha256": sha256_value(candidate_records),
        "candidate_source_revision": audit["candidate_source_revision"],
        "owner_review": {
            "complete": False,
            "sample_records": len(owner_sample),
            "sample_sha256": sha256_value(owner_sample),
            "status": "pending",
        },
        "production_blockers": [
            "owner_spot_check_not_complete",
            "source_registry_entry_and_production_promotion_not_complete",
            "candidate_is_a_small_lane_pilot_not_the_complete_sft_mix",
        ],
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_in_git": False,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    return manifest


def build_evidence(
    *,
    audit: Mapping[str, Any],
    audit_file_sha256: str,
    candidate_file_sha256: str,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    owner_sample_file_sha256: str,
    plan_file_sha256: str,
    source_file_sha256: str,
) -> dict[str, Any]:
    for digest in (
        audit_file_sha256,
        candidate_file_sha256,
        manifest_file_sha256,
        owner_sample_file_sha256,
        plan_file_sha256,
        source_file_sha256,
    ):
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("safety/tool evidence binding is not SHA-256")
    evidence: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "audit_file_sha256": audit_file_sha256,
            "candidate_file_sha256": candidate_file_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "owner_sample_file_sha256": owner_sample_file_sha256,
            "plan_file_sha256": plan_file_sha256,
            "source_file_sha256": source_file_sha256,
        },
        "candidate_source_revision": audit["candidate_source_revision"],
        "counts": {
            "languages": audit["language_mode_counts"],
            "records": audit["records"],
            "skills": audit["skill_counts"],
            "splits": audit["split_counts"],
            "tasks": audit["task_counts"],
        },
        "decision": "machine_validated_candidate_waiting_for_owner_spot_check",
        "internal_bindings": {
            "audit_report_sha256": audit["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "owner_review_complete": False,
        "production_eligible": False,
        "production_promotions": 0,
        "quality_gate_results": audit["quality_gate_results"],
        "raw_training_text_retained": False,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
    }
    evidence["evidence_sha256"] = sha256_value(evidence)
    return evidence
