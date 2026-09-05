"""Fail-closed schema and filtering for future Lumi supervised fine-tuning data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data_quality import normalize_for_matching

SFT_SCHEMA_VERSION = "lumi.sft.v0.1"
SUPPORTED_ROLES = frozenset({"assistant", "developer", "system", "tool", "tool_result", "user"})
TRAINING_ONLY_MARKERS = ("<|doc|>", "<|endofdoc|>")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_TOKEN_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*"
    r"['\"]?[A-Za-z0-9_./+=-]{12,}",
    re.IGNORECASE,
)

_LEGACY_SOURCE_REGISTRY_BINDING = (
    "configs/lumi-4b-sft-source-registry-v0.1.json",
    "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9",
)
_PRIVATE_RESEARCH_SOURCE_REGISTRY_BINDING = (
    "configs/lumi-4b-sft-source-registry-v0.24.json",
    "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
)
_PRIVATE_RESEARCH_INCLUDED_SOURCE_IDS = (
    "allenai_prosocial_dialog_safety_candidate",
    "amazon_massive_bn_bd_tool_canary",
    "banglabridge_instructions_closure_canary",
    "banglatlit_auxiliary_closure_canary",
    "coherelabs_aya_bengali_closure_canary",
    "coherelabs_aya_dataset",
    "deepmind_mathematics_dataset",
    "dmath",
    "fiscalnote_billsum_writing_candidate",
    "lumi_ai_assisted_gap_closure_canary_v0_1",
    "lumi_ai_assisted_gap_closure_scaled_v0_2",
    "lumi_ai_assisted_sft_v0_3",
    "lumi_first_party_gpt_oss_teacher_canary_v0_1",
    "lumi_first_party_multilingual_safety_tool_v0_6",
    "lumi_first_party_teacher_canary_v0_1",
    "lumi_safety_tool_sft_v0_1",
    "nvidia_nemotron_post_training_tool_canary",
    "nvidia_opencodeinstruct",
    "openassistant_oasst1",
    "openassistant_oasst2",
)
_PRIVATE_RESEARCH_EXCLUDED_SOURCE_IDS = (
    "bigcode_self_oss_instruct_50k",
    "huggingfaceh4_ultrachat_200k",
    "lumi_ai_assisted_sft_v0_1",
    "lumi_ai_assisted_sft_v0_2",
    "lumi_owned_sft_v0_1",
    "openr1_math_220k",
    "salesforce_xlam_function_calling_60k",
)
_PRIVATE_RESEARCH_AI_SOURCE_CONTRACTS = {
    "lumi_ai_assisted_gap_closure_canary_v0_1": {
        "generation_method": "ai_authored_templates_with_local_deterministic_renderer",
        "generator_id": "openai-codex-current-session",
        "human_review_completed": False,
        "human_written_from_scratch": False,
        "model_assistance_used": True,
        "owner_review_status": "not_required_for_nonproduction_canary",
        "runtime_external_model_api_calls": 0,
        "runtime_model_calls": 0,
    },
    "lumi_ai_assisted_gap_closure_scaled_v0_2": {
        "generation_method": "ai_authored_templates_with_local_deterministic_scaled_renderer",
        "generator_id": "openai-codex-current-session",
        "human_review_completed": False,
        "human_written_from_scratch": False,
        "model_assistance_used": True,
        "owner_review_status": "not_required_for_nonproduction_scaled_candidate",
        "runtime_external_model_api_calls": 0,
        "runtime_model_calls": 0,
    },
}
_PRIVATE_RESEARCH_MIX_BINDINGS = {
    "accepted_dataset_sha256": ("1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e"),
    "candidate_mix_report_sha256": (
        "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000"
    ),
    "candidate_mix_verification_sha256": (
        "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651"
    ),
}
_PRIVATE_RESEARCH_EVALUATION_SPLIT_PROHIBITIONS = {
    "deepmind_mathematics_dataset": ("test", "test_extra"),
    "dmath": ("test",),
    "fiscalnote_billsum_writing_candidate": ("ca_test", "test"),
}
_PRIVATE_RESEARCH_BENCHMARK_CONTRACT = {
    "anchor_ngram_tokens": 8,
    "minimum_anchor_matches": 2,
    "registry": "data/benchmark-signatures-v0.1.jsonl.gz",
    "registry_sha256": ("be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb"),
}
_PRIVATE_RESEARCH_PRESERVED_POLICY_SECTIONS = {
    "allowed_languages": ["bn", "de", "en", "es", "fr", "hi", "multilingual"],
    "allowed_licenses": [
        "apache-2.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "lumi-owned",
        "mit",
        "odc-by-1.0",
        "public-domain",
    ],
    "allowed_splits": ["train", "validation"],
    "benchmark_decontamination": _PRIVATE_RESEARCH_BENCHMARK_CONTRACT,
    "chat_training": {
        "assistant_only_loss": True,
        "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
        "context_window": 4096,
        "overlength_policy": "reject",
        "packing_policy": "single_conversation_until_block_diagonal_verified",
        "supervised_roles": ["assistant", "tool"],
    },
    "filters": {
        "maximum_message_characters": 65_536,
        "maximum_messages": 64,
        "minimum_quality_score": 0.75,
    },
    "task_mix": {
        "coding": 0.2,
        "conversation": 0.05,
        "factual_qa": 0.15,
        "multilingual_bengali": 0.1,
        "reasoning_math_science": 0.22,
        "safety_helpfulness": 0.08,
        "tool_api": 0.08,
        "writing_summarization": 0.12,
    },
}
_PRIVATE_RESEARCH_NOTES = {
    "ai_generated_records_require_explicit_generator_and_human_review_status": True,
    "ai_generated_records_require_owner_gate_before_production_promotion": True,
    "base_binding": "configs/lumi-4b-final-base-binding-v0.1.json",
    "private_research_machine_gap_rows_require_truthful_machine_provenance": True,
    "private_research_nonproduction_scope_active": True,
    "production_eligibility_remains_false_for_all_bound_sources": True,
    "raw_training_text_in_git": False,
    "target_tokens": "pending_base_checkpoint_evaluation",
    "task_mix_status": "planning_target_not_a_dataset_claim",
}
_PRIVATE_RESEARCH_AI_PROVENANCE_KEYS = frozenset(
    {
        "consent_basis",
        "content_sha256",
        "generation_method",
        "generator_id",
        "human_written_from_scratch",
        "model_assistance_used",
        "runtime_external_model_api_calls",
        "runtime_model_calls",
        "transform_sha256",
    }
)
_PRIVATE_RESEARCH_HUMAN_AUTHORSHIP_ALIASES = frozenset(
    {
        "fully_human_authored",
        "human_authored",
        "human_authorship_claimed",
        "human_authorship_verified",
        "human_written",
    }
)
_LEARNING_CANDIDATE_KEYS = frozenset(
    {
        "accepted_dataset_sha256",
        "ai_assisted_source_contracts",
        "candidate_mix_report_sha256",
        "candidate_mix_verification_sha256",
        "excluded_source_ids",
        "evaluation_split_prohibitions",
        "external_distribution_allowed",
        "included_source_ids",
        "machine_only_gap_rows",
        "production_dataset_seal_allowed",
        "production_sft_allowed",
        "registry_source_count",
        "scope",
        "seal_allowed",
    }
)


class BenchmarkMatcher(Protocol):
    """Subset of :class:`lumi.data_quality.BenchmarkRegistry` used here."""

    def match(self, text: str) -> tuple[str, str] | None: ...


class SourceRegistryMatcher(Protocol):
    """Subset of the source registry used to bind a candidate record."""

    def record_reasons(
        self,
        *,
        dataset_id: Any,
        language: Any,
        license_id: Any,
        revision: Any,
        source_id: Any,
        split: Any,
        task_category: Any,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class SFTPolicy:
    """Validated, immutable filtering policy."""

    allowed_languages: frozenset[str]
    allowed_licenses: frozenset[str]
    allowed_splits: frozenset[str]
    benchmark_decontamination: Mapping[str, Any]
    chat_template: str
    context_window: int
    maximum_message_characters: int
    maximum_messages: int
    minimum_quality_score: float
    packing_policy: str
    source_registry_path: str
    source_registry_required: bool
    source_registry_sha256: str
    supervised_roles: frozenset[str]
    task_mix: Mapping[str, float]
    learning_candidate: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SFTPolicy:
        if value.get("schema_version") != SFT_SCHEMA_VERSION:
            raise ValueError("unsupported SFT policy schema")

        allowed_languages = _string_set(value.get("allowed_languages"), "allowed_languages")
        allowed_licenses = _string_set(value.get("allowed_licenses"), "allowed_licenses")
        allowed_splits = _string_set(value.get("allowed_splits"), "allowed_splits")

        filters = value.get("filters")
        if not isinstance(filters, Mapping):
            raise ValueError("filters must be an object")
        minimum_quality_score = filters.get("minimum_quality_score")
        if (
            isinstance(minimum_quality_score, bool)
            or not isinstance(minimum_quality_score, (int, float))
            or not 0 <= float(minimum_quality_score) <= 1
        ):
            raise ValueError("minimum_quality_score must be between 0 and 1")
        maximum_messages = filters.get("maximum_messages")
        if isinstance(maximum_messages, bool) or not isinstance(maximum_messages, int):
            raise ValueError("maximum_messages must be an integer")
        if not 2 <= maximum_messages <= 256:
            raise ValueError("maximum_messages must be between 2 and 256")
        maximum_message_characters = filters.get("maximum_message_characters")
        if isinstance(maximum_message_characters, bool) or not isinstance(
            maximum_message_characters, int
        ):
            raise ValueError("maximum_message_characters must be an integer")
        if not 32 <= maximum_message_characters <= 1_000_000:
            raise ValueError("maximum_message_characters is outside the safe range")

        raw_mix = value.get("task_mix")
        if not isinstance(raw_mix, Mapping) or not raw_mix:
            raise ValueError("task_mix must be a non-empty object")
        task_mix: dict[str, float] = {}
        for task, weight in raw_mix.items():
            if not isinstance(task, str) or not task:
                raise ValueError("task_mix keys must be non-empty strings")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(f"task_mix weight must be positive for {task}")
            task_mix[task] = float(weight)
        if abs(sum(task_mix.values()) - 1.0) > 1e-9:
            raise ValueError("task_mix weights must sum to exactly 1")

        benchmark = value.get("benchmark_decontamination")
        if not isinstance(benchmark, Mapping):
            raise ValueError("benchmark_decontamination must be an object")
        if not isinstance(benchmark.get("registry"), str) or not benchmark["registry"]:
            raise ValueError("benchmark_decontamination.registry is required")
        if not _valid_sha256(benchmark.get("registry_sha256")):
            raise ValueError("benchmark_decontamination.registry_sha256 is invalid")

        chat_training = value.get("chat_training")
        if not isinstance(chat_training, Mapping):
            raise ValueError("chat_training must be an object")
        if chat_training.get("assistant_only_loss") is not True:
            raise ValueError("SFT must use assistant-only loss")
        if chat_training.get("chat_template") != "configs/lumi-4b-chat-template-v0.1.json":
            raise ValueError("SFT chat template binding changed")
        if chat_training.get("context_window") != 4096:
            raise ValueError("SFT context window must remain 4096")
        if chat_training.get("overlength_policy") != "reject":
            raise ValueError("overlength SFT records must fail closed")
        if (
            chat_training.get("packing_policy")
            != "single_conversation_until_block_diagonal_verified"
        ):
            raise ValueError("SFT packing may not cross conversation boundaries")
        raw_supervised_roles = chat_training.get("supervised_roles")
        if not isinstance(raw_supervised_roles, list) or set(raw_supervised_roles) != {
            "assistant",
            "tool",
        }:
            raise ValueError("SFT supervised roles must be assistant and tool")

        source_registry = value.get("source_registry")
        if not isinstance(source_registry, Mapping):
            raise ValueError("source_registry must be an object")
        if source_registry.get("schema_version") != "lumi.sft.sources.v0.1":
            raise ValueError("SFT source registry schema changed")
        if source_registry.get("required_for_filtering_and_seal") is not True:
            raise ValueError("SFT source registry must be required")
        source_registry_path = source_registry.get("path")
        source_registry_sha256 = source_registry.get("sha256")
        if not _valid_sha256(source_registry_sha256):
            raise ValueError("SFT source registry SHA-256 is invalid")
        learning_candidate = _validate_learning_candidate(value.get("learning_candidate"))
        if learning_candidate is not None and any(
            not _strict_json_equal(value.get(section), expected)
            for section, expected in _PRIVATE_RESEARCH_PRESERVED_POLICY_SECTIONS.items()
        ):
            raise ValueError("private-research preserved policy section changed")
        source_registry_binding = (source_registry_path, source_registry_sha256)
        expected_registry_binding = (
            _LEGACY_SOURCE_REGISTRY_BINDING
            if learning_candidate is None
            else _PRIVATE_RESEARCH_SOURCE_REGISTRY_BINDING
        )
        if source_registry_binding != expected_registry_binding:
            raise ValueError("SFT source registry binding changed")
        if learning_candidate is None:
            if "contract_revision" in value:
                raise ValueError("legacy SFT contract revision changed")
        else:
            notes = value.get("notes")
            if value.get("contract_revision") != "v0.2" or not _strict_json_equal(
                notes, _PRIVATE_RESEARCH_NOTES
            ):
                raise ValueError("private-research SFT contract notes changed")

        return cls(
            allowed_languages=frozenset(allowed_languages),
            allowed_licenses=frozenset(allowed_licenses),
            allowed_splits=frozenset(allowed_splits),
            benchmark_decontamination=dict(benchmark),
            chat_template=chat_training["chat_template"],
            context_window=4096,
            maximum_message_characters=maximum_message_characters,
            maximum_messages=maximum_messages,
            minimum_quality_score=float(minimum_quality_score),
            packing_policy=chat_training["packing_policy"],
            source_registry_path=source_registry_path,
            source_registry_required=True,
            source_registry_sha256=source_registry_sha256,
            supervised_roles=frozenset(raw_supervised_roles),
            task_mix=task_mix,
            learning_candidate=learning_candidate,
        )


@dataclass(frozen=True)
class SFTRecordDecision:
    """Raw-text-free decision for one candidate SFT record."""

    accepted: bool
    normalized_sha256: str
    reasons: tuple[str, ...]
    record_id: str | None
    response_sha256: str | None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} entries must be unique")
    return set(value)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equality aliasing."""

    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) is float and actual == expected
    return type(actual) is type(expected) and actual == expected


def _validate_learning_candidate(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _LEARNING_CANDIDATE_KEYS:
        raise ValueError("learning_candidate contract keys changed")
    expected_flags = {
        "external_distribution_allowed": False,
        "production_dataset_seal_allowed": False,
        "production_sft_allowed": False,
        "registry_source_count": 27,
        "scope": "private_research_nonproduction",
        "seal_allowed": True,
    }
    if any(
        not _strict_json_equal(value.get(key), expected) for key, expected in expected_flags.items()
    ):
        raise ValueError("learning_candidate scope changed")
    ai_policy = value.get("machine_only_gap_rows")
    if not _strict_json_equal(
        ai_policy,
        {
            "allowed": True,
            "completed_human_review_claim_allowed": False,
            "human_authorship_claim_allowed": False,
        },
    ):
        raise ValueError("learning_candidate machine-gap policy changed")
    included = value.get("included_source_ids")
    excluded = value.get("excluded_source_ids")
    if (
        not isinstance(included, list)
        or tuple(included) != _PRIVATE_RESEARCH_INCLUDED_SOURCE_IDS
        or not isinstance(excluded, list)
        or tuple(excluded) != _PRIVATE_RESEARCH_EXCLUDED_SOURCE_IDS
        or set(included) & set(excluded)
        or len(included) + len(excluded) != value["registry_source_count"]
    ):
        raise ValueError("learning_candidate source partition changed")
    for key, expected in _PRIVATE_RESEARCH_MIX_BINDINGS.items():
        if value.get(key) != expected:
            raise ValueError("learning_candidate candidate-mix binding changed")
    ai_contracts = value.get("ai_assisted_source_contracts")
    if not _strict_json_equal(ai_contracts, _PRIVATE_RESEARCH_AI_SOURCE_CONTRACTS):
        raise ValueError("learning_candidate AI source contract changed")
    evaluation_prohibitions = value.get("evaluation_split_prohibitions")
    if (
        not isinstance(evaluation_prohibitions, Mapping)
        or {
            source_id: tuple(splits)
            for source_id, splits in evaluation_prohibitions.items()
            if isinstance(splits, list)
        }
        != _PRIVATE_RESEARCH_EVALUATION_SPLIT_PROHIBITIONS
        or len(evaluation_prohibitions) != len(_PRIVATE_RESEARCH_EVALUATION_SPLIT_PROHIBITIONS)
    ):
        raise ValueError("learning_candidate evaluation exclusions changed")
    return {
        "accepted_dataset_sha256": value["accepted_dataset_sha256"],
        "ai_assisted_source_contracts": {
            source_id: dict(contract) for source_id, contract in sorted(ai_contracts.items())
        },
        "candidate_mix_report_sha256": value["candidate_mix_report_sha256"],
        "candidate_mix_verification_sha256": value["candidate_mix_verification_sha256"],
        "excluded_source_ids": frozenset(excluded),
        "evaluation_split_prohibitions": {
            source_id: frozenset(splits) for source_id, splits in evaluation_prohibitions.items()
        },
        "external_distribution_allowed": False,
        "included_source_ids": frozenset(included),
        "machine_only_gap_rows": dict(ai_policy),
        "production_dataset_seal_allowed": False,
        "production_sft_allowed": False,
        "registry_source_count": 27,
        "scope": "private_research_nonproduction",
        "seal_allowed": True,
    }


def load_sft_policy(path: str | Path) -> SFTPolicy:
    """Load and validate a policy JSON document."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("SFT policy must be a JSON object")
    return SFTPolicy.from_mapping(value)


def validate_message_sequence(messages: Sequence[Mapping[str, str]]) -> None:
    """Validate one chat/tool trajectory without inspecting message content."""

    if len(messages) < 2:
        raise ValueError("message_count_out_of_bounds")
    state = "prefix_or_user"
    seen_system = False
    seen_developer = False
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("invalid_message")
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            raise ValueError("unsupported_role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty_message")

        if state == "prefix_or_user" and role in {"system", "developer"}:
            if role == "system":
                if seen_system or seen_developer:
                    raise ValueError("invalid_turn_order")
                seen_system = True
            else:
                if seen_developer:
                    raise ValueError("invalid_turn_order")
                seen_developer = True
            continue

        if state == "prefix_or_user":
            if role != "user":
                raise ValueError("invalid_turn_order")
            state = "model_output"
        elif state == "model_output":
            if role == "assistant":
                state = "user_or_end"
            elif role == "tool":
                state = "tool_result"
            else:
                raise ValueError("invalid_turn_order")
        elif state == "tool_result":
            if role != "tool_result":
                raise ValueError("invalid_turn_order")
            state = "model_output"
        elif state == "user_or_end":
            if role != "user":
                raise ValueError("invalid_turn_order")
            state = "model_output"
        else:  # pragma: no cover - state values are closed above
            raise AssertionError(state)

    if messages[-1].get("role") != "assistant" or state != "user_or_end":
        raise ValueError("conversation_must_end_with_assistant")


def _record_messages(value: Any, policy: SFTPolicy, reasons: list[str]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        reasons.append("invalid_messages")
        return []
    if not 2 <= len(value) <= policy.maximum_messages:
        reasons.append("message_count_out_of_bounds")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            reasons.append("invalid_message")
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in SUPPORTED_ROLES:
            reasons.append("unsupported_role")
            continue
        if not isinstance(content, str) or not content.strip():
            reasons.append("empty_message")
            continue
        if len(content) > policy.maximum_message_characters:
            reasons.append("message_too_long")
        messages.append({"role": role, "content": content})

    if len(messages) != len(value):
        return messages

    if messages:
        try:
            validate_message_sequence(messages)
        except ValueError as error:
            reasons.append(str(error))
    return messages


def _scan_sensitive_content(text: str, reasons: list[str]) -> None:
    if any(marker in text for marker in TRAINING_ONLY_MARKERS):
        reasons.append("training_marker_literal")
    if _PRIVATE_KEY.search(text) or _AWS_ACCESS_KEY.search(text) or _TOKEN_ASSIGNMENT.search(text):
        reasons.append("secret_pattern")
    if _EMAIL.search(text) or _PHONE.search(text):
        reasons.append("pii_pattern")


def evaluate_sft_record(
    record: Any,
    *,
    policy: SFTPolicy,
    benchmark_registry: BenchmarkMatcher | None = None,
    source_registry: SourceRegistryMatcher | None = None,
) -> SFTRecordDecision:
    """Evaluate one record without retaining its raw text in the decision."""

    reasons: list[str] = []
    if not isinstance(record, Mapping):
        return SFTRecordDecision(
            accepted=False,
            normalized_sha256=_sha256(b"invalid-record"),
            reasons=("record_not_object",),
            record_id=None,
            response_sha256=None,
        )

    if record.get("schema_version") != SFT_SCHEMA_VERSION:
        reasons.append("schema_version")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        reasons.append("record_id")
        record_id = None
    split = record.get("split")
    if not isinstance(split, str) or split not in policy.allowed_splits:
        reasons.append("split")
    language = record.get("language")
    if not isinstance(language, str) or language not in policy.allowed_languages:
        reasons.append("language")
    task_category = record.get("task_category")
    if not isinstance(task_category, str) or task_category not in policy.task_mix:
        reasons.append("task_category")

    source = record.get("source")
    source_id: Any = None
    dataset_id: Any = None
    license_id: Any = None
    revision: Any = None
    if not isinstance(source, Mapping):
        reasons.append("source")
    else:
        source_id = source.get("source_id")
        dataset_id = source.get("dataset_id")
        license_id = source.get("license_id")
        revision = source.get("revision")
        if not isinstance(source_id, str) or not source_id:
            reasons.append("source_id")
        if policy.learning_candidate is not None and (
            not isinstance(source_id, str)
            or source_id not in policy.learning_candidate["included_source_ids"]
        ):
            reasons.append("learning_candidate_source_not_included")
        if (
            policy.learning_candidate is not None
            and isinstance(source_id, str)
            and isinstance(split, str)
            and split
            in policy.learning_candidate["evaluation_split_prohibitions"].get(
                source_id, frozenset()
            )
        ):
            reasons.append("learning_candidate_evaluation_split_prohibited")
        if not isinstance(dataset_id, str) or not dataset_id:
            reasons.append("source_dataset_id")
        if not isinstance(source.get("document_id"), str) or not source["document_id"]:
            reasons.append("source_document_id")
        if not isinstance(license_id, str) or license_id not in policy.allowed_licenses:
            reasons.append("license")
        if not isinstance(revision, str) or (
            _GIT_SHA.fullmatch(revision) is None and _SHA256.fullmatch(revision) is None
        ):
            reasons.append("source_revision")
        if not isinstance(source.get("uri"), str) or not source["uri"]:
            reasons.append("source_uri")
        if source_registry is not None and all(
            isinstance(item, str)
            for item in (
                dataset_id,
                language,
                license_id,
                revision,
                source_id,
                split,
                task_category,
            )
        ):
            reasons.extend(
                source_registry.record_reasons(
                    dataset_id=dataset_id,
                    language=language,
                    license_id=license_id,
                    revision=revision,
                    source_id=source_id,
                    split=split,
                    task_category=task_category,
                )
            )
        elif source_registry is not None:
            reasons.append("source_registry_record_invalid")

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("provenance")
    else:
        for field in ("content_sha256", "transform_sha256"):
            if not _valid_sha256(provenance.get(field)):
                reasons.append(f"provenance_{field}")
        consent_basis = provenance.get("consent_basis")
        if not isinstance(consent_basis, str) or consent_basis not in {
            "document_license",
            "first_party_rights_declaration",
            "public_domain",
            "source_level_open_license",
        }:
            reasons.append("consent_basis")

    if policy.learning_candidate is not None:
        ai_contract = (
            policy.learning_candidate["ai_assisted_source_contracts"].get(source_id)
            if isinstance(source_id, str)
            else None
        )
        if ai_contract is not None:
            if not isinstance(provenance, Mapping):
                reasons.append("learning_candidate_ai_provenance")
            else:
                if set(provenance) != _PRIVATE_RESEARCH_AI_PROVENANCE_KEYS:
                    reasons.append("learning_candidate_ai_provenance_keys")
                for field in ("generation_method", "generator_id"):
                    if not _strict_json_equal(provenance.get(field), ai_contract[field]):
                        reasons.append(f"learning_candidate_ai_provenance:{field}")
                for field in ("human_written_from_scratch", "model_assistance_used"):
                    if provenance.get(field) is not ai_contract[field]:
                        reasons.append(f"learning_candidate_ai_provenance:{field}")
                for field in (
                    "runtime_external_model_api_calls",
                    "runtime_model_calls",
                ):
                    if (
                        type(provenance.get(field)) is not int
                        or provenance.get(field) != ai_contract[field]
                    ):
                        reasons.append(f"learning_candidate_ai_provenance:{field}")
                if set(provenance) & _PRIVATE_RESEARCH_HUMAN_AUTHORSHIP_ALIASES:
                    reasons.append("learning_candidate_ai_human_authorship_claim")
            owner_review = record.get("owner_review")
            expected_owner_review = {
                "human_review_completed": ai_contract["human_review_completed"],
                "status": ai_contract["owner_review_status"],
            }
            if not _strict_json_equal(owner_review, expected_owner_review):
                reasons.append("learning_candidate_ai_owner_review")

    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        reasons.append("quality")
    else:
        score = quality.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 1
        ):
            reasons.append("quality_score")
        elif float(score) < policy.minimum_quality_score:
            reasons.append("quality_score_below_floor")
        if quality.get("review_status") != "accepted":
            reasons.append("review_status")
        if quality.get("pii_scan") != "passed":
            reasons.append("pii_scan")
        if quality.get("secret_scan") != "passed":
            reasons.append("secret_scan")

    safety = record.get("safety")
    if not isinstance(safety, Mapping):
        reasons.append("safety")
    else:
        risk_labels = safety.get("risk_labels")
        if not isinstance(risk_labels, list) or any(
            not isinstance(label, str) or not label for label in risk_labels
        ):
            reasons.append("risk_labels")
        if not isinstance(safety.get("policy_version"), str) or not safety["policy_version"]:
            reasons.append("safety_policy_version")

    messages = _record_messages(record.get("messages"), policy, reasons)
    combined_text = "\n".join(message["content"] for message in messages)
    _scan_sensitive_content(combined_text, reasons)

    normalized_messages = [
        {
            "content": normalize_for_matching(message["content"]),
            "role": message["role"],
        }
        for message in messages
    ]
    identity = {
        "language": language,
        "messages": normalized_messages,
        "task_category": task_category,
    }
    normalized_sha256 = _sha256(_canonical_json(identity))
    response_sha256 = (
        _sha256(normalize_for_matching(messages[-1]["content"]).encode())
        if messages and messages[-1]["role"] == "assistant"
        else None
    )

    if benchmark_registry is not None and combined_text:
        match = benchmark_registry.match(combined_text)
        if match is not None:
            reasons.append(f"benchmark_contamination:{match[0]}")

    return SFTRecordDecision(
        accepted=not reasons,
        normalized_sha256=normalized_sha256,
        reasons=tuple(dict.fromkeys(reasons)),
        record_id=record_id,
        response_sha256=response_sha256,
    )


def filter_sft_records(
    records: Iterable[Any],
    *,
    policy: SFTPolicy,
    benchmark_registry: BenchmarkMatcher | None = None,
    source_registry: SourceRegistryMatcher | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Filter a stream and emit accepted records plus a raw-text-free report."""

    accepted: list[Mapping[str, Any]] = []
    decisions: list[SFTRecordDecision] = []
    seen_ids: set[str] = set()
    seen_normalized: set[str] = set()
    rejection_counts: Counter[str] = Counter()

    for record in records:
        decision = evaluate_sft_record(
            record,
            policy=policy,
            benchmark_registry=benchmark_registry,
            source_registry=source_registry,
        )
        reasons = list(decision.reasons)
        if decision.record_id is not None and decision.record_id in seen_ids:
            reasons.append("duplicate_record_id")
        if decision.normalized_sha256 in seen_normalized:
            reasons.append("duplicate_normalized_conversation")
        if decision.record_id is not None:
            seen_ids.add(decision.record_id)
        seen_normalized.add(decision.normalized_sha256)

        final = SFTRecordDecision(
            accepted=not reasons,
            normalized_sha256=decision.normalized_sha256,
            reasons=tuple(dict.fromkeys(reasons)),
            record_id=decision.record_id,
            response_sha256=decision.response_sha256,
        )
        decisions.append(final)
        if final.accepted:
            accepted.append(record)
        else:
            rejection_counts.update(final.reasons)

    accepted_hashes = sorted(item.normalized_sha256 for item in decisions if item.accepted)
    decision_ledger = [
        {
            "accepted": item.accepted,
            "normalized_sha256": item.normalized_sha256,
            "reasons": list(item.reasons),
            "record_id": item.record_id,
            "response_sha256": item.response_sha256,
        }
        for item in decisions
    ]
    report = {
        "accepted": len(accepted),
        "accepted_dataset_sha256": _sha256(_canonical_json(accepted_hashes)),
        "decisions": decision_ledger,
        "raw_text_retained": False,
        "rejected": len(decisions) - len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "schema_version": SFT_SCHEMA_VERSION,
        "total": len(decisions),
    }
    report["report_sha256"] = _sha256(_canonical_json(report))
    return accepted, report
