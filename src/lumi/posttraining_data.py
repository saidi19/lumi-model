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
SUPPORTED_ROLES = frozenset(
    {"assistant", "developer", "system", "tool", "tool_result", "user"}
)
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
        if (
            isinstance(maximum_message_characters, bool)
            or not isinstance(maximum_message_characters, int)
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
        if source_registry_path != "configs/lumi-4b-sft-source-registry-v0.1.json":
            raise ValueError("SFT source registry path changed")
        source_registry_sha256 = source_registry.get("sha256")
        if not _valid_sha256(source_registry_sha256):
            raise ValueError("SFT source registry SHA-256 is invalid")

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
    if split not in policy.allowed_splits:
        reasons.append("split")
    language = record.get("language")
    if language not in policy.allowed_languages:
        reasons.append("language")
    task_category = record.get("task_category")
    if task_category not in policy.task_mix:
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
        if not isinstance(dataset_id, str) or not dataset_id:
            reasons.append("source_dataset_id")
        if not isinstance(source.get("document_id"), str) or not source["document_id"]:
            reasons.append("source_document_id")
        if license_id not in policy.allowed_licenses:
            reasons.append("license")
        if not isinstance(revision, str) or (
            _GIT_SHA.fullmatch(revision) is None and _SHA256.fullmatch(revision) is None
        ):
            reasons.append("source_revision")
        if not isinstance(source.get("uri"), str) or not source["uri"]:
            reasons.append("source_uri")
        if source_registry is not None:
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

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        reasons.append("provenance")
    else:
        for field in ("content_sha256", "transform_sha256"):
            if not _valid_sha256(provenance.get(field)):
                reasons.append(f"provenance_{field}")
        if provenance.get("consent_basis") not in {
            "document_license",
            "first_party_rights_declaration",
            "public_domain",
            "source_level_open_license",
        }:
            reasons.append("consent_basis")

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
