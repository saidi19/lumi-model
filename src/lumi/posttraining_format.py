"""Exact Lumi chat serialization and assistant-only SFT loss masking."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .posttraining_data import validate_message_sequence
from .tokenizer_gate import encode_untrusted_text

CHAT_TEMPLATE_SCHEMA_VERSION = "lumi.chat-template.v0.1"
EXPECTED_CONTEXT_WINDOW = 4096
EXPECTED_ROLE_TOKENS = {
    "assistant": (7, "<|assistant|>"),
    "developer": (5, "<|system|>"),
    "system": (5, "<|system|>"),
    "tool": (8, "<|tool|>"),
    "tool_result": (9, "<|tool_result|>"),
    "user": (6, "<|user|>"),
}
EXPECTED_SEQUENCE_TOKENS = {
    "bos": (1, "<|bos|>"),
    "eos": (2, "<|eos|>"),
    "message_end": (10, "<|message_end|>"),
    "pad": (0, "<|pad|>"),
}


class ChatTokenizer(Protocol):
    all_special_ids: Sequence[int]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]: ...

    def get_vocab(self) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class ChatTemplate:
    context_window: int
    ignore_index: int
    loss_roles: frozenset[str]
    role_token_ids: Mapping[str, int]
    role_token_literals: Mapping[str, str]
    sequence_token_ids: Mapping[str, int]
    sequence_token_literals: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChatTemplate:
        if value.get("schema_version") != CHAT_TEMPLATE_SCHEMA_VERSION:
            raise ValueError("unsupported chat template schema")
        if value.get("context_window") != EXPECTED_CONTEXT_WINDOW:
            raise ValueError("chat template must preserve the exact 4096-token context")
        if value.get("append_eos") is not True:
            raise ValueError("chat template must append EOS")
        if value.get("mask_role_tokens") is not True:
            raise ValueError("chat role tokens must remain loss-masked")
        if value.get("overlength_policy") != "reject":
            raise ValueError("overlength SFT records must fail closed")
        if value.get("content_encoding") != "untrusted_reserved_token_safe":
            raise ValueError("message content must use reserved-token-safe encoding")
        if value.get("label_ignore_index") != -100:
            raise ValueError("label ignore index must be -100")

        role_ids, role_literals = _validate_token_map(
            value.get("role_tokens"), EXPECTED_ROLE_TOKENS, "role_tokens"
        )
        sequence_ids, sequence_literals = _validate_token_map(
            value.get("sequence_tokens"), EXPECTED_SEQUENCE_TOKENS, "sequence_tokens"
        )
        raw_loss_roles = value.get("loss_roles")
        if not isinstance(raw_loss_roles, list) or set(raw_loss_roles) != {"assistant", "tool"}:
            raise ValueError("loss_roles must be exactly assistant and tool")
        return cls(
            context_window=EXPECTED_CONTEXT_WINDOW,
            ignore_index=-100,
            loss_roles=frozenset(raw_loss_roles),
            role_token_ids=role_ids,
            role_token_literals=role_literals,
            sequence_token_ids=sequence_ids,
            sequence_token_literals=sequence_literals,
        )


@dataclass(frozen=True)
class EncodedSFTConversation:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_tokens: int

    @property
    def masked_tokens(self) -> int:
        return len(self.labels) - self.supervised_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "masked_tokens": self.masked_tokens,
            "supervised_tokens": self.supervised_tokens,
            "total_tokens": len(self.input_ids),
        }


def pad_sft_batch(
    examples: Sequence[EncodedSFTConversation],
    *,
    template: ChatTemplate,
    pad_to_multiple_of: int = 1,
) -> dict[str, list[list[int]]]:
    """Right-pad isolated conversations while keeping every pad label masked."""

    if not examples:
        raise ValueError("SFT batch must contain at least one conversation")
    if isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, int):
        raise ValueError("pad_to_multiple_of must be an integer")
    if pad_to_multiple_of < 1 or pad_to_multiple_of > template.context_window:
        raise ValueError("pad_to_multiple_of is outside the safe range")
    maximum = max(len(example.input_ids) for example in examples)
    padded_length = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * (
        pad_to_multiple_of
    )
    if padded_length > template.context_window:
        raise ValueError("padded SFT batch exceeds the context window")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for example in examples:
        if len(example.input_ids) != len(example.labels):
            raise ValueError("SFT input and label lengths differ")
        padding = padded_length - len(example.input_ids)
        input_rows.append(
            [*example.input_ids, *([template.sequence_token_ids["pad"]] * padding)]
        )
        label_rows.append([*example.labels, *([template.ignore_index] * padding)])
        attention_rows.append([*([1] * len(example.input_ids)), *([0] * padding)])
    return {
        "attention_mask": attention_rows,
        "input_ids": input_rows,
        "labels": label_rows,
    }


def _validate_token_map(
    value: Any,
    expected: Mapping[str, tuple[int, str]],
    field: str,
) -> tuple[dict[str, int], dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{field} changed")
    ids: dict[str, int] = {}
    literals: dict[str, str] = {}
    for name, (expected_id, expected_literal) in expected.items():
        specification = value.get(name)
        if not isinstance(specification, Mapping):
            raise ValueError(f"{field}.{name} must be an object")
        if specification.get("id") != expected_id or specification.get("token") != expected_literal:
            raise ValueError(f"{field}.{name} changed")
        ids[name] = expected_id
        literals[name] = expected_literal
    return ids, literals


def load_chat_template(path: str | Path) -> ChatTemplate:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("chat template must be a JSON object")
    return ChatTemplate.from_mapping(value)


def verify_chat_tokenizer(tokenizer: ChatTokenizer, template: ChatTemplate) -> None:
    """Require the frozen tokenizer to expose every reserved chat ID exactly."""

    vocabulary = tokenizer.get_vocab()
    expected = {
        **dict(
            zip(
                template.role_token_literals.values(),
                template.role_token_ids.values(),
                strict=True,
            )
        ),
        **dict(
            zip(
                template.sequence_token_literals.values(),
                template.sequence_token_ids.values(),
                strict=True,
            )
        ),
    }
    for literal, token_id in expected.items():
        if vocabulary.get(literal) != token_id:
            raise ValueError(f"tokenizer chat token changed: {literal}")
    if not set(expected.values()).issubset(set(tokenizer.all_special_ids)):
        raise ValueError("tokenizer does not mark every chat token as special")


def encode_sft_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer: ChatTokenizer,
    template: ChatTemplate,
) -> EncodedSFTConversation:
    """Serialize one conversation and supervise only assistant/tool output tokens."""

    validate_message_sequence(messages)
    verify_chat_tokenizer(tokenizer, template)

    input_ids = [template.sequence_token_ids["bos"]]
    labels = [template.ignore_index]
    for message in messages:
        role = message["role"]
        content_ids = encode_untrusted_text(tokenizer, message["content"])
        if not content_ids:
            raise ValueError("message content encoded to zero tokens")
        supervised = role in template.loss_roles

        input_ids.append(template.role_token_ids[role])
        labels.append(template.ignore_index)
        input_ids.extend(content_ids)
        labels.extend(content_ids if supervised else [template.ignore_index] * len(content_ids))
        input_ids.append(template.sequence_token_ids["message_end"])
        labels.append(
            template.sequence_token_ids["message_end"] if supervised else template.ignore_index
        )

    input_ids.append(template.sequence_token_ids["eos"])
    labels.append(template.sequence_token_ids["eos"])
    if len(input_ids) > template.context_window:
        raise ValueError(
            f"SFT conversation has {len(input_ids)} tokens; maximum is {template.context_window}"
        )
    supervised_tokens = sum(label != template.ignore_index for label in labels)
    if supervised_tokens < 2:
        raise ValueError("SFT conversation has no trainable assistant response")
    return EncodedSFTConversation(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        supervised_tokens=supervised_tokens,
    )
