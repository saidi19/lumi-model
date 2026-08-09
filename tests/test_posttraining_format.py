from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lumi.posttraining_format import (
    encode_sft_messages,
    load_chat_template,
    pad_sft_batch,
    verify_chat_tokenizer,
)

ROOT = Path(__file__).parents[1]
TEMPLATE_PATH = ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"


class FakeTokenizer:
    def __init__(self, encoded: dict[str, list[int]] | None = None) -> None:
        self.encoded = encoded or {}
        self.all_special_ids = list(range(11))
        self.vocabulary = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|doc|>": 3,
            "<|unk|>": 4,
            "<|system|>": 5,
            "<|user|>": 6,
            "<|assistant|>": 7,
            "<|tool|>": 8,
            "<|tool_result|>": 9,
            "<|message_end|>": 10,
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        assert split_special_tokens is True
        return list(self.encoded.get(text, [100 + len(text)]))


def test_chat_format_masks_every_prompt_and_role_token() -> None:
    template = load_chat_template(TEMPLATE_PATH)
    tokenizer = FakeTokenizer({"Question": [101], "Answer": [102, 103]})
    encoded = encode_sft_messages(
        [
            {"content": "Question", "role": "user"},
            {"content": "Answer", "role": "assistant"},
        ],
        tokenizer=tokenizer,
        template=template,
    )
    assert encoded.input_ids == (1, 6, 101, 10, 7, 102, 103, 10, 2)
    assert encoded.labels == (-100, -100, -100, -100, -100, 102, 103, 10, 2)
    assert encoded.supervised_tokens == 4
    assert encoded.masked_tokens == 5


def test_tool_call_is_supervised_but_tool_result_is_masked() -> None:
    template = load_chat_template(TEMPLATE_PATH)
    tokenizer = FakeTokenizer(
        {
            "Question": [101],
            '{"query":"weather"}': [201, 202],
            '{"temperature":30}': [301, 302],
            "It is 30 C.": [401, 402],
        }
    )
    encoded = encode_sft_messages(
        [
            {"content": "Question", "role": "user"},
            {"content": '{"query":"weather"}', "role": "tool"},
            {"content": '{"temperature":30}', "role": "tool_result"},
            {"content": "It is 30 C.", "role": "assistant"},
        ],
        tokenizer=tokenizer,
        template=template,
    )
    tool_start = encoded.input_ids.index(8)
    result_start = encoded.input_ids.index(9)
    assistant_start = encoded.input_ids.index(7)
    assert encoded.labels[tool_start] == -100
    assert encoded.labels[tool_start + 1 : result_start] == (201, 202, 10)
    assert all(label == -100 for label in encoded.labels[result_start:assistant_start])
    assert encoded.labels[assistant_start + 1 :] == (401, 402, 10, 2)


def test_invalid_or_overlength_conversation_fails_closed() -> None:
    template = load_chat_template(TEMPLATE_PATH)
    tokenizer = FakeTokenizer({"huge": [100] * 4090})
    with pytest.raises(ValueError, match="invalid_turn_order"):
        encode_sft_messages(
            [
                {"content": "Answer first", "role": "assistant"},
                {"content": "Question later", "role": "user"},
            ],
            tokenizer=tokenizer,
            template=template,
        )
    with pytest.raises(ValueError, match="maximum is 4096"):
        encode_sft_messages(
            [
                {"content": "Question", "role": "user"},
                {"content": "huge", "role": "assistant"},
            ],
            tokenizer=tokenizer,
            template=template,
        )


def test_tokenizer_reserved_ids_must_match_the_frozen_contract() -> None:
    tokenizer = FakeTokenizer()
    template = load_chat_template(TEMPLATE_PATH)
    verify_chat_tokenizer(tokenizer, template)
    changed = copy.deepcopy(tokenizer)
    changed.vocabulary["<|assistant|>"] = 999
    with pytest.raises(ValueError, match="chat token changed"):
        verify_chat_tokenizer(changed, template)


def test_batch_padding_never_contributes_to_loss() -> None:
    template = load_chat_template(TEMPLATE_PATH)
    tokenizer = FakeTokenizer({"Q": [101], "short": [201], "long": [301, 302, 303]})
    short = encode_sft_messages(
        [{"content": "Q", "role": "user"}, {"content": "short", "role": "assistant"}],
        tokenizer=tokenizer,
        template=template,
    )
    long = encode_sft_messages(
        [{"content": "Q", "role": "user"}, {"content": "long", "role": "assistant"}],
        tokenizer=tokenizer,
        template=template,
    )
    batch = pad_sft_batch([short, long], template=template, pad_to_multiple_of=8)
    assert {len(row) for row in batch["input_ids"]} == {16}
    assert batch["input_ids"][0][-1] == 0
    assert batch["labels"][0][-1] == -100
    assert batch["attention_mask"][0][-1] == 0
    assert batch["attention_mask"][1][: len(long.input_ids)] == [1] * len(long.input_ids)
