from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from lumi.posttraining_format import ChatTemplate
from lumi.posttraining_oasst1_curation import SOURCE_REVISION
from lumi.posttraining_oasst1_full_run import (
    build_sft_records,
    deterministic_owner_sample,
    download_exact_object,
    measure_token_fit,
)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._url = url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._url


class _Tokenizer:
    def __init__(self) -> None:
        self._vocab = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|system|>": 5,
            "<|user|>": 6,
            "<|assistant|>": 7,
            "<|tool|>": 8,
            "<|tool_result|>": 9,
            "<|message_end|>": 10,
        }
        self.all_special_ids = list(self._vocab.values())

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        assert split_special_tokens is True
        return [100 + index for index, _ in enumerate(text.split())]


def _template() -> ChatTemplate:
    return ChatTemplate.from_mapping(
        {
            "append_eos": True,
            "content_encoding": "untrusted_reserved_token_safe",
            "context_window": 4096,
            "label_ignore_index": -100,
            "loss_roles": ["assistant", "tool"],
            "mask_role_tokens": True,
            "overlength_policy": "reject",
            "role_tokens": {
                "assistant": {"id": 7, "token": "<|assistant|>"},
                "developer": {"id": 5, "token": "<|system|>"},
                "system": {"id": 5, "token": "<|system|>"},
                "tool": {"id": 8, "token": "<|tool|>"},
                "tool_result": {"id": 9, "token": "<|tool_result|>"},
                "user": {"id": 6, "token": "<|user|>"},
            },
            "schema_version": "lumi.chat-template.v0.1",
            "sequence_tokens": {
                "bos": {"id": 1, "token": "<|bos|>"},
                "eos": {"id": 2, "token": "<|eos|>"},
                "message_end": {"id": 10, "token": "<|message_end|>"},
                "pad": {"id": 0, "token": "<|pad|>"},
            },
        }
    )


def _candidate(record_id: str = "openassistant_oasst1-test") -> dict[str, Any]:
    return {
        "answer_quality": 0.9,
        "attribution": {
            "answer_message_id": "answer",
            "answer_row_index": 1,
            "content_sha256": "a" * 64,
            "dataset_id": "OpenAssistant/oasst1",
            "license_id": "apache-2.0",
            "prompt_message_id": "prompt",
            "prompt_row_index": 0,
            "record_id": record_id,
            "revision": SOURCE_REVISION,
            "schema_version": "lumi.sft-oasst1-attribution.v0.1",
            "source_id": "openassistant_oasst1",
            "source_object_path": "data/train-00000-of-00001-b42a775f407cee45.parquet",
            "source_object_sha256": None,
            "split": "train",
            "transform_sha256": "b" * 64,
        },
        "language": "en",
        "messages": (
            {"content": "How are you?", "role": "user"},
            {"content": "I am well.", "role": "assistant"},
        ),
        "prompt_quality": 0.8,
        "record_id": record_id,
        "split": "train",
        "task_category": "conversation",
    }


def test_bounded_download_is_atomic(tmp_path: Path) -> None:
    payload = b"apache-license"
    destination = tmp_path / "LICENSE"

    result = download_exact_object(
        destination=destination,
        maximum_bytes=len(payload),
        object_path="LICENSE",
        open_url=lambda request: _Response(payload, request.full_url),
    )

    assert destination.read_bytes() == payload
    assert result["bytes"] == len(payload)
    assert result["redirect_host"] == "huggingface.co"
    assert not (tmp_path / ".LICENSE.tmp").exists()


def test_bounded_download_rejects_oversize(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="remaining network byte limit"):
        download_exact_object(
            destination=tmp_path / "LICENSE",
            maximum_bytes=3,
            object_path="LICENSE",
            open_url=lambda request: _Response(b"four", request.full_url),
        )
    assert not (tmp_path / "LICENSE").exists()


def test_candidate_conversion_and_token_measurement() -> None:
    candidate = _candidate()
    records, attributions = build_sft_records(
        [candidate],
        object_sha256_by_path={
            candidate["attribution"]["source_object_path"]: "c" * 64,
        },
    )
    accepted, metrics = measure_token_fit(
        records,
        tokenizer=_Tokenizer(),
        template=_template(),
    )

    assert len(accepted) == 1
    assert len(attributions) == 1
    assert attributions[0]["source_object_sha256"] == "c" * 64
    assert metrics["accepted_records"] == 1
    assert metrics["supervised_tokens"] > 0
    assert metrics["masked_tokens"] > 0
    assert metrics["assistant_only_loss_verified"] is True


def test_owner_sample_is_deterministic_and_bounded() -> None:
    records = []
    for index in range(40):
        candidate = _candidate(f"openassistant_oasst1-{index:02d}")
        converted, _ = build_sft_records(
            [candidate],
            object_sha256_by_path={
                candidate["attribution"]["source_object_path"]: "c" * 64,
            },
        )
        converted[0]["task_category"] = ("conversation", "factual_qa")[index % 2]
        records.extend(converted)

    first = deterministic_owner_sample(records, maximum_records=32)
    second = deterministic_owner_sample(list(reversed(records)), maximum_records=32)

    assert len(first) == 32
    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]

