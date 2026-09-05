from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_format import load_chat_template
from lumi.posttraining_seal import build_sft_dataset_seal
from lumi.posttraining_shards import materialize_sft_shards, verify_sft_shards
from lumi.posttraining_sources import SFTSource, SFTSourceRegistry, load_sft_source_registry

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "configs" / "lumi-4b-sft-data-contract-v0.1.json"
BASE_BINDING = json.loads(
    (ROOT / "configs" / "lumi-4b-final-base-binding-v0.1.json").read_text()
)


class NeverMatch:
    def match(self, text: str):
        assert text
        return None


class FakeTokenizer:
    all_special_ids = list(range(11))

    def get_vocab(self):
        return {
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

    def encode(self, text: str, *, add_special_tokens: bool, split_special_tokens: bool):
        assert add_special_tokens is False
        assert split_special_tokens is True
        return [100 + len(text), 200 + len(text)]


def _record(record_id: str, split: str) -> dict:
    return {
        "language": "en",
        "messages": [
            {"content": f"Question {record_id}", "role": "user"},
            {"content": f"Answer {record_id}", "role": "assistant"},
        ],
        "provenance": {
            "consent_basis": "document_license",
            "content_sha256": "0" * 64,
            "transform_sha256": "1" * 64,
        },
        "quality": {
            "pii_scan": "passed",
            "review_status": "accepted",
            "score": 0.95,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {"policy_version": "lumi-safety-v0.1", "risk_labels": []},
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": "OpenAssistant/oasst1",
            "document_id": f"doc-{record_id}",
            "license_id": "apache-2.0",
            "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
            "source_id": "openassistant_oasst1",
            "uri": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        },
        "split": split,
        "task_category": "conversation",
    }


def _registry() -> SFTSourceRegistry:
    policy = load_sft_policy(POLICY_PATH)
    loaded = load_sft_source_registry(
        ROOT / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    source = loaded.by_id["openassistant_oasst1"]
    promoted = SFTSource(
        **{
            **source.__dict__,
            "production_blockers": (),
            "production_eligible": True,
        }
    )
    return SFTSourceRegistry(
        observed_at=loaded.observed_at,
        registry_sha256=loaded.registry_sha256,
        sources=(promoted,),
    )


def _seal(records: list[dict], tokenizer: FakeTokenizer) -> dict:
    policy = load_sft_policy(POLICY_PATH)
    return build_sft_dataset_seal(
        records,
        base_binding=BASE_BINDING,
        benchmark_registry=NeverMatch(),
        policy=policy,
        policy_sha256="2" * 64,
        source_registry=_registry(),
        template=load_chat_template(ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"),
        tokenizer=tokenizer,
    )


def test_materialized_shards_match_seal_and_keep_prompt_labels_masked(tmp_path) -> None:
    records = [_record("train-one", "train"), _record("valid-one", "validation")]
    tokenizer = FakeTokenizer()
    template = load_chat_template(ROOT / "configs" / "lumi-4b-chat-template-v0.1.json")
    seal = _seal(records, tokenizer)
    output = tmp_path / "encoded"
    manifest = materialize_sft_shards(
        records,
        output=output,
        seal=seal,
        template=template,
        tokenizer=tokenizer,
        maximum_tokens_per_shard=4096,
    )

    assert verify_sft_shards(output) == manifest
    assert manifest["encoded_dataset_sha256"] == seal["encoded_dataset_sha256"]
    assert manifest["records"] == 2
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["splits"]["validation"]["records"] == 1
    labels_path = output / "train-00000.labels.i32le"
    labels = struct.unpack(f"<{labels_path.stat().st_size // 4}i", labels_path.read_bytes())
    assert labels[:5] == (-100, -100, -100, -100, -100)
    assert labels[-4:] == (116, 216, 10, 2)


def test_materializer_rejects_seal_or_record_drift(tmp_path) -> None:
    records = [_record("train-one", "train")]
    tokenizer = FakeTokenizer()
    template = load_chat_template(ROOT / "configs" / "lumi-4b-chat-template-v0.1.json")
    seal = _seal(records, tokenizer)
    changed_seal = copy.deepcopy(seal)
    changed_seal["records"] = 2
    with pytest.raises(ValueError, match="seal hash changed"):
        materialize_sft_shards(
            records,
            output=tmp_path / "bad-seal",
            seal=changed_seal,
            template=template,
            tokenizer=tokenizer,
        )

    changed_records = copy.deepcopy(records)
    changed_records[0]["messages"][1]["content"] = "Different answer"
    with pytest.raises(ValueError, match="do not match the sealed dataset"):
        materialize_sft_shards(
            changed_records,
            output=tmp_path / "bad-records",
            seal=seal,
            template=template,
            tokenizer=tokenizer,
        )


def test_verifier_rejects_tampered_binary_shard(tmp_path) -> None:
    records = [_record("train-one", "train")]
    tokenizer = FakeTokenizer()
    output = tmp_path / "encoded"
    materialize_sft_shards(
        records,
        output=output,
        seal=_seal(records, tokenizer),
        template=load_chat_template(ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"),
        tokenizer=tokenizer,
    )
    path = output / "train-00000.tokens.u32le"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="size changed"):
        verify_sft_shards(output)
