from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_format import load_chat_template
from lumi.posttraining_seal import build_sft_dataset_seal
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
        return [100 + len(text)]


def _record() -> dict:
    return {
        "language": "en",
        "messages": [
            {"content": "Question", "role": "user"},
            {"content": "Answer", "role": "assistant"},
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
        "record_id": "one",
        "safety": {"policy_version": "lumi-safety-v0.1", "risk_labels": []},
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": "OpenAssistant/oasst1",
            "document_id": "doc-one",
            "license_id": "apache-2.0",
            "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
            "source_id": "openassistant_oasst1",
            "uri": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        },
        "split": "train",
        "task_category": "conversation",
    }


def _registry(*, production_eligible: bool) -> SFTSourceRegistry:
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
            "production_blockers": () if production_eligible else source.production_blockers,
            "production_eligible": production_eligible,
        }
    )
    return SFTSourceRegistry(
        observed_at=loaded.observed_at,
        registry_sha256=loaded.registry_sha256,
        sources=(promoted,),
    )


def _ai_candidate_record() -> dict:
    record = _record()
    record["provenance"] = {
        **record["provenance"],
        "consent_basis": "first_party_rights_declaration",
        "generation_method": "ai_generated",
        "human_review_status": "owner_spot_check_pending",
    }
    record["source"] = {
        "dataset_id": "Afnan-Technologies-Inc/lumi-ai-assisted-sft-v0.2",
        "document_id": "candidate-one",
        "license_id": "lumi-owned",
        "revision": "d492d2fa0817d08b55941b0393cb3d95ae5572a0bcc3c6de39f902bd8491a9b0",
        "source_id": "lumi_ai_assisted_sft_v0_2",
        "uri": "private://lumi-ai-assisted/candidate-one",
    }
    return record


def test_seal_refuses_canary_only_source() -> None:
    policy = load_sft_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="source promotion blockers remain"):
        build_sft_dataset_seal(
            [_record()],
            base_binding=BASE_BINDING,
            benchmark_registry=NeverMatch(),
            policy=policy,
            policy_sha256="2" * 64,
            source_registry=_registry(production_eligible=False),
            template=load_chat_template(
                ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"
            ),
            tokenizer=FakeTokenizer(),
        )


def test_seal_refuses_ai_candidate_before_owner_and_full_mix_gates() -> None:
    policy = load_sft_policy(POLICY_PATH)
    registry = load_sft_source_registry(
        ROOT / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    with pytest.raises(ValueError, match="dataset_level_owner_spot_check_not_complete"):
        build_sft_dataset_seal(
            [_ai_candidate_record()],
            base_binding=BASE_BINDING,
            benchmark_registry=NeverMatch(),
            policy=policy,
            policy_sha256="2" * 64,
            source_registry=registry,
            template=load_chat_template(
                ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"
            ),
            tokenizer=FakeTokenizer(),
        )


def test_seal_is_deterministic_and_raw_text_free_after_promotion() -> None:
    policy = load_sft_policy(POLICY_PATH)
    arguments = {
        "base_binding": BASE_BINDING,
        "benchmark_registry": NeverMatch(),
        "policy": policy,
        "policy_sha256": "2" * 64,
        "source_registry": _registry(production_eligible=True),
        "template": load_chat_template(
            ROOT / "configs" / "lumi-4b-chat-template-v0.1.json"
        ),
        "tokenizer": FakeTokenizer(),
    }
    first = build_sft_dataset_seal([_record()], **arguments)
    second = build_sft_dataset_seal([copy.deepcopy(_record())], **arguments)
    assert first == second
    assert first["records"] == 1
    assert first["input_tokens"] > first["supervised_tokens"] > 0
    serialized = json.dumps(first)
    assert "Question" not in serialized
    assert "Answer" not in serialized
    assert first["raw_text_retained_in_manifest"] is False
    assert len(first["seal_sha256"]) == 64
