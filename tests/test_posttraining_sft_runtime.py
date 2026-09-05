from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from lumi.configuration_lumi import LumiConfig
from lumi.modeling_lumi import LumiForCausalLM
from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_format import load_chat_template
from lumi.posttraining_seal import build_sft_dataset_seal
from lumi.posttraining_sft_runtime import (
    SFT_CANARY_METRICS_FILENAME,
    EncodedSFTDataset,
    collate_sft_tensors,
    run_sft_cpu_canary,
    verify_sft_canary_checkpoint,
)
from lumi.posttraining_shards import materialize_sft_shards
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


def _materialize_tiny_dataset(root: Path) -> Path:
    records = [
        _record("train-one", "train"),
        _record("train-two", "train"),
        _record("train-three", "train"),
        _record("valid-one", "validation"),
    ]
    tokenizer = FakeTokenizer()
    template = load_chat_template(ROOT / "configs" / "lumi-4b-chat-template-v0.1.json")
    policy = load_sft_policy(POLICY_PATH)
    seal = build_sft_dataset_seal(
        records,
        base_binding=BASE_BINDING,
        benchmark_registry=NeverMatch(),
        policy=policy,
        policy_sha256="2" * 64,
        source_registry=_registry(),
        template=template,
        tokenizer=tokenizer,
    )
    output = root / "encoded"
    materialize_sft_shards(
        records,
        output=output,
        seal=seal,
        template=template,
        tokenizer=tokenizer,
        maximum_tokens_per_shard=4096,
    )
    return output


def _write_manifest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((digest, path.relative_to(root).as_posix()))
    manifest = root / "manifest.sha256"
    manifest.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _tiny_source_checkpoint(root: Path) -> tuple[Path, str]:
    torch.manual_seed(11)
    model = LumiForCausalLM(
        LumiConfig(
            vocab_size=512,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        )
    )
    checkpoint = root / "source"
    model.save_pretrained(checkpoint, safe_serialization=True)
    return checkpoint, _write_manifest(checkpoint)


def _model_state(path: Path) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in LumiForCausalLM.from_pretrained(
            path,
            local_files_only=True,
        ).state_dict().items()
    }


def test_sft_dataset_masks_prompts_and_pads_without_loss(tmp_path) -> None:
    dataset = EncodedSFTDataset(_materialize_tiny_dataset(tmp_path), split="train")
    example = dataset[0]
    assert bool(example.labels.eq(-100).any())
    assert torch.equal(
        example.labels[example.labels.ne(-100)],
        example.input_ids[example.labels.ne(-100)],
    )
    batch = collate_sft_tensors([dataset[0], dataset[1]], context_window=32)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.all(batch["labels"][batch["attention_mask"].eq(0)].eq(-100))


def test_two_step_sft_cpu_canary_is_exact_across_fresh_resume(tmp_path) -> None:
    encoded = _materialize_tiny_dataset(tmp_path)
    source, source_manifest = _tiny_source_checkpoint(tmp_path)
    common = {
        "encoded_shards": encoded,
        "source_checkpoint": source,
        "source_checkpoint_manifest_sha256": source_manifest,
        "target_steps": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "seed": 19,
    }

    uninterrupted = tmp_path / "uninterrupted"
    uninterrupted_state = run_sft_cpu_canary(
        **common,
        output=uninterrupted,
        stop_after_step=2,
    )
    resumed = tmp_path / "resumed"
    paused_state = run_sft_cpu_canary(
        **common,
        output=resumed,
        stop_after_step=1,
    )
    assert paused_state["status"] == "paused"
    resumed_state = run_sft_cpu_canary(
        **common,
        output=resumed,
        stop_after_step=2,
        resume=True,
    )

    assert uninterrupted_state["status"] == resumed_state["status"] == "complete"
    assert uninterrupted_state["records_seen"] == resumed_state["records_seen"] == 4
    assert uninterrupted_state["supervised_tokens_seen"] > 0
    assert uninterrupted_state["last_loss"] == pytest.approx(
        resumed_state["last_loss"], rel=0.0, abs=0.0
    )
    first_checkpoint = uninterrupted / uninterrupted_state["latest_checkpoint"]
    second_checkpoint = resumed / resumed_state["latest_checkpoint"]
    assert verify_sft_canary_checkpoint(first_checkpoint)["step"] == 2
    assert verify_sft_canary_checkpoint(second_checkpoint)["step"] == 2
    first_state = _model_state(first_checkpoint)
    second_state = _model_state(second_checkpoint)
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0.0, atol=0.0)
    uninterrupted_metrics = [
        json.loads(line)
        for line in (uninterrupted / SFT_CANARY_METRICS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    resumed_metrics = [
        json.loads(line)
        for line in (resumed / SFT_CANARY_METRICS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["loss"] for item in uninterrupted_metrics] == [
        item["loss"] for item in resumed_metrics
    ]
