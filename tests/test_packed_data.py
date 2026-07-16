from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from lumi import (
    LumiForCausalLM,
    PackedTokenDataset,
    build_packed_dataset,
    packed_data,
    train_packed_pilot,
    validate_packed_manifest,
)


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("packed-tokenizer")
    repository_root = Path(__file__).parents[1]
    contract = json.loads(
        (repository_root / "configs" / "tokenizer-v0.1.json").read_text(encoding="utf-8")
    )
    contract["vocab_size"] = 320
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    corpus = root / "tokenizer-corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index in range(600):
            handle.write(
                json.dumps(
                    {
                        "text": (
                            f"Lumi packing document {index}: alpha{index % 37} "
                            f"beta{index % 53} code_{index % 71} বাংলা café 日本語"
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    output = root / "tokenizer"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "train_tokenizer.py"),
            "--input",
            str(corpus),
            "--output",
            str(output),
            "--contract",
            str(contract_path),
            "--min-frequency",
            "1",
            "--no-progress",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    return output


@pytest.fixture
def packing_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(40):
            record = {"text": f"pilot record {index} block {index % 7}"}
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def stratified_corpus(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "stratified-corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for category, component, marker in (
            ("category_a", "component_a", "alpha"),
            ("category_b", "component_b", "beta"),
        ):
            for index in range(20):
                handle.write(
                    json.dumps(
                        {
                            "provenance": {
                                "category": category,
                                "component": component,
                            },
                            "text": f"{marker} {marker} source record {index}",
                        }
                    )
                    + "\n"
                )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "output": {
                    "file_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "source_files": [
                    {
                        "category": "category_a",
                        "component": "component_a",
                        "statistics": {"target_text_utf8_bytes": 100},
                        "used": True,
                    },
                    {
                        "category": "category_b",
                        "component": "component_b",
                        "statistics": {"target_text_utf8_bytes": 300},
                        "used": True,
                    },
                ],
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    return path, source_manifest


def _build(corpus: Path, tokenizer: Path, output: Path, *, resume: bool = False) -> dict:
    return build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tokenizer,
        output=output,
        sequence_length=8,
        sequences_per_shard=2,
        target_tokens=96,
        checkpoint_documents=1,
        expected_vocab_size=320,
        resume=resume,
    )


def test_pack_manifest_dataset_and_cached_resume(
    tiny_tokenizer: Path, packing_corpus: Path, tmp_path: Path
) -> None:
    output = tmp_path / "packed"
    manifest = _build(packing_corpus, tiny_tokenizer, output)

    assert manifest["status"] == "complete"
    assert manifest["end_reason"] == "target_tokens"
    assert manifest["statistics"]["tokens"] == 96
    assert manifest["statistics"]["sequences"] == 12
    assert len(manifest["shards"]) == 6
    assert all(shard["tokens"] == 16 for shard in manifest["shards"])

    dataset = PackedTokenDataset(output / "manifest.json")
    assert len(dataset) == 12
    assert dataset[0].shape == (8,)
    assert dataset[-1].dtype == torch.long
    flattened = torch.stack([dataset[index] for index in range(len(dataset))]).flatten()
    assert int((flattened == 3).sum()) >= 1
    assert int(flattened.min()) >= 0
    assert int(flattened.max()) < 320

    cached = _build(packing_corpus, tiny_tokenizer, output, resume=True)
    assert cached == manifest

    first_shard = output / manifest["shards"][0]["name"]
    with first_shard.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff\xff\xff\xff")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        validate_packed_manifest(output / "manifest.json")


def test_interrupted_build_resumes_byte_identically(
    tiny_tokenizer: Path,
    packing_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer, local_files_only=True)

    class InterruptingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("simulated interruption")
            return real_tokenizer.backend_tokenizer.encode(*args, **kwargs)

    class InterruptingTokenizer:
        backend_tokenizer = InterruptingBackend()

        def __len__(self) -> int:
            return len(real_tokenizer)

        def get_vocab(self) -> dict[str, int]:
            return real_tokenizer.get_vocab()

    interrupted_output = tmp_path / "interrupted"
    monkeypatch.setattr(
        packed_data.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: InterruptingTokenizer(),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(packing_corpus, tiny_tokenizer, interrupted_output)

    state = json.loads(
        (interrupted_output / "packing-state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "active"
    assert state["documents_seen"] == 2

    monkeypatch.setattr(
        packed_data.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: real_tokenizer,
    )
    resumed = _build(packing_corpus, tiny_tokenizer, interrupted_output, resume=True)
    fresh_output = tmp_path / "fresh"
    fresh = _build(packing_corpus, tiny_tokenizer, fresh_output)

    assert resumed["build_fingerprint"] == fresh["build_fingerprint"]
    assert resumed["statistics"] == fresh["statistics"]
    assert [item["sha256"] for item in resumed["shards"]] == [
        item["sha256"] for item in fresh["shards"]
    ]


def test_source_manifest_stratification_scans_clustered_sources(
    tiny_tokenizer: Path,
    stratified_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, source_manifest = stratified_corpus
    output = tmp_path / "stratified-packed"
    manifest = build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tiny_tokenizer,
        output=output,
        sequence_length=8,
        sequences_per_shard=4,
        target_tokens=96,
        checkpoint_documents=1,
        expected_vocab_size=320,
        source_mix_manifest_path=source_manifest,
    )

    assert manifest["end_reason"] == "source_manifest_stratified_target"
    assert manifest["statistics"]["documents_seen"] > 20
    assert [item["tokens"] for item in manifest["statistics"]["strata"]] == [24, 72]
    assert [item["target_tokens"] for item in manifest["selection"]["strata"]] == [
        24,
        72,
    ]
    dataset = PackedTokenDataset(output / "manifest.json")
    flattened = torch.stack([dataset[index] for index in range(len(dataset))]).flatten()
    tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer, local_files_only=True)
    assert "alpha" in tokenizer.decode(flattened[:24], skip_special_tokens=True)
    assert "beta" in tokenizer.decode(flattened[24:], skip_special_tokens=True)

    cached = build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tiny_tokenizer,
        output=output,
        sequence_length=8,
        sequences_per_shard=4,
        target_tokens=96,
        checkpoint_documents=1,
        expected_vocab_size=320,
        source_mix_manifest_path=source_manifest,
        resume=True,
    )
    assert cached == manifest

    changed_manifest = tmp_path / "changed-source-manifest.json"
    changed = json.loads(source_manifest.read_text(encoding="utf-8"))
    changed["output"]["sha256"] = "0" * 64
    changed_manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        build_packed_dataset(
            input_path=corpus,
            tokenizer_path=tiny_tokenizer,
            output=tmp_path / "rejected-stratified-packed",
            sequence_length=8,
            target_tokens=96,
            expected_vocab_size=320,
            source_mix_manifest_path=changed_manifest,
        )


def test_training_resume_matches_uninterrupted_run(
    tiny_tokenizer: Path, packing_corpus: Path, tmp_path: Path
) -> None:
    packed_output = tmp_path / "packed-training"
    _build(packing_corpus, tiny_tokenizer, packed_output)
    manifest_path = packed_output / "manifest.json"
    config_path = tmp_path / "tiny-training-config.json"
    config_path.write_text(
        json.dumps(
            {
                "attention_bias": False,
                "bos_token_id": 1,
                "doc_token_id": 3,
                "eos_token_id": 2,
                "head_dim": 8,
                "hidden_size": 32,
                "intermediate_size": 64,
                "max_position_embeddings": 8,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "pad_token_id": 0,
                "tie_word_embeddings": True,
                "torch_dtype": "float32",
                "use_cache": False,
                "vocab_size": 320,
            }
        ),
        encoding="utf-8",
    )
    common = {
        "manifest_path": manifest_path,
        "config_path": config_path,
        "batch_size": 2,
        "checkpoint_steps": 1,
        "device": "cpu",
        "learning_rate": 1e-3,
        "seed": 19,
        "weight_decay": 0.0,
    }

    resumed_output = tmp_path / "resumed-training"
    first = train_packed_pilot(output=resumed_output, steps=2, **common)
    assert first["status"] == "complete"
    assert first["step"] == 2
    stale_state = json.loads(
        (resumed_output / "checkpoint-00000001" / "training-state.json").read_text(
            encoding="utf-8"
        )
    )
    (resumed_output / "training-state.json").write_text(
        json.dumps(stale_state), encoding="utf-8"
    )
    with (resumed_output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"loss": -1.0, "step": 99}) + "\n")

    relocated_packed = tmp_path / "relocated-packed-training"
    shutil.copytree(packed_output, relocated_packed)
    relocated_config = tmp_path / "relocated-training-config.json"
    relocated_config.write_bytes(config_path.read_bytes())
    relocated_common = {
        **common,
        "config_path": relocated_config,
        "manifest_path": relocated_packed / "manifest.json",
    }
    resumed = train_packed_pilot(
        output=resumed_output, steps=4, resume=True, **relocated_common
    )

    fresh_output = tmp_path / "fresh-training"
    fresh = train_packed_pilot(output=fresh_output, steps=4, **common)
    assert resumed["step"] == fresh["step"] == 4
    assert resumed["tokens_seen"] == fresh["tokens_seen"] == 64

    resumed_model = LumiForCausalLM.from_pretrained(
        resumed_output / resumed["latest_checkpoint"], local_files_only=True
    )
    fresh_model = LumiForCausalLM.from_pretrained(
        fresh_output / fresh["latest_checkpoint"], local_files_only=True
    )
    for name, resumed_tensor in resumed_model.state_dict().items():
        assert torch.equal(resumed_tensor, fresh_model.state_dict()[name]), name

    def losses(path: Path) -> list[float]:
        return [
            json.loads(line)["loss"]
            for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    assert losses(resumed_output) == losses(fresh_output)
