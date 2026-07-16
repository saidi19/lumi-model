from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from lumi import (
    LumiConfig,
    LumiForCausalLM,
    PackedTokenDataset,
    build_heldout_evaluation,
    build_packed_dataset,
    evaluate_checkpoint,
    packed_data,
    parameter_breakdown,
    train_packed_pilot,
    validate_heldout_manifest,
    validate_packed_manifest,
)
from lumi.cloud_pilot import load_cloud_pilot_plan, preflight_cloud_pilot


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


@pytest.fixture
def heldout_corpus(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "heldout-corpus.jsonl"
    sources = (
        ("category_a", "component_a", "alpha", 100),
        ("category_b", "component_b", "beta", 300),
    )
    with path.open("w", encoding="utf-8") as handle:
        for category, component, marker, _ in sources:
            for index in range(80):
                text = (
                    f"{marker} source document {index} has unique value "
                    f"{index * 101 + len(marker)}"
                )
                handle.write(
                    json.dumps(
                        {
                            "content_sha256": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                            "id": f"{category}/{component}/{index}",
                            "provenance": {
                                "category": category,
                                "component": component,
                            },
                            "quality": {
                                "near_duplicate": {
                                    "fingerprint_sha256": hashlib.sha256(
                                        f"near/{category}/{component}/{index}".encode()
                                    ).hexdigest()
                                }
                            },
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    source_manifest = tmp_path / "heldout-source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "output": {
                    "file_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "source_files": [
                    {
                        "category": category,
                        "component": component,
                        "statistics": {"target_text_utf8_bytes": target_bytes},
                        "used": True,
                    }
                    for category, component, _, target_bytes in sources
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


def _build_training_for_heldout(
    corpus: Path, source_manifest: Path, tokenizer: Path, output: Path
) -> dict:
    return build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tokenizer,
        output=output,
        sequence_length=8,
        sequences_per_shard=4,
        target_tokens=96,
        checkpoint_documents=5,
        expected_vocab_size=320,
        source_mix_manifest_path=source_manifest,
    )


def _build_heldout(
    corpus: Path,
    tokenizer: Path,
    training_manifest: Path,
    output: Path,
    *,
    resume: bool = False,
    interrupt_after_documents: int | None = None,
) -> dict:
    return build_heldout_evaluation(
        input_path=corpus,
        tokenizer_path=tokenizer,
        training_manifest_path=training_manifest,
        output=output,
        target_tokens=64,
        sequence_length=8,
        sequences_per_shard=2,
        seed=23,
        candidate_fraction_denominator=1,
        checkpoint_documents=5,
        expected_vocab_size=320,
        resume=resume,
        _interrupt_after_documents=interrupt_after_documents,
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


def test_heldout_builder_replays_training_and_proves_zero_overlap(
    tiny_tokenizer: Path,
    heldout_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, source_manifest = heldout_corpus
    training_output = tmp_path / "heldout-training"
    _build_training_for_heldout(
        corpus, source_manifest, tiny_tokenizer, training_output
    )
    output = tmp_path / "heldout-evaluation"
    manifest = _build_heldout(
        corpus,
        tiny_tokenizer,
        training_output / "manifest.json",
        output,
    )

    assert manifest["selection"]["mode"] == "post_training_hash_ranked"
    assert manifest["statistics"]["tokens"] == 64
    assert manifest["statistics"]["sequences"] == 8
    assert manifest["leakage"]["document_id_overlap"] == 0
    assert manifest["leakage"]["content_sha256_overlap"] == 0
    assert manifest["leakage"]["near_duplicate_sha256_overlap"] == 0
    assert [item["tokens"] for item in manifest["statistics"]["strata"]] == [
        16,
        48,
    ]

    def ledger(name: str) -> list[dict]:
        return [
            json.loads(line)
            for line in (output / name).read_text(encoding="utf-8").splitlines()
        ]

    training_rows = ledger("training-exclusion-ledger.jsonl")
    selected_rows = ledger("selection-ledger.jsonl")
    for key in ("document_id", "content_sha256", "near_duplicate_sha256"):
        assert {row[key] for row in training_rows}.isdisjoint(
            {row[key] for row in selected_rows}
        )

    dataset = PackedTokenDataset(output / "manifest.json")
    assert len(dataset) == 8
    assert validate_heldout_manifest(output / "manifest.json") == manifest
    cached = _build_heldout(
        corpus,
        tiny_tokenizer,
        training_output / "manifest.json",
        output,
        resume=True,
    )
    assert cached == manifest


def test_heldout_interruption_resumes_byte_identically(
    tiny_tokenizer: Path,
    heldout_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, source_manifest = heldout_corpus
    training_output = tmp_path / "resume-training"
    _build_training_for_heldout(
        corpus, source_manifest, tiny_tokenizer, training_output
    )
    training_manifest = training_output / "manifest.json"
    interrupted_output = tmp_path / "interrupted-heldout"
    with pytest.raises(RuntimeError, match="simulated held-out scan interruption"):
        _build_heldout(
            corpus,
            tiny_tokenizer,
            training_manifest,
            interrupted_output,
            interrupt_after_documents=17,
        )

    connection = sqlite3.connect(interrupted_output / "heldout-state.sqlite3")
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'state'"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    durable_state = json.loads(row[0])
    assert durable_state["status"] == "scanning"
    assert durable_state["documents_seen"] == 17

    resumed = _build_heldout(
        corpus,
        tiny_tokenizer,
        training_manifest,
        interrupted_output,
        resume=True,
    )
    fresh_output = tmp_path / "fresh-heldout"
    fresh = _build_heldout(
        corpus,
        tiny_tokenizer,
        training_manifest,
        fresh_output,
    )

    assert resumed["build_fingerprint"] == fresh["build_fingerprint"]
    assert resumed["statistics"] == fresh["statistics"]
    assert resumed["selection_ledger"]["sha256"] == fresh["selection_ledger"][
        "sha256"
    ]
    assert resumed["training_exclusion_ledger"]["sha256"] == fresh[
        "training_exclusion_ledger"
    ]["sha256"]
    assert [item["sha256"] for item in resumed["shards"]] == [
        item["sha256"] for item in fresh["shards"]
    ]


def test_checkpoint_evaluation_resumes_and_caches(
    tiny_tokenizer: Path,
    heldout_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, source_manifest = heldout_corpus
    training_output = tmp_path / "evaluation-training"
    _build_training_for_heldout(
        corpus, source_manifest, tiny_tokenizer, training_output
    )
    heldout_output = tmp_path / "evaluation-heldout"
    _build_heldout(
        corpus,
        tiny_tokenizer,
        training_output / "manifest.json",
        heldout_output,
    )

    checkpoint = tmp_path / "evaluation-model"
    torch.manual_seed(29)
    model = LumiForCausalLM(
        LumiConfig(
            vocab_size=320,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            doc_token_id=3,
            tie_word_embeddings=True,
            use_cache=False,
        )
    )
    model.save_pretrained(checkpoint, safe_serialization=True)

    common = {
        "checkpoint_path": checkpoint,
        "manifest_path": heldout_output / "manifest.json",
        "tokenizer_path": tiny_tokenizer,
        "prompts": ["x"],
        "generation_max_new_tokens": 1,
        "batch_size": 2,
        "checkpoint_batches": 1,
        "device": "cpu",
    }
    interrupted_output = tmp_path / "interrupted-evaluation"
    with pytest.raises(
        RuntimeError, match="simulated checkpoint evaluation interruption"
    ):
        evaluate_checkpoint(
            output=interrupted_output,
            _interrupt_after_sequences=2,
            **common,
        )
    interrupted_state = json.loads(
        (interrupted_output / "evaluation-state.json").read_text(encoding="utf-8")
    )
    assert interrupted_state["status"] == "active"
    assert interrupted_state["sequence_cursor"] == 2

    resumed = evaluate_checkpoint(output=interrupted_output, resume=True, **common)
    fresh = evaluate_checkpoint(output=tmp_path / "fresh-evaluation", **common)
    assert resumed["loss"] == fresh["loss"]
    assert resumed["generation"][0]["new_token_ids"] == fresh["generation"][0][
        "new_token_ids"
    ]
    assert resumed["runtime"]["sequences"] == 8
    assert resumed["runtime"]["model_parameters"] == parameter_breakdown(
        model.config
    ).total
    cached = evaluate_checkpoint(output=interrupted_output, resume=True, **common)
    assert cached == resumed
    report_path = interrupted_output / "evaluation-report.json"
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["loss"]["mean"] += 1.0
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="mean loss changed"):
        evaluate_checkpoint(output=interrupted_output, resume=True, **common)


def test_cloud_pilot_preflight_pins_portable_artifacts(
    tiny_tokenizer: Path,
    stratified_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    corpus, source_manifest = stratified_corpus
    tokenizer = tmp_path / "tokenizer"
    shutil.copytree(tiny_tokenizer, tokenizer)
    packed = tmp_path / "packed"
    manifest = build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tokenizer,
        output=packed,
        sequence_length=8,
        sequences_per_shard=4,
        target_tokens=96,
        checkpoint_documents=1,
        expected_vocab_size=320,
        source_mix_manifest_path=source_manifest,
    )
    manifest_path = packed / "manifest.json"

    plan_directory = tmp_path / "configs"
    plan_directory.mkdir()
    config = LumiConfig(
        vocab_size=320,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        doc_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config_path = plan_directory / "model.json"
    config.to_json_file(config_path)
    plan_path = plan_directory / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "artifact": {
                    "build_fingerprint": manifest["build_fingerprint"],
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "sequence_length": 8,
                    "sequences": 12,
                    "source_strata": 2,
                    "tokens": 96,
                },
                "full": {
                    "batch_size": 2,
                    "checkpoint_steps": 1,
                    "gradient_accumulation_steps": 2,
                    "processed_tokens": 96,
                    "repeated_tokens": 0,
                    "steps": 3,
                    "tokens_per_step": 32,
                },
                "model": {
                    "config": config_path.name,
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "parameters": parameter_breakdown(config).total,
                },
                "optimizer": {
                    "learning_rate": 3e-4,
                    "max_grad_norm": 1.0,
                    "seed": 7,
                    "weight_decay": 0.1,
                },
                "requires_bf16": True,
                "schema_version": 1,
                "smoke": {
                    "batch_size": 1,
                    "checkpoint_steps": 1,
                    "first_steps": 1,
                    "gradient_accumulation_steps": 1,
                    "resume_steps": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    report = preflight_cloud_pilot(
        plan_path=plan_path,
        manifest_path=manifest_path,
        tokenizer_path=tokenizer,
    )
    assert report["status"] == "pass"
    assert report["artifact"]["source_strata"] == 2
    assert report["parameters"] == parameter_breakdown(config).total
    assert isinstance(report["runtime"]["bf16_supported"], bool)

    with (tokenizer / "tokenizer_config.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="tokenizer file size changed"):
        preflight_cloud_pilot(
            plan_path=plan_path,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer,
        )


def _build_scaling_cloud_fixture(
    *,
    tiny_tokenizer: Path,
    stratified_corpus: tuple[Path, Path],
    tmp_path: Path,
) -> dict[str, Path]:
    corpus, source_manifest = stratified_corpus
    tokenizer = tmp_path / "scaling-tokenizer"
    shutil.copytree(tiny_tokenizer, tokenizer)
    packed = tmp_path / "scaling-packed"
    manifest = build_packed_dataset(
        input_path=corpus,
        tokenizer_path=tokenizer,
        output=packed,
        sequence_length=8,
        sequences_per_shard=4,
        target_tokens=96,
        checkpoint_documents=1,
        expected_vocab_size=320,
        source_mix_manifest_path=source_manifest,
    )
    for shard in manifest["shards"]:
        path = packed / shard["name"]
        values = torch.from_file(
            str(path),
            shared=False,
            size=shard["tokens"],
            dtype=torch.int32,
        )
        path.write_bytes(
            b"".join(
                int(value).to_bytes(2, "little") for value in values.tolist()
            )
        )
        shard["bytes"] = path.stat().st_size
        shard["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    proof_path = packed / "selection-proof.json"
    proof_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    state_path = packed / "scaling-packing-state.sqlite3"
    state_path.write_bytes(b"sealed synthetic scaling state")
    manifest["format"] = {
        "dtype": "uint16",
        "endianness": "little",
        "layout": "flat_fixed_sequences",
        "token_bytes": 2,
    }
    manifest["leakage"] = {
        "content_sha256_overlap": 0,
        "document_id_overlap": 0,
        "exact_documents_excluded": 2,
        "family_documents_excluded": 2,
        "near_duplicate_sha256_overlap": 0,
    }
    manifest["packing"].update(
        {
            "purpose": "final",
            "target_tokens_effective": 96,
            "target_tokens_requested": 96,
        }
    )
    manifest["selection"].update(
        {
            "mode": "within_source_hash_ranked_excluding_heldout",
            "seed": 23,
            "source_index": {
                "build_fingerprint": "2" * 64,
                "bytes": 123,
                "path": "/sealed/source-index/manifest.json",
                "sha256": "1" * 64,
            },
        }
    )
    manifest["selection_proof"] = {
        "bytes": proof_path.stat().st_size,
        "name": proof_path.name,
        "sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    }
    manifest["selection_state"] = {
        "bytes": state_path.stat().st_size,
        "name": state_path.name,
        "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }
    manifest["statistics"]["documents_selected"] = manifest["statistics"][
        "documents_packed"
    ]
    manifest_path = packed / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plan_directory = tmp_path / "scaling-configs"
    plan_directory.mkdir()
    pilot_config = LumiConfig(
        vocab_size=320,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        doc_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
    reference_config = LumiConfig(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        doc_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
    pilot_config_path = plan_directory / "pilot.json"
    reference_config_path = plan_directory / "reference.json"
    pilot_config.to_json_file(pilot_config_path)
    reference_config.to_json_file(reference_config_path)
    digest = "0" * 64
    scaling_plan = {
        "approval": {"full_4b_training": False, "paid_gpu_launch": False},
        "data": {
            "heldout": {
                "build_fingerprint": digest,
                "documents": 2,
                "manifest_sha256": digest,
                "selection_ledger_sha256": digest,
                "tokens": 8,
            },
            "selection": {
                "mode": "within_source_hash_ranked_excluding_heldout",
                "seed": 23,
                "source_strata": 2,
                "target_tokens_effective": 96,
                "target_tokens_requested": 96,
            },
            "sequence_length": 8,
            "source_sample": {
                "accepted_text_bytes": 1000,
                "documents": manifest["statistics"]["documents_seen"],
                "manifest_sha256": digest,
                "observed_tokens": 200,
                "sample_sha256": digest,
                "token_count_report_sha256": digest,
            },
        },
        "evaluation": {
            "baseline_150m": {
                "heldout_loss": 5.0,
                "heldout_perplexity": 148.0,
                "parameters": 100,
                "report_sha256": digest,
                "training_tokens": 64,
            },
            "require_loss_below_baseline": True,
            "require_zero_overlap": True,
        },
        "model": {
            "config": pilot_config_path.name,
            "config_sha256": hashlib.sha256(
                pilot_config_path.read_bytes()
            ).hexdigest(),
            "parameters": parameter_breakdown(pilot_config).total,
            "reference_config": reference_config_path.name,
            "reference_config_sha256": hashlib.sha256(
                reference_config_path.read_bytes()
            ).hexdigest(),
            "reference_parameters": parameter_breakdown(reference_config).total,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "lr_schedule": "cosine",
            "max_grad_norm": 1.0,
            "min_learning_rate_ratio": 0.1,
            "schedule_steps": 12,
            "seed": 7,
            "warmup_steps": 1,
            "weight_decay": 0.0,
        },
        "schema_version": 1,
        "stages": [
            {
                "name": name,
                "processed_tokens": target,
                "repeated_tokens": 0,
                "steps": target // 8,
                "target_tokens": target,
            }
            for name, target in (
                ("smoke_resume", 8),
                ("calibration", 16),
                ("learning_gate", 32),
                ("unique_corpus", 96),
            )
        ],
        "training": {
            "batch_size": 1,
            "checkpoint_steps": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "keep_checkpoints": 2,
            "precision": "bf16_autocast_fp32_parameters",
            "tokens_per_step": 8,
        },
    }
    scaling_plan_path = plan_directory / "scaling-plan.json"
    scaling_plan_path.write_text(json.dumps(scaling_plan), encoding="utf-8")
    data_contract = {
        "final": {
            "expected_shards": len(manifest["shards"]),
            "sequences_per_shard": 4,
            "target_tokens_effective": 96,
            "target_tokens_requested": 96,
        },
        "packing": {"sequence_length": 8},
        "schema_version": 1,
    }
    data_contract_path = plan_directory / "scaling-data.json"
    data_contract_path.write_text(json.dumps(data_contract), encoding="utf-8")
    shard_bytes = sum(item["bytes"] for item in manifest["shards"])
    parameters = parameter_breakdown(pilot_config).total
    cloud_plan = {
        "approval": {
            "full_1b_stage": False,
            "full_4b_training": False,
            "paid_gpu_launch": False,
        },
        "artifact": {
            "build_fingerprint": manifest["build_fingerprint"],
            "documents_seen": manifest["statistics"]["documents_seen"],
            "documents_selected": manifest["statistics"]["documents_selected"],
            "dtype": "uint16",
            "endianness": "little",
            "heldout_exact_documents": 2,
            "heldout_family_documents": 2,
            "index_build_fingerprint": "2" * 64,
            "index_manifest_sha256": "1" * 64,
            "layout": "flat_fixed_sequences",
            "leakage": {
                "content_sha256_overlap": 0,
                "document_id_overlap": 0,
                "near_duplicate_sha256_overlap": 0,
            },
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "purpose": "final",
            "selection_mode": "within_source_hash_ranked_excluding_heldout",
            "selection_proof_sha256": manifest["selection_proof"]["sha256"],
            "selection_seed": 23,
            "selection_state_sha256": manifest["selection_state"]["sha256"],
            "sequence_length": 8,
            "sequences": 12,
            "sequences_per_shard": 4,
            "shard_bytes": shard_bytes,
            "shards": len(manifest["shards"]),
            "source_strata": 2,
            "target_tokens_requested": 96,
            "token_bytes": 2,
            "tokens": 96,
        },
        "contracts": {
            "scaling_data": {
                "path": data_contract_path.name,
                "sha256": hashlib.sha256(data_contract_path.read_bytes()).hexdigest(),
            },
            "scaling_pilot": {
                "path": scaling_plan_path.name,
                "sha256": hashlib.sha256(scaling_plan_path.read_bytes()).hexdigest(),
            },
        },
        "model": {
            "config": pilot_config_path.name,
            "config_sha256": hashlib.sha256(
                pilot_config_path.read_bytes()
            ).hexdigest(),
            "parameters": parameters,
        },
        "pilot_kind": "scaling_1b",
        "runtime": {
            "minimum_compute_capability": [0, 0],
            "minimum_cuda_devices": 1,
            "minimum_cuda_memory_bytes": 1,
            "minimum_host_memory_bytes": 1,
            "requires_bf16": True,
        },
        "schema_version": 2,
        "storage": {
            "checkpoint_bytes_per_parameter": 1,
            "fixed_overhead_bytes": 0,
            "minimum_free_disk_bytes": parameters + shard_bytes,
            "peak_checkpoint_copies": 1,
        },
    }
    cloud_plan_path = plan_directory / "cloud-plan.json"
    cloud_plan_path.write_text(json.dumps(cloud_plan), encoding="utf-8")
    return {
        "cloud_plan": cloud_plan_path,
        "data_contract": data_contract_path,
        "manifest": manifest_path,
        "proof": proof_path,
        "tokenizer": tokenizer,
        "workspace": tmp_path,
    }


def test_scaling_cloud_preflight_pins_proof_contracts_and_runtime(
    tiny_tokenizer: Path,
    stratified_corpus: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_scaling_cloud_fixture(
        tiny_tokenizer=tiny_tokenizer,
        stratified_corpus=stratified_corpus,
        tmp_path=tmp_path,
    )
    report = preflight_cloud_pilot(
        plan_path=fixture["cloud_plan"],
        manifest_path=fixture["manifest"],
        tokenizer_path=fixture["tokenizer"],
        workspace_path=fixture["workspace"],
    )
    assert report["status"] == "pass"
    assert report["kind"] == "scaling_1b"
    assert report["artifact"]["dtype"] == "uint16"
    assert report["integrity"] == {
        "auxiliary_hashes_verified": True,
        "shard_hashes_verified": True,
        "tokenizer_hashes_verified": True,
    }
    assert report["runtime"]["gates"]["minimum_free_disk"] is True
    assert report["authorization"]["paid_gpu_launch"] is False
    with pytest.raises(ValueError, match="cannot skip shard hashes"):
        preflight_cloud_pilot(
            plan_path=fixture["cloud_plan"],
            manifest_path=fixture["manifest"],
            tokenizer_path=fixture["tokenizer"],
            workspace_path=fixture["workspace"],
            require_cuda=True,
            verify_shards=False,
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        preflight_cloud_pilot(
            plan_path=fixture["cloud_plan"],
            manifest_path=fixture["manifest"],
            tokenizer_path=fixture["tokenizer"],
            workspace_path=fixture["workspace"],
            require_cuda=True,
        )

    original_proof = fixture["proof"].read_bytes()
    fixture["proof"].write_bytes(original_proof + b"tamper")
    with pytest.raises(ValueError, match="selection proof size changed"):
        preflight_cloud_pilot(
            plan_path=fixture["cloud_plan"],
            manifest_path=fixture["manifest"],
            tokenizer_path=fixture["tokenizer"],
            workspace_path=fixture["workspace"],
        )
    fixture["proof"].write_bytes(original_proof)

    with fixture["data_contract"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="contracts.scaling_data SHA-256 changed"):
        load_cloud_pilot_plan(fixture["cloud_plan"])


def test_canonical_scaling_cloud_plan_loads() -> None:
    repository_root = Path(__file__).parents[1]
    plan = load_cloud_pilot_plan(
        repository_root / "configs" / "cloud-pilot-1b-v0.1.json"
    )
    assert plan["schema_version"] == 2
    assert plan["artifact"]["tokens"] == 1_399_996_416
    assert plan["artifact"]["shards"] == 42
    assert plan["approval"]["paid_gpu_launch"] is False


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
        "gradient_checkpointing": True,
        "learning_rate": 1e-3,
        "lr_schedule": "cosine",
        "min_learning_rate_ratio": 0.2,
        "schedule_steps": 4,
        "seed": 19,
        "warmup_steps": 1,
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
