from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from lumi.distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    _write_checksum_manifest,
)
from lumi.production_training import (
    FINAL_CORPUS_SHARD_MANIFEST_GATE,
    FinalCorpusTokenDataset,
    build_account_segment_plan,
    build_production_training_preflight_plan,
    seal_checkpoint_handoff,
    stage_final_corpus_shards,
    staged_shard_name,
    validate_final_corpus_training_manifest,
    verify_checkpoint_handoff,
)


def _bytes(values: list[int]) -> bytes:
    result = array("H", values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def _free_ipv4_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _final_fixture(tmp_path: Path, *, sequence_count: int = 5) -> tuple[Path, Path, list[int]]:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    staged.mkdir()
    values = [
        token
        for sequence in range(sequence_count)
        for token in (3, sequence * 10 + 11, sequence * 10 + 12, sequence * 10 + 13)
    ]
    partitions = [values[:5], values[5:]]
    shards = []
    for sequence, tokens in enumerate(partitions):
        payload = _bytes(tokens)
        digest = hashlib.sha256(payload).hexdigest()
        source_path = source / f"source-{sequence}.bin"
        source_path.write_bytes(payload)
        document_markers = tokens.count(3)
        row = {
            "category": f"category-{sequence}",
            "content_tokens": len(tokens) - document_markers,
            "document_markers": document_markers,
            "documents": document_markers,
            "object_bytes": len(payload),
            "object_sha256": digest,
            "s3_uri": f"s3://fixture-bucket/final/{source_path.name}",
            "sequence": sequence,
            "tokens": len(tokens),
        }
        shards.append(row)
        (staged / staged_shard_name(row)).write_bytes(payload)
    manifest = {
        "content_tokens": len(values) - sequence_count,
        "document_markers": sequence_count,
        "documents": sequence_count,
        "gate": FINAL_CORPUS_SHARD_MANIFEST_GATE,
        "packing": {
            "doc_token": "<|doc|>",
            "doc_token_id": 3,
            "markers_per_document": 1,
            "token_accounting": "content_plus_one_document_marker",
        },
        "plan_fingerprint": "a" * 64,
        "schema_version": 1,
        "selection_result_sha256": "b" * 64,
        "shards": shards,
        "status": "sealed",
        "tokens": len(values),
        "training_token_limit": len(values),
    }
    manifest_path = tmp_path / "shard-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, staged, values


def test_final_corpus_dataset_crosses_physical_shard_boundaries(tmp_path: Path) -> None:
    manifest_path, staged, values = _final_fixture(tmp_path)
    adapter = validate_final_corpus_training_manifest(
        manifest_path,
        shard_root=staged,
        sequence_length=4,
        vocab_size=128,
        doc_token_id=3,
        verify_hashes=True,
    )
    dataset = FinalCorpusTokenDataset(
        manifest_path,
        staged,
        sequence_length=4,
        vocab_size=128,
        doc_token_id=3,
    )

    assert adapter["statistics"] == {
        "available_tokens": 20,
        "content_tokens": 15,
        "document_markers": 5,
        "documents": 5,
        "object_bytes": 40,
        "sequences": 5,
        "tokens": 20,
        "trailing_tokens_not_consumed": 0,
    }
    assert len(dataset) == 5
    assert dataset[1].tolist() == values[4:8]
    assert dataset[-1].tolist() == values[-4:]
    with pytest.raises(IndexError):
        _ = dataset[5]


def test_final_corpus_staging_resumes_without_redownload(tmp_path: Path) -> None:
    manifest_path, staged, _ = _final_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_by_uri = {
        row["s3_uri"]: tmp_path / "source" / f"source-{row['sequence']}.bin"
        for row in manifest["shards"]
    }
    output = tmp_path / "downloaded"
    calls: list[str] = []

    def fetch(uri: str, destination: Path) -> None:
        calls.append(uri)
        destination.write_bytes(source_by_uri[uri].read_bytes())

    first = stage_final_corpus_shards(
        manifest_path,
        output,
        fetch=fetch,
        sequence_length=4,
        vocab_size=128,
        doc_token_id=3,
    )
    assert len(calls) == 2
    second = stage_final_corpus_shards(
        manifest_path,
        output,
        fetch=fetch,
        sequence_length=4,
        vocab_size=128,
        doc_token_id=3,
    )
    assert len(calls) == 2
    assert first["build_fingerprint"] == second["build_fingerprint"]
    assert json.loads((output / "staging-state.json").read_text())["status"] == "complete"
    assert staged.is_dir()


def test_final_corpus_validation_detects_tampering(tmp_path: Path) -> None:
    manifest_path, staged, _ = _final_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = staged / staged_shard_name(manifest["shards"][0])
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size changed"):
        validate_final_corpus_training_manifest(
            manifest_path,
            shard_root=staged,
            sequence_length=4,
            vocab_size=128,
            doc_token_id=3,
            verify_hashes=True,
        )


def test_checkpoint_handoff_is_hash_bound_and_portable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-00000007"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model" / "rank-0.bin").write_bytes(b"model")
    (checkpoint / "trainer-rank-00000.pt").write_bytes(b"optimizer")
    (checkpoint / "trainer-rank-00001.pt").write_bytes(b"optimizer")
    state = {
        "checkpoint_schema_version": DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "run_fingerprint": "c" * 64,
        "run_specification": {
            "distributed": {"world_size": 2},
            "source_commit": "d" * 40,
        },
        "sequence_cursor": 17,
        "step": 7,
        "tokens_seen": 68,
    }
    (checkpoint / DISTRIBUTED_TRAINING_STATE_FILENAME).write_text(
        json.dumps(state), encoding="utf-8"
    )
    _write_checksum_manifest(checkpoint)
    handoff_path = tmp_path / "handoff.json"
    handoff = seal_checkpoint_handoff(
        checkpoint,
        handoff_path,
        source_commit="d" * 40,
        run_fingerprint="c" * 64,
        world_size=2,
    )
    assert verify_checkpoint_handoff(
        checkpoint,
        handoff_path,
        expected_run_fingerprint="c" * 64,
        expected_source_commit="d" * 40,
        expected_world_size=2,
    ) == handoff
    (checkpoint / "model" / "rank-0.bin").write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_checkpoint_handoff(checkpoint, handoff_path)


def test_final_corpus_exact_tail_runs_once_without_epoch_wrap(tmp_path: Path) -> None:
    manifest_path, staged, _ = _final_fixture(tmp_path, sequence_count=7)
    config = {
        "attention_bias": False,
        "bos_token_id": 1,
        "doc_token_id": 3,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_size": 16,
        "initialization_strategy": "residual_scaled_v0.1",
        "initializer_range": 0.02,
        "intermediate_size": 32,
        "max_position_embeddings": 4,
        "model_type": "lumi",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": True,
        "use_cache": False,
        "vocab_size": 128,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    master_port = _free_ipv4_port()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            "--nnodes=1",
            "--node-rank=0",
            "--nproc-per-node=2",
            str(root / "scripts" / "train_distributed.py"),
            "--manifest",
            str(manifest_path),
            "--final-corpus-root",
            str(staged),
            "--final-corpus-sequence-length",
            "4",
            "--final-corpus-vocab-size",
            "128",
            "--final-corpus-doc-token-id",
            "3",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--steps",
            "2",
            "--strategy",
            "ddp",
            "--gradient-accumulation-steps",
            "2",
            "--checkpoint-steps",
            "1",
            "--device",
            "cpu",
            "--backend",
            "gloo",
            "--skip-final-model-digest",
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=120,
    )
    state = json.loads((output / DISTRIBUTED_TRAINING_STATE_FILENAME).read_text())
    metrics = [
        json.loads(line)
        for line in (output / "distributed-metrics.jsonl").read_text().splitlines()
    ]
    assert state["status"] == "complete"
    assert state["step"] == 2
    assert state["sequence_cursor"] == 7
    assert state["tokens_seen"] == 28
    assert [row["global_tokens_per_step"] for row in metrics] == [16, 12]


def test_markerless_production_corpus_is_rejected_before_preflight() -> None:
    artifact = Path(
        "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/final-corpus-80b-v0.1/"
        "555f80f8396e-glue-production-run/evidence/results/shard-manifest.json"
    )
    if not artifact.is_file():
        pytest.skip("verified final corpus evidence is not mounted")
    with pytest.raises(ValueError, match="document-marker packing"):
        validate_final_corpus_training_manifest(
            artifact,
            sequence_length=4096,
            vocab_size=65_536,
            doc_token_id=3,
        )


def test_corrected_marker_aware_corpus_passes_production_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = Path(
        "/Users/afnan/Documents/Flowaistudio/lumi-artifacts/final-corpus-80b-v0.1/"
        "bdd656e14cee-marker-production-run/evidence/results/shard-manifest.json"
    )
    if not artifact.is_file():
        pytest.skip("verified marker-aware corpus evidence is not mounted")

    plan = build_production_training_preflight_plan(
        contract_path=root / "configs/4b-production-preflight-v0.1.json",
        manifest_path=artifact,
        model_config_path=root / "configs/lumi-4b-v0.2.json",
        source_commit="e" * 40,
    )

    assert plan["corpus"]["documents"] == 45_768_283
    assert plan["corpus"]["tokens"] == 80_000_000_000
    assert plan["corpus"]["trailing_tokens_not_consumed"] == 8_364
    assert plan["corpus"]["shards"] == 160


def test_account_segment_plan_is_credit_and_checkpoint_bounded() -> None:
    plan = build_account_segment_plan(
        run_fingerprint="f" * 64,
        current_step=0,
        current_tokens=0,
        total_tokens=80_000_000_000,
        nominal_global_tokens_per_step=1_048_576,
        checkpoint_steps=100,
        measured_tokens_per_second=100_000,
        throughput_safety_factor=0.8,
        hourly_rate_usd=31.92,
        available_credit_usd=7_500,
        credit_reserve_usd=500,
        provider_runtime_limit_hours=168,
    )

    assert plan["status"] == "planning_only"
    assert plan["stop"]["step"] % 100 == 0
    assert plan["stop"]["tokens"] <= 80_000_000_000
    assert plan["projected"]["cost_usd"] <= 7_000
    assert plan["projected"]["hours"] <= 168
