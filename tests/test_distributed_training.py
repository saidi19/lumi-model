from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from lumi import LumiConfig
from lumi.distributed_smoke import (
    plan_distributed_cuda_smoke,
    write_owner_authorization,
)
from lumi.distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    DistributedContext,
    EpochPermutationPlanner,
    _reconcile_checkpoint_metric,
    _validate_execution_scope,
    _validate_paid_authorization,
    _write_checksum_manifest,
    build_distributed_run_identity,
    distributed_batch_indices,
    verify_distributed_checkpoint,
)


def test_distributed_batches_partition_one_global_cursor() -> None:
    first, first_cursor = distributed_batch_indices(
        cursor=0,
        per_rank_batch_size=2,
        dataset_length=20,
        seed=7,
        rank=0,
        world_size=2,
    )
    second, second_cursor = distributed_batch_indices(
        cursor=0,
        per_rank_batch_size=2,
        dataset_length=20,
        seed=7,
        rank=1,
        world_size=2,
    )

    assert first_cursor == second_cursor == 4
    assert len(first) == len(second) == 2
    assert set(first).isdisjoint(second)
    replay, replay_cursor = distributed_batch_indices(
        cursor=0,
        per_rank_batch_size=2,
        dataset_length=20,
        seed=7,
        rank=0,
        world_size=2,
    )
    assert (replay, replay_cursor) == (first, first_cursor)


def test_epoch_permutation_planner_caches_and_crosses_epoch_exactly() -> None:
    planner = EpochPermutationPlanner(dataset_length=5, seed=11)

    first, cursor = planner.global_indices(cursor=3, count=4)
    cached = planner._permutation
    replay, replay_cursor = planner.global_indices(cursor=3, count=4)

    assert first == replay
    assert cursor == replay_cursor == 7
    assert planner._epoch == 1
    assert planner._permutation is not cached
    expected_first_epoch = torch.randperm(5, generator=torch.Generator().manual_seed(11)).tolist()
    expected_second_epoch = torch.randperm(5, generator=torch.Generator().manual_seed(12)).tolist()
    assert first == expected_first_epoch[3:] + expected_second_epoch[:2]


def test_local_preflight_scope_allows_only_tiny_cpu_work() -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )
    tiny = LumiConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=32,
    )

    scope = _validate_execution_scope(
        authorization_path=None,
        config=tiny,
        context=context,
        run_fingerprint="a" * 64,
        source_commit=None,
        strategy="ddp",
        steps=4,
        target_tokens=512,
    )

    assert scope["mode"] == "local_cpu_preflight"
    assert scope["paid_gpu"] is False
    assert scope["full_4b_training"] is False


def test_local_preflight_scope_rejects_4b_config() -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )
    config = LumiConfig(
        vocab_size=65_536,
        hidden_size=3_072,
        intermediate_size=8_384,
        num_hidden_layers=36,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=4_096,
    )

    with pytest.raises(PermissionError, match="model is too large"):
        _validate_execution_scope(
            authorization_path=None,
            config=config,
            context=context,
            run_fingerprint="a" * 64,
            source_commit=None,
            strategy="fsdp_full_shard",
            steps=1,
            target_tokens=8_192,
        )


def test_paid_smoke_authorization_cannot_approve_full_4b(tmp_path) -> None:
    path = tmp_path / "authorization.json"
    path.write_text(
        json.dumps(
            {
                "approved_through_step": 10,
                "full_4b_training": True,
                "owner_acknowledgement": (
                    "I approve this paid Lumi distributed smoke run through the recorded step."
                ),
                "paid_gpu": True,
                "run_fingerprint": "a" * 64,
                "run_kind": "lumi_distributed_smoke_v0.1",
                "schema_version": 1,
                "source_commit": "c" * 40,
                "status": "owner_approved",
                "strategy": "fsdp_full_shard",
                "world_size": 2,
            }
        ),
        encoding="utf-8",
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        backend="nccl",
        device=torch.device("cuda", 0),
    )

    with pytest.raises(ValueError, match="cannot approve full 4B"):
        _validate_paid_authorization(
            path,
            run_fingerprint="a" * 64,
            source_commit="c" * 40,
            context=context,
            strategy="fsdp_full_shard",
            steps=10,
        )


def test_distributed_identity_binds_paid_source_commit() -> None:
    common = {
        "config_bytes": b"{}",
        "manifest": {"build_fingerprint": "data"},
        "manifest_sha256": "d" * 64,
        "backend": "gloo",
        "world_size": 2,
        "strategy": "ddp",
        "per_rank_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "keep_checkpoints": 1,
        "seed": 7,
        "gradient_checkpointing": False,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "schedule_steps": None,
        "min_learning_rate_ratio": 0.1,
        "attention_implementation": None,
        "document_isolation": False,
        "optimizer_profile": "legacy",
        "precision": "fp32",
        "gradient_sync_mode": "every_microbatch",
        "torch_compile": False,
        "torch_version": "test",
    }

    historical = build_distributed_run_identity(**common)
    paid = build_distributed_run_identity(**common, source_commit="a" * 40)

    assert "source_commit" not in historical
    assert paid["source_commit"] == "a" * 40
    assert historical != paid
    with pytest.raises(ValueError, match="full 40-character"):
        build_distributed_run_identity(**common, source_commit="short")


def test_cuda_smoke_plan_and_owner_authorization_remain_narrow(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    data.mkdir()
    sequence_length = 4096
    sequences = 64
    row = [3, *([17] * (sequence_length - 1))]
    values = torch.tensor(row * sequences, dtype=torch.uint16).numpy().tobytes()
    shard = data / "shard-00000.bin"
    shard.write_bytes(values)
    shard_sha256 = hashlib.sha256(values).hexdigest()
    manifest = {
        "build_fingerprint": "fixture-build",
        "format": {
            "dtype": "uint16",
            "endianness": "little",
            "layout": "flat_fixed_sequences",
            "token_bytes": 2,
        },
        "packing": {"sequence_length": sequence_length},
        "provenance": {
            "build_fingerprint": (
                "0bce4d9e4a51de1b831b0226e0052ea33130733eae2bb61992c6eacaf203e0c8"
            ),
            "manifest_sha256": (
                "9228fa610d4a985d05e15057e8ef12619a14cbedbb5a6a5ab513d3aeca2832c8"
            ),
        },
        "schema_version": 1,
        "shards": [
            {
                "bytes": len(values),
                "name": shard.name,
                "sequences": sequences,
                "sha256": shard_sha256,
                "tokens": sequences * sequence_length,
            }
        ],
        "statistics": {
            "document_markers": sequences,
            "sequences": sequences,
            "tokens": sequences * sequence_length,
        },
        "status": "complete",
        "tokenizer": {"doc_token_id": 3, "vocab_size": 65536},
    }
    manifest_path = data / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = plan_distributed_cuda_smoke(
        contract_path=root / "configs/distributed-cuda-smoke-v0.1.json",
        manifest_path=manifest_path,
        config_path=root / "configs/lumi-4b-v0.2.json",
        source_commit="b" * 40,
        require_cuda=False,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    authorization = write_owner_authorization(
        plan_path=plan_path,
        output=tmp_path / "authorization.json",
        provider="test-provider",
        instance_type="2x-test-gpu",
        hourly_rate_usd=1.0,
        max_runtime_minutes=30,
    )

    assert plan["target"] == {
        "fresh_through_step": 1,
        "resume_through_step": 2,
        "tokens_through_resume": 16_384,
    }
    assert authorization["approved_through_step"] == 2
    assert authorization["paid_gpu"] is True
    assert authorization["full_4b_training"] is False


def test_distributed_checkpoint_manifest_detects_tampering(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-00000002"
    checkpoint.mkdir()
    (checkpoint / "model").mkdir()
    (checkpoint / "model" / "shard.bin").write_bytes(b"model")
    for rank in range(2):
        torch.save({"optimizer": {}, "rng_state": {}}, checkpoint / f"trainer-rank-{rank:05d}.pt")
    state = {
        "checkpoint_schema_version": DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "run_fingerprint": "b" * 64,
        "run_specification": {"distributed": {"world_size": 2}},
        "step": 2,
    }
    (checkpoint / DISTRIBUTED_TRAINING_STATE_FILENAME).write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    _write_checksum_manifest(checkpoint)

    assert verify_distributed_checkpoint(checkpoint)["step"] == 2
    (checkpoint / "model" / "shard.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        verify_distributed_checkpoint(checkpoint)


def test_distributed_checkpoint_manifest_rejects_path_escape(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    (checkpoint / "manifest.sha256").write_text(
        f"{digest}  ../outside.bin\n",
        encoding="utf-8",
    )
    (checkpoint / DISTRIBUTED_TRAINING_STATE_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid file"):
        verify_distributed_checkpoint(checkpoint)


def test_checkpoint_metric_recovers_publish_before_root_state_crash(tmp_path) -> None:
    metrics = tmp_path / "distributed-metrics.jsonl"
    step_one = {"checkpoint_seconds": None, "loss": 5.0, "step": 1}
    uncommitted = {"loss": 4.0, "step": 3}
    metrics.write_text(
        "\n".join(json.dumps(item) for item in (step_one, uncommitted)) + "\n",
        encoding="utf-8",
    )
    checkpoint_metric = {"checkpoint_seconds": None, "loss": 4.5, "step": 2}
    state = {"checkpoint_metric": checkpoint_metric, "step": 2}

    from lumi.pilot_training import _trim_metrics

    _trim_metrics(metrics, 2)
    _reconcile_checkpoint_metric(metrics, state)

    records = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert records[-1]["loss"] == 4.5
    assert records[-1]["recovered_from_checkpoint"] is True
    assert "checkpoint_metric" not in state
