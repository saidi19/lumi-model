from __future__ import annotations

import hashlib
import json

import pytest
import torch

from lumi import LumiConfig
from lumi.distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    DistributedContext,
    EpochPermutationPlanner,
    _reconcile_checkpoint_metric,
    _validate_execution_scope,
    _validate_paid_authorization,
    _write_checksum_manifest,
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
            context=context,
            strategy="fsdp_full_shard",
            steps=10,
        )


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
