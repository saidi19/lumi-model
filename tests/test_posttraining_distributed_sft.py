from __future__ import annotations

from lumi.posttraining_distributed_sft import (
    SFT_DISTRIBUTED_OBJECTIVE,
    build_distributed_sft_run_identity,
)


def _identity() -> dict:
    return build_distributed_sft_run_identity(
        source_config_bytes=b'{"model_type":"lumi"}',
        source_checkpoint_manifest_sha256="a" * 64,
        shard_manifest={
            "context_window": 4096,
            "manifest_sha256": "b" * 64,
            "seal_sha256": "c" * 64,
        },
        shard_manifest_file_sha256="d" * 64,
        backend="gloo",
        world_size=2,
        strategy="fsdp_full_shard",
        per_rank_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        target_steps=2,
        seed=2027,
        lr_schedule="cosine",
        warmup_steps=0,
        min_learning_rate_ratio=0.1,
        attention_implementation="eager",
        optimizer_profile="grouped_adamw_beta2_0.95_v0.1",
        precision="fp32",
        gradient_sync_mode="last_microbatch",
        source_commit=None,
    )


def test_distributed_sft_identity_binds_masked_objective_and_data() -> None:
    identity = _identity()

    assert identity["objective"] == SFT_DISTRIBUTED_OBJECTIVE
    assert identity["data"] == {
        "manifest_file_sha256": "d" * 64,
        "manifest_sha256": "b" * 64,
        "seal_sha256": "c" * 64,
        "sequence_order": "cached_epoch_randperm_v0.1",
        "split": "train",
    }
    assert identity["training"] == {
        "attention_implementation": "eager",
        "context_window": 4096,
        "loss_weighting": "global_shifted_supervised_token_mean_v0.1",
        "padding": "rank_local_right_padding_labels_minus_100_v0.1",
    }
    assert identity["distributed"]["strategy"] == "fsdp_full_shard"
