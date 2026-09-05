#!/usr/bin/env python3
"""Run or fresh-resume Lumi distributed masked-label SFT under torchrun."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.distributed_training import (
    destroy_distributed,
    initialize_distributed_from_environment,
    write_final_model_digest,
)
from lumi.posttraining_distributed_sft import train_distributed_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-shards", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--strategy", choices=("ddp", "fsdp_full_shard"), required=True)
    parser.add_argument("--per-rank-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=1)
    parser.add_argument("--keep-checkpoints", type=int)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument(
        "--optimizer-profile",
        choices=("legacy", "grouped_adamw_beta2_0.95_v0.1"),
        default="grouped_adamw_beta2_0.95_v0.1",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--gradient-sync-mode",
        choices=("every_microbatch", "last_microbatch"),
        default="last_microbatch",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("gloo", "nccl"))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--source-commit")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-model-digest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = initialize_distributed_from_environment(
        device=args.device,
        backend=args.backend,
        timeout_seconds=args.timeout_seconds,
    )
    state = None
    try:
        state = train_distributed_sft(
            context=context,
            encoded_shards=args.encoded_shards,
            source_checkpoint=args.source_checkpoint,
            source_checkpoint_manifest_sha256=args.source_checkpoint_manifest_sha256,
            output=args.output,
            target_steps=args.target_steps,
            stop_after_step=args.stop_after_step,
            strategy=args.strategy,
            per_rank_batch_size=args.per_rank_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoints=args.keep_checkpoints,
            seed=args.seed,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            attention_implementation=args.attention_implementation,
            optimizer_profile=args.optimizer_profile,
            precision=args.precision,
            gradient_sync_mode=args.gradient_sync_mode,
            source_commit=args.source_commit,
            authorization_path=args.authorization,
            resume=args.resume,
        )
    finally:
        destroy_distributed()
    if context.is_rank_zero:
        assert state is not None
        if not args.skip_final_model_digest:
            write_final_model_digest(
                config_path=args.source_checkpoint / "config.json",
                checkpoint=args.output / state["latest_checkpoint"],
                output=args.output,
            )
        print(
            f"PASS: distributed SFT status={state['status']} step={state['step']} "
            f"records={state['records_seen']} supervised={state['supervised_tokens_seen']} "
            f"loss={state['last_loss']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
