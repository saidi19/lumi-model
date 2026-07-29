#!/usr/bin/env python3
"""Run or resume Lumi's guarded DDP/FSDP trainer under torchrun."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.distributed_training import (
    destroy_distributed,
    initialize_distributed_from_environment,
    train_distributed_packed,
    write_final_model_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--strategy", choices=("ddp", "fsdp_full_shard"), required=True)
    parser.add_argument("--per-rank-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=10)
    parser.add_argument("--keep-checkpoints", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--document-isolation", action="store_true")
    parser.add_argument(
        "--optimizer-profile",
        choices=("legacy", "grouped_adamw_beta2_0.95_v0.1"),
        default="legacy",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--gradient-sync-mode",
        choices=("every_microbatch", "last_microbatch"),
        default="every_microbatch",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("gloo", "nccl"))
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--final-corpus-root", type=Path)
    parser.add_argument("--final-corpus-sequence-length", type=int, default=4096)
    parser.add_argument("--final-corpus-vocab-size", type=int, default=65_536)
    parser.add_argument("--final-corpus-doc-token-id", type=int, default=3)
    parser.add_argument("--skip-final-model-digest", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
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
        state = train_distributed_packed(
            context=context,
            manifest_path=args.manifest,
            config_path=args.config,
            output=args.output,
            steps=args.steps,
            strategy=args.strategy,
            per_rank_batch_size=args.per_rank_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoints=args.keep_checkpoints,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            schedule_steps=args.schedule_steps,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
            attention_implementation=args.attention_implementation,
            document_isolation=args.document_isolation,
            optimizer_profile=args.optimizer_profile,
            precision=args.precision,
            gradient_sync_mode=args.gradient_sync_mode,
            torch_compile=args.torch_compile,
            source_commit=args.source_commit,
            authorization_path=args.authorization,
            final_corpus_root=args.final_corpus_root,
            final_corpus_sequence_length=args.final_corpus_sequence_length,
            final_corpus_vocab_size=args.final_corpus_vocab_size,
            final_corpus_doc_token_id=args.final_corpus_doc_token_id,
            resume=args.resume,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    finally:
        destroy_distributed()
    if context.is_rank_zero:
        assert state is not None
        if not args.skip_final_model_digest:
            write_final_model_digest(
                config_path=args.config,
                checkpoint=args.output / state["latest_checkpoint"],
                output=args.output,
            )
        print(
            f"PASS: distributed step={state['step']} tokens={state['tokens_seen']} "
            f"loss={state['last_loss']:.6f} checkpoint={state['latest_checkpoint']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
