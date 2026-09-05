"""Distributed assistant-and-tool-only SFT with exact fresh-process resume."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .configuration_lumi import LumiConfig
from .distributed_training import (
    DISTRIBUTED_METRICS_FILENAME,
    DISTRIBUTED_STRATEGIES,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    GRADIENT_SYNC_MODES,
    DistributedContext,
    EpochPermutationPlanner,
    _autocast_context,
    _prune_checkpoints,
    _rank_zero_object,
    _reduce_max,
    _reduce_sum,
    _restore_distributed_checkpoint,
    _save_distributed_checkpoint,
    _setup_output,
    _validate_execution_scope,
    _wrap_model,
)
from .modeling_lumi import LumiForCausalLM
from .pilot_training import (
    ATTENTION_IMPLEMENTATIONS,
    LR_SCHEDULES,
    OPTIMIZER_PROFILES,
    _append_metric,
    _create_optimizer,
    _fingerprint,
    _synchronize,
    _utc_now,
    _write_json_atomic,
    scheduled_learning_rate,
)
from .posttraining_sft_runtime import (
    EncodedSFTDataset,
    _verify_checksum_manifest,
    collate_sft_tensors,
)
from .posttraining_shards import verify_sft_shards

SFT_DISTRIBUTED_STATE_SCHEMA_VERSION = "lumi.sft-distributed-state.v0.1"
SFT_DISTRIBUTED_OBJECTIVE = "assistant_and_tool_only_cross_entropy_v0.1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _collective_validate_inputs(
    *,
    context: DistributedContext,
    encoded_shards: Path,
    source_checkpoint: Path,
    source_checkpoint_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Hash each node once and require one source/data identity on all ranks."""

    local: dict[str, Any]
    manifest: dict[str, Any] | None = None
    try:
        manifest_path = encoded_shards / "manifest.json"
        if context.local_rank == 0:
            manifest = verify_sft_shards(encoded_shards)
            _verify_checksum_manifest(
                source_checkpoint,
                expected_manifest_sha256=source_checkpoint_manifest_sha256,
            )
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "lumi.sft-encoded-shards.v0.1":
                raise ValueError("unsupported SFT encoded-shard manifest")
            observed_source = _sha256_file(source_checkpoint / "manifest.sha256")
            if observed_source != source_checkpoint_manifest_sha256:
                raise ValueError("SFT source-checkpoint manifest changed")
        local = {
            "error": None,
            "manifest_file_sha256": _sha256_file(manifest_path),
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "rank": context.rank,
            "seal_sha256": manifest["seal_sha256"],
            "source_checkpoint_manifest_sha256": source_checkpoint_manifest_sha256,
        }
    except Exception as error:  # noqa: BLE001 - report every rank before failing
        local = {"error": f"{type(error).__name__}: {error}", "rank": context.rank}

    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    failures = [item for item in gathered if item is not None and item.get("error")]
    if failures:
        details = "; ".join(
            f"rank {item['rank']}: {item['error']}" for item in failures
        )
        raise RuntimeError(f"distributed SFT input validation failed: {details}")
    identities = {
        (
            item["manifest_file_sha256"],
            item["manifest_internal_sha256"],
            item["seal_sha256"],
            item["source_checkpoint_manifest_sha256"],
        )
        for item in gathered
        if item is not None
    }
    if len(identities) != 1:
        raise RuntimeError("distributed ranks do not share one SFT input identity")
    assert manifest is not None
    manifest_file_sha256 = next(iter(identities))[0]
    return manifest, manifest_file_sha256


def build_distributed_sft_run_identity(
    *,
    source_config_bytes: bytes,
    source_checkpoint_manifest_sha256: str,
    shard_manifest: dict[str, Any],
    shard_manifest_file_sha256: str,
    backend: str,
    world_size: int,
    strategy: str,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    target_steps: int,
    seed: int,
    lr_schedule: str,
    warmup_steps: int,
    min_learning_rate_ratio: float,
    attention_implementation: str,
    optimizer_profile: str,
    precision: str,
    gradient_sync_mode: str,
    source_commit: str | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "data": {
            "manifest_file_sha256": shard_manifest_file_sha256,
            "manifest_sha256": shard_manifest["manifest_sha256"],
            "seal_sha256": shard_manifest["seal_sha256"],
            "sequence_order": "cached_epoch_randperm_v0.1",
            "split": "train",
        },
        "distributed": {
            "backend": backend,
            "checkpoint_format": "dcp-model_rank-local-optimizer-v0.1",
            "gradient_sync_mode": gradient_sync_mode,
            "per_rank_batch_size": per_rank_batch_size,
            "precision": precision,
            "strategy": strategy,
            "world_size": world_size,
        },
        "objective": SFT_DISTRIBUTED_OBJECTIVE,
        "optimizer": {
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_schedule": lr_schedule,
            "maximum_gradient_norm": max_grad_norm,
            "minimum_learning_rate_ratio": min_learning_rate_ratio,
            "optimizer_profile": optimizer_profile,
            "target_steps": target_steps,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
        },
        "schema_version": SFT_DISTRIBUTED_STATE_SCHEMA_VERSION,
        "seed": seed,
        "source": {
            "config_sha256": hashlib.sha256(source_config_bytes).hexdigest(),
            "manifest_sha256": source_checkpoint_manifest_sha256,
        },
        "torch_version": torch.__version__,
        "training": {
            "attention_implementation": attention_implementation,
            "context_window": int(shard_manifest["context_window"]),
            "loss_weighting": "global_shifted_supervised_token_mean_v0.1",
            "padding": "rank_local_right_padding_labels_minus_100_v0.1",
        },
    }
    if source_commit is not None:
        normalized = source_commit.strip().lower()
        if len(normalized) != 40 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError("source commit must be a full 40-character Git SHA-1")
        identity["source_commit"] = normalized
    return identity


def _validate_runtime_options(
    *,
    context: DistributedContext,
    target_steps: int,
    stop_after_step: int,
    strategy: str,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    checkpoint_steps: int,
    precision: str,
    gradient_sync_mode: str,
    lr_schedule: str,
    warmup_steps: int,
    attention_implementation: str,
    optimizer_profile: str,
) -> None:
    if not dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    if (context.rank, context.world_size) != (dist.get_rank(), dist.get_world_size()):
        raise RuntimeError("distributed context does not match the process group")
    if strategy not in DISTRIBUTED_STRATEGIES:
        raise ValueError("unsupported distributed SFT strategy")
    if gradient_sync_mode not in GRADIENT_SYNC_MODES:
        raise ValueError("unsupported distributed SFT gradient-sync mode")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported distributed SFT precision")
    if not 1 <= stop_after_step <= target_steps:
        raise ValueError("distributed SFT stop step must fit the target")
    if min(per_rank_batch_size, gradient_accumulation_steps, checkpoint_steps) < 1:
        raise ValueError("distributed SFT batch and checkpoint settings must be positive")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("distributed SFT optimizer settings are invalid")
    if lr_schedule not in LR_SCHEDULES or optimizer_profile not in OPTIMIZER_PROFILES:
        raise ValueError("distributed SFT optimizer profile changed")
    if warmup_steps < 0 or warmup_steps >= target_steps:
        raise ValueError("distributed SFT warmup does not fit the target")
    if attention_implementation not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError("unsupported distributed SFT attention implementation")
    if attention_implementation == "flash_attention_2" and context.device.type != "cuda":
        raise ValueError("FlashAttention distributed SFT requires CUDA")
    if context.device.type == "cuda" and precision != "bf16":
        raise ValueError("CUDA distributed SFT is pinned to BF16")


def train_distributed_sft(
    *,
    context: DistributedContext,
    encoded_shards: str | Path,
    source_checkpoint: str | Path,
    source_checkpoint_manifest_sha256: str,
    output: str | Path,
    target_steps: int,
    stop_after_step: int,
    strategy: str,
    per_rank_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_steps: int = 1,
    keep_checkpoints: int | None = None,
    seed: int = 2027,
    resume: bool = False,
    lr_schedule: str = "cosine",
    warmup_steps: int = 0,
    min_learning_rate_ratio: float = 0.1,
    attention_implementation: str = "eager",
    optimizer_profile: str = "grouped_adamw_beta2_0.95_v0.1",
    precision: str = "fp32",
    gradient_sync_mode: str = "last_microbatch",
    source_commit: str | None = None,
    authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run hash-bound distributed SFT and stop at a resumable boundary."""

    _validate_runtime_options(
        context=context,
        target_steps=target_steps,
        stop_after_step=stop_after_step,
        strategy=strategy,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        checkpoint_steps=checkpoint_steps,
        precision=precision,
        gradient_sync_mode=gradient_sync_mode,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        attention_implementation=attention_implementation,
        optimizer_profile=optimizer_profile,
    )
    encoded_shards = Path(encoded_shards)
    source_checkpoint = Path(source_checkpoint)
    output = Path(output)
    authorization = None if authorization_path is None else Path(authorization_path)
    shard_manifest, shard_manifest_file_sha256 = _collective_validate_inputs(
        context=context,
        encoded_shards=encoded_shards,
        source_checkpoint=source_checkpoint,
        source_checkpoint_manifest_sha256=source_checkpoint_manifest_sha256,
    )
    source_config_bytes = (source_checkpoint / "config.json").read_bytes()
    source_config = LumiConfig(**json.loads(source_config_bytes))
    source_config._attn_implementation = attention_implementation
    if int(shard_manifest["context_window"]) > source_config.max_position_embeddings:
        raise ValueError("distributed SFT data exceeds the source context window")

    run_identity = build_distributed_sft_run_identity(
        source_config_bytes=source_config_bytes,
        source_checkpoint_manifest_sha256=source_checkpoint_manifest_sha256,
        shard_manifest=shard_manifest,
        shard_manifest_file_sha256=shard_manifest_file_sha256,
        backend=context.backend,
        world_size=context.world_size,
        strategy=strategy,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        target_steps=target_steps,
        seed=seed,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        attention_implementation=attention_implementation,
        optimizer_profile=optimizer_profile,
        precision=precision,
        gradient_sync_mode=gradient_sync_mode,
        source_commit=source_commit,
    )
    run_fingerprint = _fingerprint(run_identity)
    maximum_target_tokens = (
        target_steps
        * context.world_size
        * per_rank_batch_size
        * gradient_accumulation_steps
        * source_config.max_position_embeddings
    )
    scope = _validate_execution_scope(
        authorization_path=authorization,
        config=source_config,
        context=context,
        run_fingerprint=run_fingerprint,
        source_commit=source_commit,
        strategy=strategy,
        steps=target_steps,
        target_tokens=maximum_target_tokens,
    )
    run_specification = {
        **run_identity,
        "authorization": scope,
        "locations": {
            "encoded_shards": str(encoded_shards.resolve()),
            "source_checkpoint": str(source_checkpoint.resolve()),
        },
    }
    state = _setup_output(
        context=context,
        output=output,
        run_fingerprint=run_fingerprint,
        run_specification=run_specification,
        resume=resume,
    )
    state.setdefault("records_seen", int(state["sequence_cursor"]))
    state.setdefault("supervised_tokens_seen", 0)

    torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LumiForCausalLM.from_pretrained(source_checkpoint, local_files_only=True)
    model.config._attn_implementation = attention_implementation
    distributed_model = _wrap_model(
        model,
        context=context,
        strategy=strategy,
        precision=precision,
    )
    optimizer = _create_optimizer(
        distributed_model,
        device=context.device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_profile=optimizer_profile,
    )
    torch.manual_seed(seed + 10_000 + context.rank)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 10_000 + context.rank)
    if state["latest_checkpoint"] is not None:
        _restore_distributed_checkpoint(
            context=context,
            checkpoint=output / state["latest_checkpoint"],
            model=distributed_model,
            optimizer=optimizer,
        )

    dataset = EncodedSFTDataset(encoded_shards, split="train")
    if len(dataset) < context.world_size * per_rank_batch_size:
        raise ValueError("distributed SFT data cannot fill one global microbatch")
    if dataset.maximum_record_tokens > source_config.max_position_embeddings:
        raise ValueError("distributed SFT record exceeds the source context window")
    planner = EpochPermutationPlanner(dataset_length=len(dataset), seed=seed)
    metrics_path = output / DISTRIBUTED_METRICS_FILENAME
    state_path = output / DISTRIBUTED_TRAINING_STATE_FILENAME
    distributed_model.train()

    while int(state["step"]) < stop_after_step:
        learning_rate_at_step = scheduled_learning_rate(
            base_learning_rate=learning_rate,
            step=int(state["step"]) + 1,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            schedule_steps=target_steps if lr_schedule == "cosine" else None,
            min_learning_rate_ratio=min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_step
        optimizer.zero_grad(set_to_none=True)
        step_cursor = int(state["sequence_cursor"])
        microbatches: list[tuple[dict[str, torch.Tensor], int, int]] = []
        local_step_tokens = 0
        local_step_supervised = 0
        for _ in range(gradient_accumulation_steps):
            indices, step_cursor = planner.rank_batch(
                cursor=step_cursor,
                per_rank_batch_size=per_rank_batch_size,
                rank=context.rank,
                world_size=context.world_size,
            )
            batch = collate_sft_tensors(
                [dataset[index] for index in indices],
                context_window=source_config.max_position_embeddings,
                pad_token_id=source_config.pad_token_id,
            )
            shifted_supervised = int(batch["labels"][:, 1:].ne(-100).sum())
            if shifted_supervised < 1:
                raise ValueError("distributed SFT microbatch has no shifted supervision")
            tokens = int(batch["attention_mask"].sum())
            microbatches.append((batch, shifted_supervised, tokens))
            local_step_supervised += shifted_supervised
            local_step_tokens += tokens
        global_step_supervised = int(_reduce_sum(local_step_supervised, context))
        global_step_tokens = int(_reduce_sum(local_step_tokens, context))
        if global_step_supervised < 1:
            raise RuntimeError("distributed SFT step has no global supervision")

        _synchronize(context.device)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()
        local_loss_numerator = 0.0
        for index, (cpu_batch, shifted_supervised, _) in enumerate(microbatches):
            batch = {
                name: value.to(
                    context.device,
                    non_blocking=context.device.type == "cuda",
                )
                for name, value in cpu_batch.items()
            }
            defer_sync = (
                gradient_sync_mode == "last_microbatch"
                and index < len(microbatches) - 1
            )
            synchronization = distributed_model.no_sync() if defer_sync else nullcontext()
            with synchronization:
                with _autocast_context(context.device, precision):
                    loss = distributed_model(**batch).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("distributed SFT produced non-finite loss")
                gradient_weight = (
                    context.world_size * shifted_supervised / global_step_supervised
                )
                (loss * gradient_weight).backward()
            local_loss_numerator += float(loss.detach()) * shifted_supervised

        if strategy == "fsdp_full_shard":
            gradient_norm = distributed_model.clip_grad_norm_(max_grad_norm)
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                distributed_model.parameters(),
                max_grad_norm,
            )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("distributed SFT produced non-finite gradient")
        optimizer.step()
        _synchronize(context.device)
        elapsed = _reduce_max(max(time.perf_counter() - started, 1e-9), context)
        global_loss = _reduce_sum(local_loss_numerator, context) / global_step_supervised
        global_gradient_norm = _reduce_max(float(gradient_norm), context)

        state["step"] = int(state["step"]) + 1
        state["sequence_cursor"] = step_cursor
        state["records_seen"] = step_cursor
        state["tokens_seen"] = int(state["tokens_seen"]) + global_step_tokens
        state["supervised_tokens_seen"] = (
            int(state["supervised_tokens_seen"]) + global_step_supervised
        )
        state["last_loss"] = global_loss
        state["best_loss"] = (
            global_loss
            if state["best_loss"] is None
            else min(float(state["best_loss"]), global_loss)
        )
        state["updated_at"] = _utc_now()
        metric = {
            "backend": context.backend,
            "device": context.device.type,
            "elapsed_seconds": elapsed,
            "global_records_per_step": (
                context.world_size
                * per_rank_batch_size
                * gradient_accumulation_steps
            ),
            "gradient_norm": global_gradient_norm,
            "learning_rate": learning_rate_at_step,
            "loss": global_loss,
            "objective": SFT_DISTRIBUTED_OBJECTIVE,
            "step": state["step"],
            "strategy": strategy,
            "supervised_tokens": global_step_supervised,
            "supervised_tokens_seen": state["supervised_tokens_seen"],
            "tokens": global_step_tokens,
            "tokens_per_second": global_step_tokens / elapsed,
            "tokens_seen": state["tokens_seen"],
            "world_size": context.world_size,
        }
        if context.is_rank_zero:
            print(
                f"step={state['step']:06d} loss={global_loss:.6f} "
                f"supervised_tokens={global_step_supervised}",
                flush=True,
            )
        should_checkpoint = (
            int(state["step"]) % checkpoint_steps == 0
            or int(state["step"]) == stop_after_step
        )
        if should_checkpoint:
            _synchronize(context.device)
            checkpoint_started = time.perf_counter()
            checkpoint = _save_distributed_checkpoint(
                context=context,
                output=output,
                model=distributed_model,
                optimizer=optimizer,
                state=state,
                checkpoint_metric={**metric, "checkpoint_seconds": None},
            )
            _synchronize(context.device)
            checkpoint_seconds = _reduce_max(
                time.perf_counter() - checkpoint_started,
                context,
            )
            state["latest_checkpoint"] = checkpoint.name
            state["last_checkpoint_seconds"] = checkpoint_seconds
            metric["checkpoint_seconds"] = checkpoint_seconds
        else:
            metric["checkpoint_seconds"] = None

        if context.is_rank_zero:
            _append_metric(metrics_path, metric)
            if should_checkpoint:
                _write_json_atomic(state_path, state)
        dist.barrier()
        if should_checkpoint and keep_checkpoints is not None:
            _rank_zero_object(
                context,
                lambda: _prune_checkpoints(
                    output,
                    run_fingerprint=run_fingerprint,
                    keep_checkpoints=keep_checkpoints,
                ),
            )
            dist.barrier()

    if context.is_rank_zero:
        state["status"] = (
            "complete" if int(state["step"]) == target_steps else "paused"
        )
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)
    dist.barrier()
    result = [state if context.is_rank_zero else None]
    dist.broadcast_object_list(result, src=0)
    return result[0]


__all__ = [
    "SFT_DISTRIBUTED_OBJECTIVE",
    "SFT_DISTRIBUTED_STATE_SCHEMA_VERSION",
    "build_distributed_sft_run_identity",
    "train_distributed_sft",
]
