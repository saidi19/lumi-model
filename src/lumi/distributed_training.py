"""Correctness-first distributed training primitives for Lumi.

This module is intentionally separate from :mod:`lumi.pilot_training`.  The
archived single-device checkpoints keep their byte-for-byte identity while the
distributed path can use rank-aware data assignment and sharded checkpoints.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from .configuration_lumi import LumiConfig
from .modeling_lumi import (
    LumiDecoderLayer,
    LumiForCausalLM,
    build_packed_document_metadata,
)
from .packed_data import PackedTokenDataset, validate_packed_manifest
from .parameter_count import parameter_breakdown
from .pilot_training import (
    ATTENTION_IMPLEMENTATIONS,
    LR_SCHEDULES,
    OPTIMIZER_PROFILES,
    _append_metric,
    _canonical_json_bytes,
    _create_optimizer,
    _fingerprint,
    _fsync_tree,
    _optimizer_identity,
    _synchronize,
    _trim_metrics,
    _utc_now,
    _write_json_atomic,
    scheduled_learning_rate,
)
from .production_training import (
    FINAL_CORPUS_SHARD_MANIFEST_GATE,
    FinalCorpusTokenDataset,
    validate_final_corpus_training_manifest,
)

DISTRIBUTED_TRAINING_SCHEMA_VERSION = 1
DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION = 1
DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION = 1
DISTRIBUTED_TRAINING_STATE_FILENAME = "distributed-training-state.json"
DISTRIBUTED_METRICS_FILENAME = "distributed-metrics.jsonl"
DISTRIBUTED_STRATEGIES = {"ddp", "fsdp_full_shard"}
DISTRIBUTED_PRECISIONS = {"fp32", "bf16"}
GRADIENT_SYNC_MODES = {"every_microbatch", "last_microbatch"}

LOCAL_PREFLIGHT_MAX_PARAMETERS = 20_000_000
LOCAL_PREFLIGHT_MAX_STEPS = 8
LOCAL_PREFLIGHT_MAX_TOKENS = 2_000_000
LOCAL_PREFLIGHT_MAX_WORLD_SIZE = 2
PAID_SMOKE_MAX_STEPS = 1_000
PAID_LEARNING_PILOT_MAX_STEPS = 80
PAID_LEARNING_PILOT_MAX_TOKENS = 83_886_080
PAID_LEARNING_PILOT_WORLD_SIZES = {4, 8}
PAID_LEARNING_PILOT_MAX_RUNTIME_MINUTES = {4: 480, 8: 240}
PAID_LEARNING_PILOT_RUN_KIND = "lumi_4b_learning_pilot_v0.1"
PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT = (
    "I approve this bounded paid Lumi-4B learning pilot through the recorded step."
)
PAID_4B_PRODUCTION_RUN_KIND = "lumi_4b_80b_pretraining_v0.1"
PAID_4B_PRODUCTION_WORLD_SIZE = 8
PAID_4B_PRODUCTION_MAX_STEPS = 76_294
PAID_4B_PRODUCTION_TOKEN_CAP = 80_000_000_000
PAID_4B_PRODUCTION_CHECKPOINT_STEPS = 250
PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES = 10_080
PAID_4B_PRODUCTION_MAX_COST_USD = 7_000
PAID_4B_PRODUCTION_FLASH_ATTENTION_VERSION = "2.8.3"
PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT = (
    "I approve this paid Lumi-4B/80B production segment through the recorded "
    "step and token cap."
)


def _validate_source_commit(source_commit: str | None) -> str | None:
    if source_commit is None:
        return None
    normalized = source_commit.strip().lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source commit must be a full 40-character Git SHA-1")
    return normalized


def _validate_production_cuda_runtime(context: DistributedContext) -> None:
    """Fail closed unless this rank is on the measured 80GB H100 runtime."""

    if context.device.type != "cuda" or context.backend != "nccl":
        raise ValueError("production runtime requires CUDA/NCCL")
    capability = torch.cuda.get_device_capability(context.device)
    if capability != (9, 0):
        raise RuntimeError("production runtime requires H100 compute capability 9.0")
    name = torch.cuda.get_device_name(context.device)
    if "h100" not in name.lower():
        raise RuntimeError("production runtime requires homogeneous H100 GPUs")
    total_memory = torch.cuda.get_device_properties(context.device).total_memory
    if not 80_000_000_000 <= total_memory <= 90_000_000_000:
        raise RuntimeError("production runtime requires 80GB H100 GPUs")
    try:
        import flash_attn
    except ImportError as error:
        raise RuntimeError("production runtime requires FlashAttention") from error
    if flash_attn.__version__ != PAID_4B_PRODUCTION_FLASH_ATTENTION_VERSION:
        raise RuntimeError("production FlashAttention version changed")


def build_distributed_run_identity(
    *,
    config_bytes: bytes,
    manifest: dict[str, Any],
    manifest_sha256: str,
    backend: str,
    world_size: int,
    strategy: str,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    keep_checkpoints: int | None,
    checkpoint_steps: int | None = None,
    seed: int,
    gradient_checkpointing: bool,
    lr_schedule: str,
    warmup_steps: int,
    schedule_steps: int | None,
    min_learning_rate_ratio: float,
    attention_implementation: str | None,
    document_isolation: bool,
    optimizer_profile: str,
    precision: str,
    gradient_sync_mode: str,
    torch_compile: bool,
    source_commit: str | None = None,
    torch_version: str | None = None,
) -> dict[str, Any]:
    """Build the canonical identity shared by planning and training.

    Historical CPU-preflight identities stay byte-compatible when
    ``source_commit`` is omitted. Paid CUDA smoke runs must provide a full
    commit and therefore bind their authorization to the exact source tree.
    """

    normalized_commit = _validate_source_commit(source_commit)
    optimizer_identity = _optimizer_identity(
        batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        keep_checkpoints=keep_checkpoints,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        max_grad_norm=max_grad_norm,
        min_learning_rate_ratio=min_learning_rate_ratio,
        schedule_steps=schedule_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        attention_implementation=attention_implementation,
        document_isolation=document_isolation,
        optimizer_profile=optimizer_profile,
    )
    identity: dict[str, Any] = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data": {
            "build_fingerprint": manifest["build_fingerprint"],
            "manifest_sha256": manifest_sha256,
            "sequence_order": "cached_epoch_randperm_v0.1",
        },
        "distributed": {
            "backend": backend,
            "checkpoint_format": "dcp-model_rank-local-optimizer-v0.1",
            "gradient_sync_mode": gradient_sync_mode,
            "per_rank_batch_size": per_rank_batch_size,
            "precision": precision,
            "strategy": strategy,
            "torch_compile": torch_compile,
            "world_size": world_size,
        },
        "optimizer": optimizer_identity,
        "schema_version": DISTRIBUTED_TRAINING_SCHEMA_VERSION,
        "seed": seed,
        "torch_version": torch.__version__ if torch_version is None else torch_version,
    }
    if checkpoint_steps is not None:
        if (
            not isinstance(checkpoint_steps, int)
            or isinstance(checkpoint_steps, bool)
            or checkpoint_steps < 1
        ):
            raise ValueError("identity checkpoint steps must be positive")
        identity["distributed"]["checkpoint_steps"] = checkpoint_steps
    if normalized_commit is not None:
        identity["source_commit"] = normalized_commit
    return identity


@dataclass(frozen=True)
class DistributedContext:
    """One initialized process-group participant."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


class EpochPermutationPlanner:
    """Cache one deterministic epoch permutation instead of rebuilding it per batch."""

    def __init__(self, *, dataset_length: int, seed: int) -> None:
        if dataset_length < 1:
            raise ValueError("distributed dataset length must be positive")
        self.dataset_length = dataset_length
        self.seed = seed
        self._epoch: int | None = None
        self._permutation: torch.Tensor | None = None

    def _for_epoch(self, epoch: int) -> torch.Tensor:
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(
                self.dataset_length,
                generator=generator,
                dtype=torch.int64,
            )
            self._epoch = epoch
        assert self._permutation is not None
        return self._permutation

    def global_indices(self, *, cursor: int, count: int) -> tuple[list[int], int]:
        if cursor < 0 or count < 1:
            raise ValueError("distributed cursor must be non-negative and count positive")
        indices: list[int] = []
        next_cursor = cursor
        while len(indices) < count:
            epoch, offset = divmod(next_cursor, self.dataset_length)
            permutation = self._for_epoch(epoch)
            take = min(count - len(indices), self.dataset_length - offset)
            indices.extend(permutation[offset : offset + take].tolist())
            next_cursor += take
        return indices, next_cursor

    def rank_batch(
        self,
        *,
        cursor: int,
        per_rank_batch_size: int,
        rank: int,
        world_size: int,
    ) -> tuple[list[int], int]:
        if per_rank_batch_size < 1 or world_size < 1:
            raise ValueError("batch size and world size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be inside the distributed world")
        global_batch_size = per_rank_batch_size * world_size
        global_indices, next_cursor = self.global_indices(
            cursor=cursor,
            count=global_batch_size,
        )
        start = rank * per_rank_batch_size
        return global_indices[start : start + per_rank_batch_size], next_cursor


@dataclass(frozen=True)
class FinalCorpusMicrobatch:
    """One rank's exact contribution to a possibly partial final microbatch."""

    active_global_sequences: int
    active_local_sequences: int
    global_offset: int

    @property
    def rank_is_active(self) -> bool:
        return self.active_local_sequences > 0


def final_corpus_microbatch_plan(
    *,
    active_sequences_this_step: int,
    accumulation_index: int,
    per_rank_batch_size: int,
    rank: int,
    world_size: int,
) -> FinalCorpusMicrobatch:
    """Allocate an exact corpus tail without duplicating any real sequence.

    Full production steps are unaffected.  On the final partial step, ranks may
    receive fewer than ``per_rank_batch_size`` sequences while every rank still
    participates in the distributed backward pass.
    """

    if active_sequences_this_step < 1:
        raise ValueError("active final-corpus sequences must be positive")
    if accumulation_index < 0:
        raise ValueError("accumulation index cannot be negative")
    if per_rank_batch_size < 1 or world_size < 1:
        raise ValueError("batch size and world size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside the distributed world")

    global_microbatch = per_rank_batch_size * world_size
    consumed_before = accumulation_index * global_microbatch
    remaining = active_sequences_this_step - consumed_before
    if remaining <= 0:
        raise ValueError("accumulation index is outside the active final-corpus step")
    active_global = min(global_microbatch, remaining)
    rank_start = rank * per_rank_batch_size
    active_local = max(0, min(per_rank_batch_size, active_global - rank_start))
    return FinalCorpusMicrobatch(
        active_global_sequences=active_global,
        active_local_sequences=active_local,
        global_offset=rank_start,
    )


def _environment_integer(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required; launch with torchrun")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 0:
        raise RuntimeError(f"{name} cannot be negative")
    return value


def initialize_distributed_from_environment(
    *,
    device: str = "auto",
    backend: str | None = None,
    timeout_seconds: int = 1_800,
) -> DistributedContext:
    """Initialize the process group described by ``torchrun`` environment variables."""

    if dist.is_initialized():
        raise RuntimeError("a distributed process group is already initialized")
    if timeout_seconds < 1:
        raise ValueError("distributed timeout must be positive")

    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE")
    if world_size < 2:
        raise RuntimeError("the distributed trainer requires at least two processes")
    if rank >= world_size:
        raise RuntimeError("RANK must be smaller than WORLD_SIZE")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("distributed device must be auto, cpu, or cuda")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA distributed training was requested but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
        torch.cuda.set_device(local_rank)
        training_device = torch.device("cuda", local_rank)
    else:
        training_device = torch.device("cpu")

    selected_backend = backend or ("nccl" if training_device.type == "cuda" else "gloo")
    if selected_backend not in {"gloo", "nccl"}:
        raise ValueError("distributed backend must be gloo or nccl")
    if selected_backend == "nccl" and training_device.type != "cuda":
        raise ValueError("NCCL requires CUDA")
    if training_device.type == "cuda" and selected_backend != "nccl":
        raise ValueError("production CUDA training requires NCCL")

    dist.init_process_group(
        backend=selected_backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=selected_backend,
        device=training_device,
    )


def destroy_distributed() -> None:
    """Destroy the active process group without hiding an earlier training error."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_batch_indices(
    *,
    cursor: int,
    per_rank_batch_size: int,
    dataset_length: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[list[int], int]:
    """Return one rank's disjoint slice of a deterministic global batch.

    The returned cursor advances by the *global* batch size.  Every rank can
    therefore reconstruct its own samples from the same checkpoint cursor
    without serializing a sampler object.
    """

    if cursor < 0:
        raise ValueError("distributed data cursor cannot be negative")
    if per_rank_batch_size < 1 or dataset_length < 1 or world_size < 1:
        raise ValueError("batch size, dataset length, and world size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside the distributed world")
    planner = EpochPermutationPlanner(dataset_length=dataset_length, seed=seed)
    return planner.rank_batch(
        cursor=cursor,
        per_rank_batch_size=per_rank_batch_size,
        rank=rank,
        world_size=world_size,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksum_manifest(root: Path) -> int:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            entries.append((_sha256_file(path), path.relative_to(root).as_posix()))
    if not entries:
        raise RuntimeError("refusing to publish an empty distributed checkpoint")
    manifest = root / "manifest.sha256"
    with manifest.open("w", encoding="utf-8") as handle:
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(entries)


def verify_distributed_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Verify one immutable distributed checkpoint and return its state."""

    checkpoint = Path(checkpoint)
    manifest = checkpoint / "manifest.sha256"
    state_path = checkpoint / DISTRIBUTED_TRAINING_STATE_FILENAME
    if not checkpoint.is_dir() or not manifest.is_file() or not state_path.is_file():
        raise ValueError(f"incomplete distributed checkpoint: {checkpoint}")
    entries = 0
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint manifest line {line_number}") from error
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid checkpoint digest at line {line_number}")
        relative_path = Path(relative)
        path = checkpoint / relative_path
        if (
            relative_path.is_absolute()
            or not path.resolve().is_relative_to(checkpoint.resolve())
            or not path.is_file()
        ):
            raise ValueError(f"checkpoint manifest references an invalid file: {relative}")
        if _sha256_file(path) != digest:
            raise ValueError(f"distributed checkpoint hash changed: {relative}")
        entries += 1
    if entries < 3:
        raise ValueError("distributed checkpoint manifest is unexpectedly small")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("checkpoint_schema_version") != DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported distributed checkpoint schema")
    expected_ranks = state.get("run_specification", {}).get("distributed", {}).get("world_size")
    if not isinstance(expected_ranks, int) or expected_ranks < 2:
        raise ValueError("checkpoint has an invalid distributed world size")
    for rank in range(expected_ranks):
        if not (checkpoint / f"trainer-rank-{rank:05d}.pt").is_file():
            raise ValueError(f"checkpoint is missing trainer state for rank {rank}")
    return state


def _discover_latest_checkpoint(
    output: Path, *, run_fingerprint: str
) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        state = verify_distributed_checkpoint(path)
        step = state.get("step")
        if state.get("run_fingerprint") == run_fingerprint and isinstance(step, int):
            candidates.append((step, path, state))
    if not candidates:
        return None
    _, path, state = max(candidates, key=lambda item: item[0])
    return path, state


def _prune_checkpoints(
    output: Path, *, run_fingerprint: str, keep_checkpoints: int
) -> list[str]:
    candidates = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        state = verify_distributed_checkpoint(path)
        if state.get("run_fingerprint") == run_fingerprint:
            candidates.append((int(state["step"]), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for _, path in candidates[keep_checkpoints:]:
        shutil.rmtree(path)
        removed.append(path.name)
    if removed:
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return removed


def _rank_zero_object(context: DistributedContext, callback) -> Any:
    envelope: list[Any] = [None]
    if context.is_rank_zero:
        try:
            envelope[0] = {"ok": True, "value": callback()}
        except Exception as error:  # noqa: BLE001 - serialized to every rank
            envelope[0] = {
                "error": f"{type(error).__name__}: {error}",
                "ok": False,
            }
    dist.broadcast_object_list(envelope, src=0)
    result = envelope[0]
    if not result["ok"]:
        raise RuntimeError(f"rank-zero operation failed: {result['error']}")
    return result["value"]


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _wrap_model(
    model: LumiForCausalLM,
    *,
    context: DistributedContext,
    strategy: str,
    precision: str,
) -> nn.Module:
    if strategy == "ddp":
        model.to(context.device)
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        ddp_options: dict[str, Any] = {
            "device_ids": device_ids,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if "forward_sync_buffers" in inspect.signature(DistributedDataParallel).parameters:
            ddp_options["forward_sync_buffers"] = False
        else:
            ddp_options["broadcast_buffers"] = False
        return DistributedDataParallel(model, **ddp_options)

    mixed_precision = None
    if precision == "bf16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LumiDecoderLayer},
    )
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=context.device,
        forward_prefetch=False,
        limit_all_gathers=True,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=False,
        use_orig_params=True,
    )


def _runtime_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_runtime_rng_state(payload: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["cpu"])
    if device.type == "cuda" and "cuda" in payload:
        torch.cuda.set_rng_state(payload["cuda"], device=device)


def _save_distributed_checkpoint(
    *,
    context: DistributedContext,
    output: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    checkpoint_metric: dict[str, Any],
) -> Path:
    checkpoint_name = f"checkpoint-{state['step']:08d}"
    checkpoint = output / checkpoint_name

    def prepare() -> str:
        if checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
        temporary_name = f".{checkpoint_name}.{uuid.uuid4().hex}.tmp"
        (output / temporary_name).mkdir()
        return temporary_name

    temporary_name = _rank_zero_object(context, prepare)
    temporary = output / temporary_name
    dist.barrier()

    model_state = get_model_state_dict(model)
    dcp.save({"model": model_state}, checkpoint_id=temporary / "model")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "rng_state": _runtime_rng_state(context.device),
        },
        temporary / f"trainer-rank-{context.rank:05d}.pt",
    )
    dist.barrier()

    checkpoint_state = {
        **state,
        "checkpoint_schema_version": DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_metric": checkpoint_metric,
        "latest_checkpoint": checkpoint_name,
        "status": "checkpoint",
        "updated_at": _utc_now(),
    }

    def publish() -> str:
        _write_json_atomic(
            temporary / DISTRIBUTED_TRAINING_STATE_FILENAME,
            checkpoint_state,
        )
        _write_checksum_manifest(temporary)
        _fsync_tree(temporary)
        temporary.replace(checkpoint)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        verify_distributed_checkpoint(checkpoint)
        return checkpoint_name

    _rank_zero_object(context, publish)
    dist.barrier()
    return checkpoint


def _restore_distributed_checkpoint(
    *,
    context: DistributedContext,
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    verify_distributed_checkpoint(checkpoint)
    model_state = get_model_state_dict(model)
    dcp.load({"model": model_state}, checkpoint_id=checkpoint / "model")
    incompatible = set_model_state_dict(model, model_state)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"distributed model restore was incompatible: {incompatible}")
    trainer = torch.load(
        checkpoint / f"trainer-rank-{context.rank:05d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(trainer["optimizer"])
    _restore_runtime_rng_state(trainer["rng_state"], context.device)
    dist.barrier()


def _state_dict_digest(state: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    parameter_values = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        parameter_values += tensor.numel()
    return {
        "algorithm": "sha256-canonical-full-model-state-v0.1",
        "parameter_values_with_tied_aliases": parameter_values,
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
    }


def write_final_model_digest(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Load a distributed model checkpoint into one plain CPU model and hash it.

    This deliberately runs *after* the process group is destroyed.  Gathering
    a classic FSDP full state in-process is unstable on macOS CPU builds and is
    unnecessary: DCP can reshard the same checkpoint directly into a plain
    model for independent verification.
    """

    checkpoint = Path(checkpoint)
    output = Path(output)
    verify_distributed_checkpoint(checkpoint)
    config = LumiConfig(**json.loads(Path(config_path).read_bytes()))
    model = LumiForCausalLM(config)
    state = model.state_dict()
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model", no_dist=True)
    incompatible = model.load_state_dict(state)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"plain-model distributed restore was incompatible: {incompatible}")
    result = {
        **_state_dict_digest(state),
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_sha256": _sha256_file(checkpoint / "manifest.sha256"),
    }
    _write_json_atomic(output / "final-model-digest.json", result)
    return result


def _validate_paid_authorization(
    path: Path,
    *,
    run_fingerprint: str,
    source_commit: str | None,
    context: DistributedContext,
    strategy: str,
    steps: int,
    target_tokens: int | None = None,
    global_tokens_per_step: int | None = None,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != DISTRIBUTED_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported distributed launch authorization")
    if authorization.get("status") != "owner_approved":
        raise ValueError("distributed launch is not owner-approved")
    run_kind = authorization.get("run_kind")
    if run_kind not in {
        "lumi_distributed_smoke_v0.1",
        PAID_LEARNING_PILOT_RUN_KIND,
        PAID_4B_PRODUCTION_RUN_KIND,
    }:
        raise ValueError("authorization targets a different run kind")
    if authorization.get("run_fingerprint") != run_fingerprint:
        raise ValueError("authorization does not match the distributed run")
    normalized_commit = _validate_source_commit(source_commit)
    if normalized_commit is None:
        raise ValueError("paid distributed smoke requires an exact source commit")
    if authorization.get("source_commit") != normalized_commit:
        raise ValueError("authorization source commit changed")
    if authorization.get("world_size") != context.world_size:
        raise ValueError("authorization world size changed")
    if authorization.get("strategy") != strategy:
        raise ValueError("authorization strategy changed")
    approved_step = authorization.get("approved_through_step")
    if authorization.get("paid_gpu") is not True:
        raise ValueError("paid CUDA launch must be explicit")

    if run_kind == "lumi_distributed_smoke_v0.1":
        if authorization.get("full_4b_training") is not False:
            raise ValueError("the bounded authorization cannot approve full 4B training")
        if not isinstance(approved_step, int) or not (
            steps <= approved_step <= PAID_SMOKE_MAX_STEPS
        ):
            raise ValueError("authorization step boundary is invalid")
        if authorization.get("owner_acknowledgement") != (
            "I approve this paid Lumi distributed smoke run through the recorded step."
        ):
            raise ValueError("authorization acknowledgement changed")
        return authorization

    if run_kind == PAID_LEARNING_PILOT_RUN_KIND:
        if authorization.get("full_4b_training") is not False:
            raise ValueError("the bounded authorization cannot approve full 4B training")
        if context.world_size not in PAID_LEARNING_PILOT_WORLD_SIZES:
            raise ValueError("learning-pilot authorization requires four or eight GPUs")
        if strategy != "fsdp_full_shard":
            raise ValueError("learning-pilot authorization requires FSDP full sharding")
        if approved_step != PAID_LEARNING_PILOT_MAX_STEPS or steps > approved_step:
            raise ValueError("learning-pilot step boundary is invalid")
        approved_tokens = authorization.get("approved_token_cap")
        if approved_tokens != PAID_LEARNING_PILOT_MAX_TOKENS:
            raise ValueError("learning-pilot token boundary changed")
        if target_tokens is None or not 0 < target_tokens <= approved_tokens:
            raise ValueError("learning-pilot target tokens exceed authorization")
        if authorization.get("owner_acknowledgement") != (
            PAID_LEARNING_PILOT_OWNER_ACKNOWLEDGEMENT
        ):
            raise ValueError("learning-pilot authorization acknowledgement changed")
        max_runtime = authorization.get("max_runtime_minutes")
        max_allowed_runtime = PAID_LEARNING_PILOT_MAX_RUNTIME_MINUTES[context.world_size]
        if not isinstance(max_runtime, int) or isinstance(max_runtime, bool) or not (
            30 <= max_runtime <= max_allowed_runtime
        ):
            raise ValueError("learning-pilot runtime cap is invalid")
        hourly_rate = authorization.get("hourly_rate_usd")
        if not isinstance(hourly_rate, (int, float)) or isinstance(hourly_rate, bool):
            raise ValueError("learning-pilot hourly rate is invalid")
        if not 0 < float(hourly_rate) <= 60:
            raise ValueError("learning-pilot hourly rate is unsafe")
        if not str(authorization.get("provider", "")).strip():
            raise ValueError("learning-pilot provider is missing")
        if not str(authorization.get("instance_type", "")).strip():
            raise ValueError("learning-pilot instance type is missing")
        return authorization

    if authorization.get("full_4b_training") is not True:
        raise ValueError("production authorization must explicitly approve full 4B training")
    if context.world_size != PAID_4B_PRODUCTION_WORLD_SIZE:
        raise ValueError("production authorization requires exactly eight GPUs")
    if strategy != "fsdp_full_shard":
        raise ValueError("production authorization requires FSDP full sharding")
    _validate_production_cuda_runtime(context)
    if (
        not isinstance(approved_step, int)
        or isinstance(approved_step, bool)
        or approved_step != steps
        or not 1 <= approved_step <= PAID_4B_PRODUCTION_MAX_STEPS
    ):
        raise ValueError("production authorization step boundary is invalid")
    if (
        approved_step != PAID_4B_PRODUCTION_MAX_STEPS
        and approved_step % PAID_4B_PRODUCTION_CHECKPOINT_STEPS
    ):
        raise ValueError("production authorization must stop at a checkpoint boundary")
    if (
        not isinstance(global_tokens_per_step, int)
        or isinstance(global_tokens_per_step, bool)
        or global_tokens_per_step <= 0
    ):
        raise ValueError("production authorization requires global token geometry")
    expected_target_tokens = min(
        approved_step * global_tokens_per_step,
        PAID_4B_PRODUCTION_TOKEN_CAP,
    )
    approved_tokens = authorization.get("approved_token_cap")
    if (
        target_tokens != expected_target_tokens
        or approved_tokens != expected_target_tokens
    ):
        raise ValueError("production authorization token boundary changed")
    if authorization.get("total_token_cap") != PAID_4B_PRODUCTION_TOKEN_CAP:
        raise ValueError("production authorization total token cap changed")

    approved_from_step = authorization.get("approved_from_step")
    if (
        not isinstance(approved_from_step, int)
        or isinstance(approved_from_step, bool)
        or not 0 <= approved_from_step < approved_step
    ):
        raise ValueError("production authorization start step is invalid")
    if (
        approved_from_step
        and approved_from_step % PAID_4B_PRODUCTION_CHECKPOINT_STEPS
    ):
        raise ValueError("production authorization must start at a checkpoint boundary")
    expected_from_tokens = min(
        approved_from_step * global_tokens_per_step,
        PAID_4B_PRODUCTION_TOKEN_CAP,
    )
    if authorization.get("approved_from_tokens") != expected_from_tokens:
        raise ValueError("production authorization start token cursor changed")

    segment_index = authorization.get("segment_index")
    segment_count = authorization.get("segment_count")
    if (
        not isinstance(segment_index, int)
        or isinstance(segment_index, bool)
        or not isinstance(segment_count, int)
        or isinstance(segment_count, bool)
        or not 1 <= segment_index <= segment_count <= 10
    ):
        raise ValueError("production authorization segment identity is invalid")
    if authorization.get("owner_acknowledgement") != (
        PAID_4B_PRODUCTION_OWNER_ACKNOWLEDGEMENT
    ):
        raise ValueError("production authorization acknowledgement changed")
    for field in ("production_contract_sha256", "production_plan_sha256"):
        digest = authorization.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"production authorization {field} is invalid")

    max_runtime = authorization.get("max_runtime_minutes")
    if not isinstance(max_runtime, int) or isinstance(max_runtime, bool) or not (
        30 <= max_runtime <= PAID_4B_PRODUCTION_MAX_RUNTIME_MINUTES
    ):
        raise ValueError("production authorization runtime cap is invalid")
    hourly_rate = authorization.get("hourly_rate_usd")
    if not isinstance(hourly_rate, (int, float)) or isinstance(hourly_rate, bool):
        raise ValueError("production authorization hourly rate is invalid")
    if not 0 < float(hourly_rate) <= 60:
        raise ValueError("production authorization hourly rate is unsafe")
    max_cost = authorization.get("max_cost_usd")
    if not isinstance(max_cost, (int, float)) or isinstance(max_cost, bool):
        raise ValueError("production authorization cost cap is invalid")
    if not 0 < float(max_cost) <= PAID_4B_PRODUCTION_MAX_COST_USD:
        raise ValueError("production authorization cost cap is unsafe")
    if float(hourly_rate) * max_runtime / 60 > float(max_cost):
        raise ValueError("production authorization runtime exceeds its cost cap")
    if not str(authorization.get("provider", "")).strip():
        raise ValueError("production authorization provider is missing")
    if not str(authorization.get("instance_type", "")).strip():
        raise ValueError("production authorization instance type is missing")
    return authorization


def _validate_execution_scope(
    *,
    authorization_path: Path | None,
    config: LumiConfig,
    context: DistributedContext,
    run_fingerprint: str,
    source_commit: str | None,
    strategy: str,
    steps: int,
    target_tokens: int,
    global_tokens_per_step: int | None = None,
) -> dict[str, Any]:
    if authorization_path is not None:
        if context.device.type != "cuda" or context.backend != "nccl":
            raise ValueError("paid distributed authorization requires CUDA/NCCL")
        return _validate_paid_authorization(
            authorization_path,
            run_fingerprint=run_fingerprint,
            source_commit=source_commit,
            context=context,
            strategy=strategy,
            steps=steps,
            target_tokens=target_tokens,
            global_tokens_per_step=global_tokens_per_step,
        )

    parameters = parameter_breakdown(config).total
    if context.device.type != "cpu" or context.backend != "gloo":
        raise PermissionError("CUDA training requires a hash-bound owner authorization")
    if context.world_size > LOCAL_PREFLIGHT_MAX_WORLD_SIZE:
        raise PermissionError("unauthorized local preflight world size is too large")
    if parameters > LOCAL_PREFLIGHT_MAX_PARAMETERS:
        raise PermissionError("unauthorized local preflight model is too large")
    if steps > LOCAL_PREFLIGHT_MAX_STEPS or target_tokens > LOCAL_PREFLIGHT_MAX_TOKENS:
        raise PermissionError("unauthorized local preflight budget is too large")
    return {
        "full_4b_training": False,
        "mode": "local_cpu_preflight",
        "paid_gpu": False,
        "parameter_cap": LOCAL_PREFLIGHT_MAX_PARAMETERS,
        "step_cap": LOCAL_PREFLIGHT_MAX_STEPS,
        "token_cap": LOCAL_PREFLIGHT_MAX_TOKENS,
        "world_size_cap": LOCAL_PREFLIGHT_MAX_WORLD_SIZE,
    }


def _validate_authorized_resume_boundary(
    authorization: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    global_tokens_per_step: int,
) -> None:
    """Bind a paid production segment to its imported or recovered checkpoint."""

    if authorization.get("run_kind") != PAID_4B_PRODUCTION_RUN_KIND:
        return
    step = state.get("step")
    tokens = state.get("tokens_seen")
    if (
        not isinstance(step, int)
        or isinstance(step, bool)
        or not isinstance(tokens, int)
        or isinstance(tokens, bool)
    ):
        raise ValueError("production resume state has invalid cursors")
    if not authorization["approved_from_step"] <= step <= authorization[
        "approved_through_step"
    ]:
        raise ValueError("production resume step is outside the authorized segment")
    if not authorization["approved_from_tokens"] <= tokens <= authorization[
        "approved_token_cap"
    ]:
        raise ValueError("production resume tokens are outside the authorized segment")
    expected_tokens = min(
        step * global_tokens_per_step,
        PAID_4B_PRODUCTION_TOKEN_CAP,
    )
    if tokens != expected_tokens:
        raise ValueError("production resume token cursor does not match its step")


def _reduce_mean(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def _reduce_sum(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _reduce_max(value: float | int, context: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _collective_validate_packed_manifest(
    manifest_path: Path,
    context: DistributedContext,
    *,
    final_corpus_root: Path | None = None,
    final_corpus_sequence_length: int = 4096,
    final_corpus_vocab_size: int = 65_536,
    final_corpus_doc_token_id: int = 3,
) -> tuple[dict[str, Any], str]:
    """Validate local corpus visibility and hash each node's copy once."""

    manifest: dict[str, Any] | None = None
    local_result: dict[str, Any]
    try:
        unvalidated = json.loads(manifest_path.read_text(encoding="utf-8"))
        if unvalidated.get("gate") == FINAL_CORPUS_SHARD_MANIFEST_GATE:
            if final_corpus_root is None:
                raise ValueError("the final corpus manifest requires --final-corpus-root")
            manifest = validate_final_corpus_training_manifest(
                manifest_path,
                shard_root=final_corpus_root,
                sequence_length=final_corpus_sequence_length,
                vocab_size=final_corpus_vocab_size,
                doc_token_id=final_corpus_doc_token_id,
                verify_hashes=context.local_rank == 0,
            )
        else:
            if final_corpus_root is not None:
                raise ValueError("--final-corpus-root requires a sealed final corpus manifest")
            manifest = validate_packed_manifest(
                manifest_path,
                verify_hashes=context.local_rank == 0,
            )
        local_result = {
            "build_fingerprint": manifest["build_fingerprint"],
            "error": None,
            "manifest_sha256": _sha256_file(manifest_path),
            "rank": context.rank,
        }
    except Exception as error:  # noqa: BLE001 - report every rank before failing
        local_result = {
            "error": f"{type(error).__name__}: {error}",
            "rank": context.rank,
        }
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_result)
    failures = [item for item in gathered if item is not None and item.get("error")]
    if failures:
        details = "; ".join(
            f"rank {item['rank']}: {item['error']}" for item in failures
        )
        raise RuntimeError(f"distributed packed-data validation failed: {details}")
    identities = {
        (item["manifest_sha256"], item["build_fingerprint"])
        for item in gathered
        if item is not None
    }
    if len(identities) != 1:
        raise RuntimeError("distributed ranks do not share one packed-data manifest")
    assert manifest is not None
    manifest_sha256, _ = next(iter(identities))
    return manifest, manifest_sha256


def _reconcile_checkpoint_metric(
    metrics_path: Path,
    state: dict[str, Any],
) -> None:
    """Recover the metric committed inside a newer checkpoint after a crash."""

    checkpoint_metric = state.pop("checkpoint_metric", None)
    if checkpoint_metric is None:
        return
    committed_step = int(state["step"])
    if int(checkpoint_metric.get("step", -1)) != committed_step:
        raise ValueError("checkpoint metric does not match its committed step")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(record["step"]) for record in records]
    if steps != sorted(set(steps)):
        raise ValueError("distributed metrics are duplicated or out of order")
    if committed_step in steps:
        existing = records[steps.index(committed_step)]
        stable_fields = set(checkpoint_metric) - {"checkpoint_seconds"}
        if any(existing.get(field) != checkpoint_metric[field] for field in stable_fields):
            raise ValueError("checkpoint metric conflicts with the durable metrics log")
        return
    expected_previous = committed_step - 1
    if (steps[-1] if steps else 0) != expected_previous:
        raise ValueError("distributed metrics have a gap before the checkpoint")
    recovered = {
        **checkpoint_metric,
        "checkpoint_seconds": None,
        "recovered_from_checkpoint": True,
    }
    _append_metric(metrics_path, recovered)


def _setup_output(
    *,
    context: DistributedContext,
    output: Path,
    run_fingerprint: str,
    run_specification: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    state_path = output / DISTRIBUTED_TRAINING_STATE_FILENAME
    metrics_path = output / DISTRIBUTED_METRICS_FILENAME

    def setup() -> dict[str, Any]:
        if output.exists() and not output.is_dir():
            raise NotADirectoryError(output)
        output.mkdir(parents=True, exist_ok=True)
        if state_path.exists():
            if not resume:
                raise FileExistsError(f"distributed state exists; pass --resume: {state_path}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("run_fingerprint") != run_fingerprint:
                raise ValueError("distributed state does not match the requested run")
        else:
            if any(output.iterdir()):
                raise FileExistsError(f"distributed output is not empty: {output}")
            state = {
                "best_loss": None,
                "created_at": _utc_now(),
                "last_loss": None,
                "latest_checkpoint": None,
                "run_fingerprint": run_fingerprint,
                "run_specification": run_specification,
                "schema_version": DISTRIBUTED_TRAINING_SCHEMA_VERSION,
                "sequence_cursor": 0,
                "status": "active",
                "step": 0,
                "tokens_seen": 0,
                "updated_at": _utc_now(),
            }
            _write_json_atomic(state_path, state)

        recovered = _discover_latest_checkpoint(output, run_fingerprint=run_fingerprint)
        if recovered is not None:
            checkpoint, checkpoint_state = recovered
            if checkpoint_state["step"] >= state["step"]:
                state = {
                    **state,
                    **checkpoint_state,
                    "latest_checkpoint": checkpoint.name,
                    "status": "active",
                }
        _trim_metrics(metrics_path, int(state["step"]))
        _reconcile_checkpoint_metric(metrics_path, state)
        _write_json_atomic(state_path, state)
        return state

    return _rank_zero_object(context, setup)


def train_distributed_packed(
    *,
    context: DistributedContext,
    manifest_path: str | Path,
    config_path: str | Path,
    output: str | Path,
    steps: int,
    strategy: str,
    per_rank_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    checkpoint_steps: int = 10,
    keep_checkpoints: int | None = None,
    seed: int = 7,
    resume: bool = False,
    gradient_checkpointing: bool = False,
    lr_schedule: str = "constant",
    warmup_steps: int = 0,
    schedule_steps: int | None = None,
    min_learning_rate_ratio: float = 0.1,
    attention_implementation: str | None = None,
    document_isolation: bool = False,
    optimizer_profile: str = "legacy",
    precision: str = "fp32",
    gradient_sync_mode: str = "every_microbatch",
    torch_compile: bool = False,
    source_commit: str | None = None,
    authorization_path: str | Path | None = None,
    final_corpus_root: str | Path | None = None,
    final_corpus_sequence_length: int = 4096,
    final_corpus_vocab_size: int = 65_536,
    final_corpus_doc_token_id: int = 3,
) -> dict[str, Any]:
    """Train Lumi with DDP or FSDP and exact global-cursor resume."""

    if not dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    if (context.rank, context.world_size) != (dist.get_rank(), dist.get_world_size()):
        raise RuntimeError("distributed context does not match the active process group")
    if strategy not in DISTRIBUTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(DISTRIBUTED_STRATEGIES)}")
    if precision not in DISTRIBUTED_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(DISTRIBUTED_PRECISIONS)}")
    if gradient_sync_mode not in GRADIENT_SYNC_MODES:
        raise ValueError(f"gradient sync mode must be one of {sorted(GRADIENT_SYNC_MODES)}")
    if steps < 1 or per_rank_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("steps and batch sizes must be positive")
    if checkpoint_steps < 1:
        raise ValueError("checkpoint steps must be positive")
    if keep_checkpoints is not None and keep_checkpoints < 1:
        raise ValueError("keep checkpoints must be positive")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("optimizer hyperparameters are invalid")
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"learning-rate schedule must be one of {sorted(LR_SCHEDULES)}")
    if optimizer_profile not in OPTIMIZER_PROFILES:
        raise ValueError(f"optimizer profile must be one of {sorted(OPTIMIZER_PROFILES)}")
    if attention_implementation is not None and (
        attention_implementation not in ATTENTION_IMPLEMENTATIONS
    ):
        raise ValueError("unsupported attention implementation")
    if context.device.type == "cuda" and precision != "bf16":
        raise ValueError("production CUDA training is pinned to bf16 in v0.1")
    if precision == "bf16" and context.device.type == "cuda" and (
        not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bf16")
    if attention_implementation == "flash_attention_2" and context.device.type != "cuda":
        raise ValueError("flash_attention_2 requires CUDA")
    if document_isolation and context.device.type == "cuda" and (
        attention_implementation != "flash_attention_2"
    ):
        raise ValueError("CUDA document isolation requires flash_attention_2")
    if lr_schedule == "cosine":
        if schedule_steps is None or schedule_steps < steps or warmup_steps >= schedule_steps:
            raise ValueError("cosine schedule does not cover the requested steps")
    elif schedule_steps is not None:
        raise ValueError("schedule steps are only valid for cosine learning rate")

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output = Path(output)
    authorization = None if authorization_path is None else Path(authorization_path)
    identity_checkpoint_steps = None
    if authorization is not None:
        unvalidated_authorization = json.loads(
            authorization.read_text(encoding="utf-8")
        )
        if unvalidated_authorization.get("run_kind") == PAID_4B_PRODUCTION_RUN_KIND:
            identity_checkpoint_steps = checkpoint_steps
    final_root = None if final_corpus_root is None else Path(final_corpus_root)
    manifest, manifest_sha256 = _collective_validate_packed_manifest(
        manifest_path,
        context,
        final_corpus_root=final_root,
        final_corpus_sequence_length=final_corpus_sequence_length,
        final_corpus_vocab_size=final_corpus_vocab_size,
        final_corpus_doc_token_id=final_corpus_doc_token_id,
    )
    config_bytes = config_path.read_bytes()
    config = LumiConfig(**json.loads(config_bytes))
    config._attn_implementation = attention_implementation
    sequence_length = int(manifest["packing"]["sequence_length"])
    if config.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("model and distributed dataset vocabulary sizes differ")
    if config.doc_token_id != manifest["tokenizer"]["doc_token_id"]:
        raise ValueError("model and distributed dataset document-token IDs differ")
    if sequence_length > config.max_position_embeddings:
        raise ValueError("distributed sequence length exceeds the model maximum")
    final_corpus = manifest.get("gate") == "lumi_4b_final_corpus_adapter_v0.1"

    run_identity = build_distributed_run_identity(
        config_bytes=config_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        backend=context.backend,
        world_size=context.world_size,
        strategy=strategy,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        keep_checkpoints=keep_checkpoints,
        checkpoint_steps=identity_checkpoint_steps,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        schedule_steps=schedule_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        attention_implementation=attention_implementation,
        document_isolation=document_isolation,
        optimizer_profile=optimizer_profile,
        precision=precision,
        gradient_sync_mode=gradient_sync_mode,
        torch_compile=torch_compile,
        source_commit=source_commit,
    )
    run_fingerprint = _fingerprint(run_identity)
    global_tokens_per_step = (
        context.world_size
        * per_rank_batch_size
        * gradient_accumulation_steps
        * sequence_length
    )
    target_tokens = steps * global_tokens_per_step
    if final_corpus:
        target_tokens = min(target_tokens, int(manifest["statistics"]["tokens"]))
    scope = _validate_execution_scope(
        authorization_path=authorization,
        config=config,
        context=context,
        run_fingerprint=run_fingerprint,
        source_commit=source_commit,
        strategy=strategy,
        steps=steps,
        target_tokens=target_tokens,
        global_tokens_per_step=global_tokens_per_step,
    )
    run_specification = {
        **run_identity,
        "authorization": scope,
        "distributed": {
            **run_identity["distributed"],
            "global_tokens_per_step": global_tokens_per_step,
        },
        "locations": {
            "config_path": str(config_path.resolve()),
            "final_corpus_root": None if final_root is None else str(final_root.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        },
    }
    state = _setup_output(
        context=context,
        output=output,
        run_fingerprint=run_fingerprint,
        run_specification=run_specification,
        resume=resume,
    )
    _validate_authorized_resume_boundary(
        scope,
        state,
        global_tokens_per_step=global_tokens_per_step,
    )

    torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LumiForCausalLM(config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if torch_compile:
        # PyTorch recommends compiling the inner module before DDP/FSDP. FSDP
        # compilation also requires use_orig_params=True, which _wrap_model pins.
        model.compile(dynamic=False, fullgraph=False)
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

    if final_corpus:
        assert final_root is not None
        dataset: torch.utils.data.Dataset[torch.Tensor] = FinalCorpusTokenDataset(
            manifest_path,
            final_root,
            sequence_length=final_corpus_sequence_length,
            vocab_size=final_corpus_vocab_size,
            doc_token_id=final_corpus_doc_token_id,
            verify_hashes=False,
        )
    else:
        dataset = PackedTokenDataset(manifest_path, verify_hashes=False)
    if len(dataset) < context.world_size * per_rank_batch_size:
        raise ValueError("distributed dataset cannot fill one global microbatch")
    nominal_sequences_per_step = (
        context.world_size * per_rank_batch_size * gradient_accumulation_steps
    )
    if final_corpus:
        maximum_steps = (len(dataset) + nominal_sequences_per_step - 1) // (
            nominal_sequences_per_step
        )
        if steps > maximum_steps:
            raise ValueError("requested steps exceed the exact final-corpus token boundary")
    distributed_model.train()
    sequence_planner = EpochPermutationPlanner(dataset_length=len(dataset), seed=seed)
    metrics_path = output / DISTRIBUTED_METRICS_FILENAME
    state_path = output / DISTRIBUTED_TRAINING_STATE_FILENAME

    while state["step"] < steps:
        step_learning_rate = scheduled_learning_rate(
            base_learning_rate=learning_rate,
            step=state["step"] + 1,
            schedule=lr_schedule,
            warmup_steps=warmup_steps,
            schedule_steps=schedule_steps,
            min_learning_rate_ratio=min_learning_rate_ratio,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = step_learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_local_loss = 0.0
        step_cursor = int(state["sequence_cursor"])
        if final_corpus and step_cursor >= len(dataset):
            raise ValueError("final-corpus cursor reached the exact token boundary early")
        active_sequences_this_step = nominal_sequences_per_step
        accumulations_this_step = gradient_accumulation_steps
        if final_corpus:
            active_sequences_this_step = min(
                len(dataset) - step_cursor,
                nominal_sequences_per_step,
            )
            accumulations_this_step = (
                active_sequences_this_step
                + context.world_size * per_rank_batch_size
                - 1
            ) // (context.world_size * per_rank_batch_size)
        _synchronize(context.device)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()

        for accumulation_index in range(accumulations_this_step):
            rank_is_active = True
            active_local_sequences = per_rank_batch_size
            if final_corpus:
                microbatch = final_corpus_microbatch_plan(
                    active_sequences_this_step=active_sequences_this_step,
                    accumulation_index=accumulation_index,
                    per_rank_batch_size=per_rank_batch_size,
                    rank=context.rank,
                    world_size=context.world_size,
                )
                global_indices, step_cursor = sequence_planner.global_indices(
                    cursor=step_cursor,
                    count=microbatch.active_global_sequences,
                )
                active_local_sequences = microbatch.active_local_sequences
                rank_is_active = microbatch.rank_is_active
                if rank_is_active:
                    start = microbatch.global_offset
                    indices = global_indices[start : start + active_local_sequences]
                else:
                    # Inactive ranks must still enter FSDP collectives. Their
                    # zero loss scale ensures this dummy row contributes no gradient.
                    indices = [global_indices[0]]
                loss_scale = (
                    context.world_size
                    * active_local_sequences
                    / active_sequences_this_step
                    if rank_is_active
                    else 0.0
                )
            else:
                indices, step_cursor = sequence_planner.rank_batch(
                    cursor=step_cursor,
                    per_rank_batch_size=per_rank_batch_size,
                    rank=context.rank,
                    world_size=context.world_size,
                )
                loss_scale = 1.0 / gradient_accumulation_steps
            cpu_batch = torch.stack([dataset[index] for index in indices])
            if context.device.type == "cuda":
                cpu_batch = cpu_batch.pin_memory()
            batch = cpu_batch.to(context.device, non_blocking=context.device.type == "cuda")
            model_kwargs: dict[str, Any] = {}
            if document_isolation:
                metadata = build_packed_document_metadata(
                    cpu_batch,
                    doc_token_id=config.doc_token_id,
                )
                cumulative = metadata.cu_seq_lens.to(context.device, non_blocking=True)
                model_kwargs = {
                    "cu_seq_lens_k": cumulative,
                    "cu_seq_lens_q": cumulative,
                    "document_isolation": True,
                    "max_length_k": metadata.max_length,
                    "max_length_q": metadata.max_length,
                    "position_ids": metadata.position_ids.to(
                        context.device,
                        non_blocking=True,
                    ),
                }
            should_defer_sync = (
                gradient_sync_mode == "last_microbatch"
                and accumulation_index < accumulations_this_step - 1
            )
            synchronization = distributed_model.no_sync() if should_defer_sync else nullcontext()
            with synchronization:
                with _autocast_context(context.device, precision):
                    loss = distributed_model(input_ids=batch, labels=batch, **model_kwargs).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite distributed loss at step {state['step'] + 1}")
                (loss * loss_scale).backward()
            if rank_is_active:
                accumulated_local_loss += float(loss.detach()) * (
                    active_local_sequences if final_corpus else 1
                )

        if strategy == "fsdp_full_shard":
            gradient_norm = distributed_model.clip_grad_norm_(max_grad_norm)
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                distributed_model.parameters(),
                max_grad_norm,
            )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite distributed gradient at step {state['step'] + 1}")
        optimizer.step()
        _synchronize(context.device)
        elapsed = _reduce_max(max(time.perf_counter() - started, 1e-9), context)
        if final_corpus:
            global_loss = _reduce_sum(accumulated_local_loss, context) / (
                active_sequences_this_step
            )
        else:
            global_loss = _reduce_mean(
                accumulated_local_loss / gradient_accumulation_steps,
                context,
            )
        global_gradient_norm = _reduce_max(float(gradient_norm), context)
        actual_global_tokens = active_sequences_this_step * sequence_length

        state["step"] += 1
        state["sequence_cursor"] = step_cursor
        state["tokens_seen"] = step_cursor * sequence_length
        state["last_loss"] = global_loss
        state["best_loss"] = (
            global_loss if state["best_loss"] is None else min(state["best_loss"], global_loss)
        )
        state["updated_at"] = _utc_now()
        metric = {
            "backend": context.backend,
            "device": context.device.type,
            "elapsed_seconds": elapsed,
            "global_tokens_per_step": actual_global_tokens,
            "gradient_norm": global_gradient_norm,
            "learning_rate": step_learning_rate,
            "loss": global_loss,
            "peak_memory_allocated_bytes": (
                int(_reduce_max(torch.cuda.max_memory_allocated(context.device), context))
                if context.device.type == "cuda"
                else None
            ),
            "peak_memory_reserved_bytes": (
                int(_reduce_max(torch.cuda.max_memory_reserved(context.device), context))
                if context.device.type == "cuda"
                else None
            ),
            "step": state["step"],
            "strategy": strategy,
            "tokens_per_second": actual_global_tokens / elapsed,
            "tokens_seen": state["tokens_seen"],
            "world_size": context.world_size,
        }
        if context.is_rank_zero:
            print(
                f"step={state['step']:06d} loss={global_loss:.6f} "
                f"global_tokens_per_second={metric['tokens_per_second']:.1f}",
                flush=True,
            )

        should_checkpoint = (
            state["step"] % checkpoint_steps == 0
            or state["step"] == steps
            or (final_corpus and step_cursor == len(dataset))
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
        state["status"] = "complete"
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)
    dist.barrier()
    result = [state if context.is_rank_zero else None]
    dist.broadcast_object_list(result, src=0)
    return result[0]
