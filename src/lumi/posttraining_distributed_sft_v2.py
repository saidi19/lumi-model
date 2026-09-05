"""Hardened distributed assistant-and-tool-only SFT with exact resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
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

from .configuration_lumi import LumiConfig
from .distributed_training import (
    DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
    DISTRIBUTED_METRICS_FILENAME,
    DISTRIBUTED_STRATEGIES,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
    GRADIENT_SYNC_MODES,
    DistributedContext,
    EpochPermutationPlanner,
    _autocast_context,
    _rank_zero_object,
    _reduce_max,
    _reduce_sum,
    _restore_runtime_rng_state,
    _runtime_rng_state,
    _validate_execution_scope,
    _wrap_model,
)
from .modeling_lumi import LumiForCausalLM
from .pilot_training import (
    ATTENTION_IMPLEMENTATIONS,
    LR_SCHEDULES,
    OPTIMIZER_PROFILES,
    _create_optimizer,
    _fingerprint,
    _synchronize,
    _utc_now,
    scheduled_learning_rate,
)
from .posttraining_sft_runtime_v4 import (
    EncodedSFTDataset,
    _append_jsonl,
    _ensure_private_directory,
    _fsync_tree,
    _harden_private_tree,
    _require_private_regular,
    _trim_metrics,
    _verify_checksum_manifest,
    _write_checksum_manifest,
    _write_json_atomic,
    _write_json_exclusive,
    collate_sft_tensors,
)

SFT_DISTRIBUTED_STATE_SCHEMA_VERSION = "lumi.sft-distributed-state.v0.2"
SFT_DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION = DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION
SFT_DISTRIBUTED_OBJECTIVE = "assistant_and_tool_only_cross_entropy_v0.1"
SFT_DISTRIBUTED_STARTED_FILENAME = "sft-distributed-started.json"
SFT_DISTRIBUTED_RESUME_STARTED_FILENAME = "sft-distributed-resume-started.json"
SFT_DISTRIBUTED_STARTED_SCHEMA_VERSION = "lumi.sft-distributed-started.v0.2"
SFT_DISTRIBUTED_RESUME_STARTED_SCHEMA_VERSION = (
    "lumi.sft-distributed-resume-started.v0.2"
)

_FINAL_MODEL_DIGEST_FILENAME = "final-model-digest.json"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-[0-9]{8}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_READ_AUDIT_FIELDS = (
    "binary_bytes_read",
    "encoded_index_row_reads",
    "file_opens",
    "fully_hashed_shards",
    "index_bytes_read",
    "metadata_inventory_verified_before_selected_content_read",
    "opened_shards",
    "records_loaded",
    "unbuffered_selected_content_reads",
)
_TOP_STATE_REQUIRED_FIELDS = frozenset(
    {
        "best_loss",
        "created_at",
        "last_loss",
        "latest_checkpoint",
        "records_seen",
        "run_fingerprint",
        "run_specification",
        "schema_version",
        "sequence_cursor",
        "status",
        "step",
        "supervised_tokens_seen",
        "tokens_seen",
        "updated_at",
    }
)
_TOP_STATE_OPTIONAL_FIELDS = frozenset({"last_checkpoint_seconds"})
_CHECKPOINT_ONLY_FIELDS = frozenset(
    {"checkpoint_metric", "checkpoint_schema_version"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_audit_from_verification(
    verification: Mapping[str, Any],
    *,
    record_limit: int,
) -> dict[str, Any]:
    try:
        audit = {field: verification[field] for field in _READ_AUDIT_FIELDS}
    except KeyError as error:
        raise ValueError("distributed SFT read audit is incomplete") from error
    integer_fields = {
        "binary_bytes_read",
        "encoded_index_row_reads",
        "file_opens",
        "fully_hashed_shards",
        "index_bytes_read",
        "opened_shards",
        "records_loaded",
    }
    if any(
        type(audit[field]) is not int or audit[field] < 0
        for field in integer_fields
    ):
        raise ValueError("distributed SFT read audit counts are invalid")
    if (
        audit["fully_hashed_shards"] != 0
        or audit["records_loaded"] != record_limit
        or audit["metadata_inventory_verified_before_selected_content_read"]
        is not True
        or audit["unbuffered_selected_content_reads"] is not True
        or verification.get("all_inventory_file_hashes_verified") is not False
        or verification.get("full_content_hashes_verified") is not False
    ):
        raise ValueError("distributed SFT bounded-read contract changed")
    return audit


def _data_identity(
    *,
    shard_manifest: Mapping[str, Any],
    shard_verification: Mapping[str, Any],
    record_limit: int,
    world_size: int,
) -> dict[str, Any]:
    if shard_verification.get("profile") != "token_consistent_v0.2":
        raise ValueError("distributed SFT requires token-consistent v0.2 shards")
    if shard_verification.get("manifest_sha256") != shard_manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("distributed SFT manifest identities disagree")
    read_audit = _read_audit_from_verification(
        shard_verification,
        record_limit=record_limit,
    )
    identity = {
        "inventory_sha256": _require_sha256(
            shard_verification.get("inventory_sha256"),
            label="distributed SFT inventory identity",
        ),
        "manifest_file_sha256": _require_sha256(
            shard_verification.get("manifest_file_sha256"),
            label="distributed SFT manifest-file identity",
        ),
        "manifest_sha256": _require_sha256(
            shard_manifest.get("manifest_sha256"),
            label="distributed SFT manifest identity",
        ),
        "metadata_inventory_sha256": _require_sha256(
            shard_verification.get("metadata_inventory_sha256"),
            label="distributed SFT metadata-inventory identity",
        ),
        "read_audit": read_audit,
        "read_audit_accounting": {
            "all_rank_read_audits_equal": True,
            "rank_local_loads": world_size,
        },
        "record_limit": record_limit,
        "seal_sha256": _require_sha256(
            shard_manifest.get("seal_sha256"),
            label="distributed SFT seal identity",
        ),
        "selection_scope": "contiguous_prefix_v0.1",
        "sequence_order": "cached_epoch_randperm_v0.1",
        "selected_prefix_sha256": _require_sha256(
            shard_verification.get("selected_prefix_sha256"),
            label="distributed SFT selected-prefix identity",
        ),
        "split": "train",
        "verification_profile": "token_consistent_v0.2",
    }
    return identity


def build_distributed_sft_run_identity(
    *,
    source_config_bytes: bytes,
    source_checkpoint_manifest_sha256: str,
    shard_manifest: Mapping[str, Any],
    shard_verification: Mapping[str, Any],
    encoded_shards_record_limit: int,
    backend: str,
    world_size: int,
    strategy: str,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    checkpoint_steps: int,
    keep_checkpoints: int | None,
    target_steps: int,
    seed: int,
    lr_schedule: str,
    warmup_steps: int,
    min_learning_rate_ratio: float,
    attention_implementation: str,
    optimizer_profile: str,
    precision: str,
    gradient_sync_mode: str,
    pad_to_multiple_of: int,
    source_commit: str | None,
) -> dict[str, Any]:
    """Build the exact identity that every participating rank must share."""

    identity: dict[str, Any] = {
        "data": _data_identity(
            shard_manifest=shard_manifest,
            shard_verification=shard_verification,
            record_limit=encoded_shards_record_limit,
            world_size=world_size,
        ),
        "distributed": {
            "backend": backend,
            "checkpoint_format": "dcp-model_rank-local-optimizer-v0.1",
            "checkpoint_steps": checkpoint_steps,
            "gradient_sync_mode": gradient_sync_mode,
            "keep_checkpoints": keep_checkpoints,
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
            "manifest_sha256": _require_sha256(
                source_checkpoint_manifest_sha256,
                label="distributed SFT source manifest",
            ),
        },
        "torch_version": torch.__version__,
        "training": {
            "attention_implementation": attention_implementation,
            "context_window": int(shard_manifest["context_window"]),
            "loss_weighting": "global_shifted_supervised_token_mean_v0.1",
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding": "rank_local_right_padding_labels_minus_100_v0.1",
        },
    }
    if source_commit is not None:
        normalized = source_commit.strip().lower()
        if len(normalized) != 40 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
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
    keep_checkpoints: int | None,
    precision: str,
    gradient_sync_mode: str,
    lr_schedule: str,
    warmup_steps: int,
    min_learning_rate_ratio: float,
    attention_implementation: str,
    optimizer_profile: str,
    pad_to_multiple_of: int,
    record_limit: int,
    seed: int,
) -> None:
    if not dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    if (context.rank, context.world_size) != (dist.get_rank(), dist.get_world_size()):
        raise RuntimeError("distributed context does not match the process group")
    integer_values = (
        target_steps,
        stop_after_step,
        per_rank_batch_size,
        gradient_accumulation_steps,
        checkpoint_steps,
        pad_to_multiple_of,
        record_limit,
        seed,
    )
    if any(type(value) is not int for value in integer_values):
        raise ValueError("distributed SFT integer settings changed type")
    if strategy not in DISTRIBUTED_STRATEGIES:
        raise ValueError("unsupported distributed SFT strategy")
    if gradient_sync_mode not in GRADIENT_SYNC_MODES:
        raise ValueError("unsupported distributed SFT gradient-sync mode")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported distributed SFT precision")
    if target_steps < 1 or not 1 <= stop_after_step <= target_steps:
        raise ValueError("distributed SFT stop step must fit the target")
    if min(
        per_rank_batch_size,
        gradient_accumulation_steps,
        checkpoint_steps,
        pad_to_multiple_of,
        record_limit,
    ) < 1:
        raise ValueError("distributed SFT batch and checkpoint settings must be positive")
    if record_limit > 12:
        raise ValueError("distributed SFT record limit exceeds the bounded loader")
    if keep_checkpoints is not None and (
        type(keep_checkpoints) is not int or keep_checkpoints < 1
    ):
        raise ValueError("distributed SFT retained-checkpoint count is invalid")
    numeric_values = (learning_rate, weight_decay, max_grad_norm)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise ValueError("distributed SFT optimizer settings are not finite")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("distributed SFT optimizer settings are invalid")
    if (
        isinstance(min_learning_rate_ratio, bool)
        or not isinstance(min_learning_rate_ratio, (int, float))
        or not math.isfinite(float(min_learning_rate_ratio))
        or not 0 <= min_learning_rate_ratio <= 1
    ):
        raise ValueError("distributed SFT minimum learning-rate ratio is invalid")
    if lr_schedule not in LR_SCHEDULES or optimizer_profile not in OPTIMIZER_PROFILES:
        raise ValueError("distributed SFT optimizer profile changed")
    if type(warmup_steps) is not int or warmup_steps < 0 or warmup_steps >= target_steps:
        raise ValueError("distributed SFT warmup does not fit the target")
    if attention_implementation not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError("unsupported distributed SFT attention implementation")
    if attention_implementation == "flash_attention_2" and context.device.type != "cuda":
        raise ValueError("FlashAttention distributed SFT requires CUDA")
    if context.device.type == "cuda" and precision != "bf16":
        raise ValueError("CUDA distributed SFT is pinned to BF16")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fresh_marker() -> dict[str, Any]:
    return {
        "schema_version": SFT_DISTRIBUTED_STARTED_SCHEMA_VERSION,
        "status": "started_no_automatic_retry",
    }


def _resume_marker(*, stop_after_step: int, target_steps: int) -> dict[str, Any]:
    return {
        "schema_version": SFT_DISTRIBUTED_RESUME_STARTED_SCHEMA_VERSION,
        "status": "started_no_automatic_retry",
        "stop_after_step": stop_after_step,
        "target_steps": target_steps,
    }


def _validate_output_entry(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("distributed SFT output contains a symlink")
    if path.name in {
        SFT_DISTRIBUTED_STARTED_FILENAME,
        SFT_DISTRIBUTED_RESUME_STARTED_FILENAME,
        DISTRIBUTED_TRAINING_STATE_FILENAME,
        DISTRIBUTED_METRICS_FILENAME,
        _FINAL_MODEL_DIGEST_FILENAME,
    }:
        _require_private_regular(path)
        return
    if _CHECKPOINT_PATTERN.fullmatch(path.name) is not None and stat.S_ISDIR(
        info.st_mode
    ):
        return
    raise ValueError("distributed SFT output inventory changed")


def _mark_invocation(
    *,
    context: DistributedContext,
    output: Path,
    resume: bool,
    stop_after_step: int,
    target_steps: int,
) -> None:
    def mark() -> None:
        _ensure_private_directory(output)
        started_path = output / SFT_DISTRIBUTED_STARTED_FILENAME
        resume_path = output / SFT_DISTRIBUTED_RESUME_STARTED_FILENAME
        if resume:
            for entry in output.iterdir():
                _validate_output_entry(entry)
            _require_private_regular(started_path)
            if started_path.read_bytes() != _pretty_json_bytes(_fresh_marker()):
                raise ValueError("distributed SFT fresh marker changed")
            if not (output / DISTRIBUTED_TRAINING_STATE_FILENAME).is_file():
                raise ValueError("distributed SFT resume state is missing")
            _write_json_exclusive(
                resume_path,
                _resume_marker(
                    stop_after_step=stop_after_step,
                    target_steps=target_steps,
                ),
            )
        else:
            if any(output.iterdir()):
                raise FileExistsError(f"distributed SFT output is not empty: {output}")
            _write_json_exclusive(started_path, _fresh_marker())

    _rank_zero_object(context, mark)
    dist.barrier()


def _collective_load_inputs(
    *,
    context: DistributedContext,
    encoded_shards: Path,
    source_checkpoint: Path,
    source_checkpoint_manifest_sha256: str,
    encoded_shards_inventory_sha256: str,
    encoded_shards_manifest_file_sha256: str,
    encoded_shards_manifest_sha256: str,
    encoded_shards_metadata_inventory_sha256: str,
    encoded_shards_record_limit: int,
    encoded_shards_seal_sha256: str,
) -> tuple[EncodedSFTDataset, bytes, str]:
    dataset: EncodedSFTDataset | None = None
    source_config_bytes: bytes | None = None
    observed_source_manifest: str | None = None
    local: dict[str, Any]
    try:
        observed_source_manifest = _verify_checksum_manifest(
            source_checkpoint,
            expected_manifest_sha256=source_checkpoint_manifest_sha256,
            require_private=True,
        )
        source_config_bytes = (source_checkpoint / "config.json").read_bytes()
        dataset = EncodedSFTDataset(
            encoded_shards,
            split="train",
            expected_inventory_sha256=encoded_shards_inventory_sha256,
            expected_manifest_file_sha256=encoded_shards_manifest_file_sha256,
            expected_manifest_sha256=encoded_shards_manifest_sha256,
            expected_metadata_inventory_sha256=(
                encoded_shards_metadata_inventory_sha256
            ),
            expected_seal_sha256=encoded_shards_seal_sha256,
            record_limit=encoded_shards_record_limit,
        )
        local_data = _data_identity(
            shard_manifest=dataset.manifest,
            shard_verification=dataset.verification,
            record_limit=encoded_shards_record_limit,
            world_size=context.world_size,
        )
        local = {
            "config_sha256": hashlib.sha256(source_config_bytes).hexdigest(),
            "data": local_data,
            "error": None,
            "rank": context.rank,
            "source_manifest_sha256": observed_source_manifest,
        }
    except Exception as error:  # noqa: BLE001 - every rank must reach the collective
        local = {
            "error": f"{type(error).__name__}: {error}",
            "rank": context.rank,
        }

    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    failures = [item for item in gathered if item is not None and item.get("error")]
    if failures:
        details = "; ".join(
            f"rank {item['rank']}: {item['error']}" for item in failures
        )
        raise RuntimeError(f"distributed SFT input validation failed: {details}")
    identities = {
        json.dumps(
            {
                "config_sha256": item["config_sha256"],
                "data": item["data"],
                "source_manifest_sha256": item["source_manifest_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in gathered
        if item is not None
    }
    if len(identities) != 1:
        raise RuntimeError("distributed ranks do not share one SFT input identity")
    if dataset is None or source_config_bytes is None or observed_source_manifest is None:
        raise AssertionError("successful distributed input validation lost local inputs")
    return dataset, source_config_bytes, observed_source_manifest


def _collective_require_run_identity(
    context: DistributedContext,
    run_identity: Mapping[str, Any],
) -> str:
    local = {
        "fingerprint": _fingerprint(run_identity),
        "identity": dict(run_identity),
        "rank": context.rank,
    }
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    fingerprints = {
        item["fingerprint"] for item in gathered if item is not None
    }
    identities = {
        json.dumps(item["identity"], sort_keys=True, separators=(",", ":"))
        for item in gathered
        if item is not None
    }
    if len(fingerprints) != 1 or len(identities) != 1:
        raise RuntimeError("distributed ranks do not share one SFT run identity")
    return next(iter(fingerprints))


def _require_private_checkpoint_tree(checkpoint: Path) -> None:
    for path in checkpoint.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("distributed SFT checkpoint contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
                raise ValueError("distributed SFT checkpoint directory is not private")
        elif stat.S_ISREG(info.st_mode):
            _require_private_regular(path)
        else:
            raise ValueError("distributed SFT checkpoint contains a special file")


def verify_distributed_sft_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Verify one exact-inventory private v2 distributed SFT checkpoint."""

    checkpoint = Path(checkpoint)
    _verify_checksum_manifest(
        checkpoint,
        require_exact_inventory=True,
        require_private=True,
    )
    _require_private_checkpoint_tree(checkpoint)
    state_path = checkpoint / DISTRIBUTED_TRAINING_STATE_FILENAME
    _require_private_regular(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("checkpoint_schema_version")
        != SFT_DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION
        or state.get("schema_version") != SFT_DISTRIBUTED_STATE_SCHEMA_VERSION
        or not isinstance(state.get("checkpoint_metric"), Mapping)
    ):
        raise ValueError("unsupported distributed SFT checkpoint schema")
    expected_ranks = (
        state.get("run_specification", {})
        .get("distributed", {})
        .get("world_size")
    )
    if type(expected_ranks) is not int or expected_ranks < 2:
        raise ValueError("distributed SFT checkpoint world size is invalid")
    for rank in range(expected_ranks):
        _require_private_regular(checkpoint / f"trainer-rank-{rank:05d}.pt")
    return state


def _discover_latest_checkpoint(
    output: Path,
    *,
    run_fingerprint: str,
) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in output.iterdir():
        if _CHECKPOINT_PATTERN.fullmatch(path.name) is None:
            continue
        state = verify_distributed_sft_checkpoint(path)
        if state.get("run_fingerprint") == run_fingerprint:
            step = state.get("step")
            if type(step) is not int or step < 1:
                raise ValueError("distributed SFT checkpoint step is invalid")
            candidates.append((step, path, state))
    if not candidates:
        return None
    _, checkpoint, state = max(candidates, key=lambda item: item[0])
    return checkpoint, state


def _normalize_top_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(state)
    checkpoint_schema = normalized.pop("checkpoint_schema_version", None)
    checkpoint_metric = normalized.pop("checkpoint_metric", None)
    if checkpoint_schema is not None and (
        checkpoint_schema != SFT_DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION
        or not isinstance(checkpoint_metric, Mapping)
    ):
        raise ValueError("distributed SFT recovered checkpoint fields changed")
    if checkpoint_schema is None and checkpoint_metric is not None:
        raise ValueError("distributed SFT checkpoint metric leaked without its schema")
    fields = frozenset(normalized)
    if (
        not _TOP_STATE_REQUIRED_FIELDS.issubset(fields)
        or not fields.issubset(
            _TOP_STATE_REQUIRED_FIELDS | _TOP_STATE_OPTIONAL_FIELDS
        )
    ):
        raise ValueError("distributed SFT top-level state field inventory changed")
    if normalized.get("schema_version") != SFT_DISTRIBUTED_STATE_SCHEMA_VERSION:
        raise ValueError("distributed SFT top-level state schema changed")
    return normalized


def _reconcile_checkpoint_metric(metrics_path: Path, state: Mapping[str, Any]) -> None:
    checkpoint_metric = state.get("checkpoint_metric")
    if checkpoint_metric is None:
        return
    if not isinstance(checkpoint_metric, Mapping):
        raise ValueError("distributed SFT checkpoint metric is invalid")
    committed_step = int(state["step"])
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(record["step"]) for record in records]
    if steps != sorted(set(steps)):
        raise ValueError("distributed SFT metrics are duplicated or out of order")
    if committed_step in steps:
        existing = records[steps.index(committed_step)]
        stable_fields = set(checkpoint_metric) - {"checkpoint_seconds"}
        if any(
            existing.get(field) != checkpoint_metric[field]
            for field in stable_fields
        ):
            raise ValueError("distributed SFT checkpoint metric conflicts with log")
        return
    if (steps[-1] if steps else 0) != committed_step - 1:
        raise ValueError("distributed SFT metrics have a gap before checkpoint")
    _append_jsonl(
        metrics_path,
        {
            **checkpoint_metric,
            "checkpoint_seconds": None,
            "recovered_from_checkpoint": True,
        },
    )


def _setup_state(
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
        for entry in output.iterdir():
            _validate_output_entry(entry)
        if resume:
            _require_private_regular(state_path)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("run_fingerprint") != run_fingerprint
                or state.get("run_specification") != run_specification
            ):
                raise ValueError("distributed SFT state does not match requested run")
        else:
            if {entry.name for entry in output.iterdir()} != {
                SFT_DISTRIBUTED_STARTED_FILENAME
            }:
                raise FileExistsError(
                    f"distributed SFT output is not marker-only: {output}"
                )
            now = _utc_now()
            state = {
                "best_loss": None,
                "created_at": now,
                "last_loss": None,
                "latest_checkpoint": None,
                "records_seen": 0,
                "run_fingerprint": run_fingerprint,
                "run_specification": run_specification,
                "schema_version": SFT_DISTRIBUTED_STATE_SCHEMA_VERSION,
                "sequence_cursor": 0,
                "status": "active",
                "step": 0,
                "supervised_tokens_seen": 0,
                "tokens_seen": 0,
                "updated_at": now,
            }

        recovered = _discover_latest_checkpoint(
            output,
            run_fingerprint=run_fingerprint,
        )
        if recovered is not None:
            checkpoint, checkpoint_state = recovered
            if int(checkpoint_state["step"]) >= int(state["step"]):
                state = {
                    **state,
                    **checkpoint_state,
                    "latest_checkpoint": checkpoint.name,
                    "status": "active",
                }
        _trim_metrics(metrics_path, committed_step=int(state["step"]))
        _reconcile_checkpoint_metric(metrics_path, state)
        state = _normalize_top_state(state)
        _write_json_atomic(state_path, state)
        return state

    state = _rank_zero_object(context, setup)
    dist.barrier()
    return state


def _save_checkpoint(
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
        if os.path.lexists(checkpoint):
            raise FileExistsError(
                f"refusing to overwrite distributed SFT checkpoint: {checkpoint}"
            )
        temporary_name = f".{checkpoint_name}.{uuid.uuid4().hex}.tmp"
        temporary = output / temporary_name
        temporary.mkdir(mode=0o700)
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
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_schema_version": SFT_DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION,
        "latest_checkpoint": checkpoint_name,
        "status": "checkpoint",
        "updated_at": _utc_now(),
    }

    def publish() -> str:
        _write_json_atomic(
            temporary / DISTRIBUTED_TRAINING_STATE_FILENAME,
            checkpoint_state,
        )
        _harden_private_tree(temporary)
        _write_checksum_manifest(temporary)
        _harden_private_tree(temporary)
        _fsync_tree(temporary)
        temporary.replace(checkpoint)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        verify_distributed_sft_checkpoint(checkpoint)
        return checkpoint_name

    _rank_zero_object(context, publish)
    dist.barrier()
    return checkpoint


def _restore_checkpoint(
    *,
    context: DistributedContext,
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    verify_distributed_sft_checkpoint(checkpoint)
    model_state = get_model_state_dict(model)
    dcp.load({"model": model_state}, checkpoint_id=checkpoint / "model")
    incompatible = set_model_state_dict(model, model_state)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"distributed SFT model restore was incompatible: {incompatible}"
        )
    trainer_path = checkpoint / f"trainer-rank-{context.rank:05d}.pt"
    _require_private_regular(trainer_path)
    trainer = torch.load(trainer_path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(trainer["optimizer"])
    _restore_runtime_rng_state(trainer["rng_state"], context.device)
    dist.barrier()


def _prune_checkpoints(
    output: Path,
    *,
    run_fingerprint: str,
    keep_checkpoints: int,
) -> list[str]:
    candidates: list[tuple[int, Path]] = []
    for path in output.iterdir():
        if _CHECKPOINT_PATTERN.fullmatch(path.name) is None:
            continue
        state = verify_distributed_sft_checkpoint(path)
        if state.get("run_fingerprint") == run_fingerprint:
            candidates.append((int(state["step"]), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    removed: list[str] = []
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


def train_distributed_sft(
    *,
    context: DistributedContext,
    encoded_shards: str | Path,
    source_checkpoint: str | Path,
    source_checkpoint_manifest_sha256: str,
    encoded_shards_inventory_sha256: str,
    encoded_shards_manifest_file_sha256: str,
    encoded_shards_manifest_sha256: str,
    encoded_shards_metadata_inventory_sha256: str,
    encoded_shards_record_limit: int,
    encoded_shards_seal_sha256: str,
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
    pad_to_multiple_of: int = 8,
    source_commit: str | None = None,
    authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run bounded, hash-bound distributed SFT and stop at an exact boundary."""

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
        keep_checkpoints=keep_checkpoints,
        precision=precision,
        gradient_sync_mode=gradient_sync_mode,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        attention_implementation=attention_implementation,
        optimizer_profile=optimizer_profile,
        pad_to_multiple_of=pad_to_multiple_of,
        record_limit=encoded_shards_record_limit,
        seed=seed,
    )
    encoded_shards_path = Path(encoded_shards)
    source_checkpoint_path = Path(source_checkpoint)
    output_path = Path(output)
    authorization = None if authorization_path is None else Path(authorization_path)
    _mark_invocation(
        context=context,
        output=output_path,
        resume=resume,
        stop_after_step=stop_after_step,
        target_steps=target_steps,
    )
    dataset, source_config_bytes, source_manifest_sha256 = _collective_load_inputs(
        context=context,
        encoded_shards=encoded_shards_path,
        source_checkpoint=source_checkpoint_path,
        source_checkpoint_manifest_sha256=source_checkpoint_manifest_sha256,
        encoded_shards_inventory_sha256=encoded_shards_inventory_sha256,
        encoded_shards_manifest_file_sha256=(
            encoded_shards_manifest_file_sha256
        ),
        encoded_shards_manifest_sha256=encoded_shards_manifest_sha256,
        encoded_shards_metadata_inventory_sha256=(
            encoded_shards_metadata_inventory_sha256
        ),
        encoded_shards_record_limit=encoded_shards_record_limit,
        encoded_shards_seal_sha256=encoded_shards_seal_sha256,
    )
    source_config = LumiConfig(**json.loads(source_config_bytes))
    source_config._attn_implementation = attention_implementation
    if int(dataset.manifest["context_window"]) > source_config.max_position_embeddings:
        raise ValueError("distributed SFT data exceeds the source context window")

    run_identity = build_distributed_sft_run_identity(
        source_config_bytes=source_config_bytes,
        source_checkpoint_manifest_sha256=source_manifest_sha256,
        shard_manifest=dataset.manifest,
        shard_verification=dataset.verification,
        encoded_shards_record_limit=encoded_shards_record_limit,
        backend=context.backend,
        world_size=context.world_size,
        strategy=strategy,
        per_rank_batch_size=per_rank_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        checkpoint_steps=checkpoint_steps,
        keep_checkpoints=keep_checkpoints,
        target_steps=target_steps,
        seed=seed,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        attention_implementation=attention_implementation,
        optimizer_profile=optimizer_profile,
        precision=precision,
        gradient_sync_mode=gradient_sync_mode,
        pad_to_multiple_of=pad_to_multiple_of,
        source_commit=source_commit,
    )
    run_fingerprint = _collective_require_run_identity(context, run_identity)
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
            "encoded_shards": str(encoded_shards_path.resolve()),
            "source_checkpoint": str(source_checkpoint_path.resolve()),
        },
    }
    state = _setup_state(
        context=context,
        output=output_path,
        run_fingerprint=run_fingerprint,
        run_specification=run_specification,
        resume=resume,
    )

    torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LumiForCausalLM.from_pretrained(
        source_checkpoint_path,
        local_files_only=True,
    )
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
        _restore_checkpoint(
            context=context,
            checkpoint=output_path / state["latest_checkpoint"],
            model=distributed_model,
            optimizer=optimizer,
        )

    if len(dataset) < context.world_size * per_rank_batch_size:
        raise ValueError("distributed SFT data cannot fill one global microbatch")
    if dataset.maximum_record_tokens > source_config.max_position_embeddings:
        raise ValueError("distributed SFT record exceeds the source context window")
    planner = EpochPermutationPlanner(dataset_length=len(dataset), seed=seed)
    metrics_path = output_path / DISTRIBUTED_METRICS_FILENAME
    state_path = output_path / DISTRIBUTED_TRAINING_STATE_FILENAME
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
                pad_to_multiple_of=pad_to_multiple_of,
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
            synchronization = (
                distributed_model.no_sync() if defer_sync else nullcontext()
            )
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
            checkpoint = _save_checkpoint(
                context=context,
                output=output_path,
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
            _append_jsonl(metrics_path, metric)
            if should_checkpoint:
                _write_json_atomic(state_path, state)
        dist.barrier()
        if should_checkpoint and keep_checkpoints is not None:
            _rank_zero_object(
                context,
                lambda: _prune_checkpoints(
                    output_path,
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
        state = _normalize_top_state(state)
        _write_json_atomic(state_path, state)
    dist.barrier()
    result = [state if context.is_rank_zero else None]
    dist.broadcast_object_list(result, src=0)
    return result[0]


__all__ = [
    "SFT_DISTRIBUTED_CHECKPOINT_SCHEMA_VERSION",
    "SFT_DISTRIBUTED_OBJECTIVE",
    "SFT_DISTRIBUTED_RESUME_STARTED_FILENAME",
    "SFT_DISTRIBUTED_RESUME_STARTED_SCHEMA_VERSION",
    "SFT_DISTRIBUTED_STARTED_FILENAME",
    "SFT_DISTRIBUTED_STARTED_SCHEMA_VERSION",
    "SFT_DISTRIBUTED_STATE_SCHEMA_VERSION",
    "build_distributed_sft_run_identity",
    "train_distributed_sft",
    "verify_distributed_sft_checkpoint",
]
