"""Bounded local-only DDP/FSDP CPU correctness preflight orchestrator.

The orchestrator consumes a hash-bound plan, diagnoses the historical v0.1
top-level state-shape omission without retaining payload text, and then starts
exactly six sequential two-rank localhost launches.  Child stdout and stderr
are discarded; the only durable launch logs are small allowlisted JSON
summaries.  A fresh exclusive marker is published before a child can open any
selected encoded record, so a started attempt can never be retried in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .distributed_training import (
    DISTRIBUTED_METRICS_FILENAME,
    DISTRIBUTED_TRAINING_STATE_FILENAME,
)
from .posttraining_distributed_sft_v2 import (
    SFT_DISTRIBUTED_RESUME_STARTED_FILENAME,
    SFT_DISTRIBUTED_STARTED_FILENAME,
    verify_distributed_sft_checkpoint,
)

PLAN_SCHEMA_VERSION = "lumi.sft-distributed-cpu-preflight-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-distributed-cpu-preflight-report.v0.2"
PLAN_STATUS = "authorized_bounded_local_cpu_distributed_preflight_v0_2"
REPORT_STATUS = "PASS_local_cpu_ddp_fsdp_exact_resume_v0_2"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-distributed-cpu-preflight-v0.2.json")
EXPECTED_AUTHORIZATIONS = [
    {
        "bytes": 85,
        "sha256": "bea13465b5158253445668cee99964a1190d128c93bbc652ae7e52b7e9ade104",
        "text": (
            "okay sob thik thakle next step shuru kor ar jodi kichu skip thake "
            "to age sob shes kor"
        ),
    },
    {
        "bytes": 44,
        "sha256": "b6eed736abf2990c50d31647ec682a743ea16c731f47f9f48f59789d29abf12b",
        "text": "mamma kaj shuru kor paused thakle resume kor",
    },
]

EXPECTED_AUTHORIZATION_BOUNDARY = {
    "candidate_or_source_row_access_authorized": False,
    "dataset_or_shard_mutation_authorized": False,
    "external_network_authorized": False,
    "full_4b_checkpoint_restore_authorized": False,
    "gpu_authorized": False,
    "historical_v0_1_read_only_diagnostic_authorized": True,
    "inference_authorized": False,
    "localhost_gloo_rendezvous_authorized": True,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "registry_or_promotion_mutation_authorized": False,
    "remote_or_instance_action_authorized": False,
    "selected_encoded_prefix_read_authorized": True,
    "tiny_local_cpu_distributed_training_authorized": True,
}

EXPECTED_INPUT = {
    "encoded_directory": (
        "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/encoded"
    ),
    "inventory_sha256": (
        "61ce3f4915810f39c7a6b53eb70d3eee3f19e7027c4975babdf4160677a39588"
    ),
    "manifest_file_sha256": (
        "b2ec92c7e4f75a770680c794cad1e3053cc1c8a92d288a2ce2bb5a07e86941bb"
    ),
    "manifest_internal_sha256": (
        "a75a1cb8ef1853aad519fefb3c24b0179828deb315b1d7d14b8013f2771febed"
    ),
    "metadata_inventory_sha256": (
        "cb3afd8d6ee1768fcd2ff28d74d7c8b7f992f544e55c3d0c20470d063f2a3a8c"
    ),
    "record_limit": 4,
    "seal_internal_sha256": (
        "8c575ab8f6acafe326ec5f19c3c750c86ebe42c5f851513eaee28da8751e05b7"
    ),
    "source_checkpoint": (
        "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/source"
    ),
    "source_checkpoint_manifest_sha256": (
        "2aeb1e3352b7b31c3ab16babee514ba69166626c4292ae5ed594e7a42fe3fa91"
    ),
}

EXPECTED_PER_RANK_LOAD_AUDIT = {
    "binary_bytes_read": 1_752,
    "encoded_index_row_reads": 4,
    "file_opens": 4,
    "fully_hashed_shards": 0,
    "index_bytes_read": 644,
    "metadata_inventory_verified_before_selected_content_read": True,
    "opened_shards": 1,
    "records_loaded": 4,
    "unbuffered_selected_content_reads": True,
}

EXPECTED_TRAINING = {
    "attention_implementation": "eager",
    "backend": "gloo",
    "checkpoint_steps": 1,
    "collective_timeout_seconds": 300,
    "gradient_accumulation_steps": 1,
    "gradient_sync_mode": "last_microbatch",
    "learning_rate": 0.0001,
    "lr_schedule": "cosine",
    "maximum_gradient_norm": 1.0,
    "minimum_learning_rate_ratio": 0.1,
    "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
    "pad_to_multiple_of": 8,
    "per_rank_batch_size": 1,
    "precision": "fp32",
    "seed": 2027,
    "target_steps": 2,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "world_size": 2,
}

EXPECTED_EXECUTION = {
    "global_optimizer_steps": 8,
    "launches": 6,
    "logical_training_samples": 16,
    "rank_local_dataset_loads": 12,
    "rank_local_optimizer_steps": 16,
    "sequential_launches": True,
    "torchrun_module": "torch.distributed.run",
}

EXPECTED_LIMITS = {
    "maximum_encoded_binary_bytes_read": 21_024,
    "maximum_encoded_index_bytes_read": 7_728,
    "maximum_encoded_index_rows_read": 48,
    "maximum_external_network_bytes": 0,
    "maximum_global_optimizer_steps": 8,
    "maximum_gpu_use": 0,
    "maximum_historical_v0_1_private_bytes_read": 360_000_000,
    "maximum_inference_calls": 0,
    "maximum_launch_log_bytes_each": 4_096,
    "maximum_launches": 6,
    "maximum_local_runtime_minutes": 30,
    "maximum_logical_training_samples": 16,
    "maximum_private_write_bytes": 536_870_912,
    "maximum_rank_local_dataset_loads": 12,
    "maximum_rank_local_optimizer_steps": 16,
    "maximum_report_bytes": 1_048_576,
    "maximum_retries_after_started_attempt": 0,
    "verification_write_bytes_reserved": 1_048_576,
}

EXPECTED_OUTPUTS = {
    "ddp_reference": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/"
        "ddp-reference"
    ),
    "ddp_resumed": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/"
        "ddp-resumed"
    ),
    "fsdp_reference": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/"
        "fsdp-reference"
    ),
    "fsdp_resumed": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/"
        "fsdp-resumed"
    ),
    "launch_logs": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/logs"
    ),
    "no_retry_marker": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private/"
        "distributed-cpu-preflight-started.json"
    ),
    "private_root": (
        "artifacts/lumi-sft-distributed-cpu-preflight-v0.2/attempt-001/private"
    ),
    "report": "evidence/lumi-4b-sft-distributed-cpu-preflight-v0.2.json",
    "verification": (
        "evidence/lumi-4b-sft-distributed-cpu-preflight-verification-v0.2.json"
    ),
}

EXPECTED_HISTORICAL_V0_1 = {
    "common_value_difference_keys": [
        "created_at",
        "last_checkpoint_seconds",
        "updated_at",
    ],
    "file_bytes": 168_626_988,
    "file_count": 68,
    "inventory_sha256": (
        "f3a510b3e1411454254a62c503a67c2cabaa7a8e8e24b084cddd69c5704f8a58"
    ),
    "private_root": "artifacts/lumi-sft-distributed-cpu-preflight-v0.1",
    "resume_extra_key": "checkpoint_schema_version",
    "resume_extra_value": 1,
    "strategies": ["ddp", "fsdp"],
}

EXPECTED_HISTORICAL_STATE_KEYS = frozenset(
    {
        "best_loss",
        "created_at",
        "last_checkpoint_seconds",
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
EXPECTED_FINAL_STATE_KEYS = EXPECTED_HISTORICAL_STATE_KEYS

EXPECTED_FIXED_BINDINGS = {
    "cpu_smoke_recovery_plan": {
        "path": "configs/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-v0.3.json",
        "sha256": "deec88d1415526fd89053e7f5841af59293365e91c84d62177a49374b4d912d8",
    },
    "current_data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.2.json",
        "sha256": "281fba028c58f5bc0fd489c1d70694e54d37d5dffd09dcc79676d42029497d33",
    },
    "current_seal": {
        "path": "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.2.json",
        "sha256": "1c78bcea6016b4903c4b6262f24cfcc718b6a88b68a2d1c1e3fc13bbb8ba0ce9",
    },
    "current_shard_receipt": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-v0.2.json",
        "sha256": "3daf1a936177cbd9092c842fdbdd171e61c06a09e12cdf54b715230bb58040cb",
    },
    "current_shard_verification": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-verification-v0.2.json",
        "sha256": "701c8b2e9659526ec3f0a780f099bff32d22c83cb6b8d0f404460934d623c86a",
    },
    "configuration_implementation": {
        "path": "src/lumi/configuration_lumi.py",
        "sha256": "ab56700d837e314cad4a4493039f699d4f0a0239dfea16e893e1279b261c05aa",
    },
    "cpu_smoke_recovery_report": {
        "path": "evidence/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-v0.3.json",
        "sha256": "0b13bbb14cedb063069f260e3cdaf1273a213bef2e8fe95648b3246a610b03ea",
    },
    "cpu_smoke_recovery_verification": {
        "path": (
            "evidence/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-"
            "verification-v0.3.json"
        ),
        "sha256": "b6638d363135390a1827fbef5c2b429a1b20a96b870392bde095caa8b237254f",
    },
    "distributed_training_runtime": {
        "path": "src/lumi/distributed_training.py",
        "sha256": "6c3b2311ed632898770a0a48feb4e8cd3b4c13b98249131a91c1c87ab0e4b88e",
    },
    "encoded_manifest": {
        "path": (
            "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/"
            "private/encoded/manifest.json"
        ),
        "sha256": EXPECTED_INPUT["manifest_file_sha256"],
    },
    "historical_implementation": {
        "path": "src/lumi/posttraining_distributed_sft.py",
        "sha256": "b747128310ceb83706b2b3d1842025c587b08fda681e6a7daf9e7d6786152f28",
    },
    "historical_plan": {
        "path": "configs/lumi-4b-sft-distributed-cpu-preflight-v0.1.json",
        "sha256": "ce69aa8d9bbac368fa7df2597b3734631a7e973c5ddf144bd93ca0357f3c2dcd",
    },
    "historical_report": {
        "path": "evidence/lumi-4b-sft-distributed-cpu-preflight-v0.1.json",
        "sha256": "cd060b96447b3decf2da4a04791fb62b797cf4026d8acc655db809a4a12d9a82",
    },
    "historical_test": {
        "path": "tests/test_posttraining_distributed_sft.py",
        "sha256": "6075384e4ceb04adff439509f0fbb29c16915855f26633644bff25564e45f223",
    },
    "historical_verification": {
        "path": (
            "evidence/lumi-4b-sft-distributed-cpu-preflight-verification-v0.1.json"
        ),
        "sha256": "7b395abebdbb223be0c2de3a09d63399e15d1eaa0825acd33aea717d9da07abb",
    },
    "historical_worker": {
        "path": "scripts/train_distributed_sft.py",
        "sha256": "33d9358e359815099c17dda60e1e76e9c522d05043e579756f6c23577cd20315",
    },
    "model_implementation": {
        "path": "src/lumi/modeling_lumi.py",
        "sha256": "bf7b908d282a752132918dddad648fd8435d3f8ad9db41eac1c2e64715926fad",
    },
    "optimizer_implementation": {
        "path": "src/lumi/pilot_training.py",
        "sha256": "d59707436f50775152f841d3c1290f4d6edb9abfdbe65eaa7263063f0d18102a",
    },
    "source_checkpoint_manifest": {
        "path": (
            "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/source/"
            "manifest.sha256"
        ),
        "sha256": EXPECTED_INPUT["source_checkpoint_manifest_sha256"],
    },
    "sft_runtime_v4": {
        "path": "src/lumi/posttraining_sft_runtime_v4.py",
        "sha256": "5782357e8f6e3a734d93c493d3ad1801fc3af81eeb2fd34d494541cad991ed27",
    },
}

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_distributed_cpu_preflight_v2.py",
    "orchestrator_runner": "scripts/run_sft_distributed_cpu_preflight_v2.py",
    "test": "tests/test_sft_distributed_cpu_preflight_v2.py",
    "verifier": "scripts/verify_sft_distributed_cpu_preflight_v2.py",
    "verification_implementation": "src/lumi/posttraining_distributed_cpu_verification_v2.py",
    "verification_test": "tests/test_sft_distributed_cpu_verification_v2.py",
    "worker_implementation": "src/lumi/posttraining_distributed_sft_v2.py",
    "worker_runner": "scripts/train_distributed_sft_v2.py",
    "worker_test": "tests/test_posttraining_distributed_sft_v2.py",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RAW_TEXT_KEYS = frozenset(
    {
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "record_id",
        "response",
        "text",
        "user_prompt",
    }
)
_TIMING_STATE_FIELDS = frozenset(
    {"created_at", "last_checkpoint_seconds", "updated_at"}
)
_TIMING_METRIC_FIELDS = frozenset(
    {"checkpoint_seconds", "elapsed_seconds", "tokens_per_second"}
)
_MAXIMUM_JSON_INPUT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class LaunchSpec:
    code: str
    output_key: str
    resume: bool
    stop_after_step: int
    strategy: str


LAUNCH_SPECS = (
    LaunchSpec("ddp_reference", "ddp_reference", False, 2, "ddp"),
    LaunchSpec("ddp_pause", "ddp_resumed", False, 1, "ddp"),
    LaunchSpec("ddp_resume", "ddp_resumed", True, 2, "ddp"),
    LaunchSpec(
        "fsdp_reference", "fsdp_reference", False, 2, "fsdp_full_shard"
    ),
    LaunchSpec("fsdp_pause", "fsdp_resumed", False, 1, "fsdp_full_shard"),
    LaunchSpec("fsdp_resume", "fsdp_resumed", True, 2, "fsdp_full_shard"),
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in distributed CPU preflight input")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {item}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("distributed CPU preflight JSON is not UTF-8") from error


def assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError(f"raw training text key forbidden at {path}.{key}")
            assert_raw_text_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_raw_text_free(child, f"{path}[{index}]")


def _read_regular_file(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    payload = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("distributed CPU preflight input is not one regular file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ValueError("distributed CPU preflight input exceeds its byte bound")
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            if maximum_bytes is not None and len(payload) > maximum_bytes:
                raise ValueError("distributed CPU preflight input exceeds its byte bound")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("distributed CPU preflight input changed while read")
    finally:
        os.close(descriptor)
    return bytes(payload)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_read_regular_file(Path(path))).hexdigest()


def _sha256_regular_file_bounded(
    path: Path, *, maximum_bytes: int
) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    observed = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise ValueError("bounded distributed CPU file identity changed")
        while chunk := os.read(descriptor, 1024 * 1024):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError("bounded distributed CPU file read cap exceeded")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            observed != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise ValueError("bounded distributed CPU file changed while read")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), observed


def _read_json(path: Path, *, maximum_bytes: int = _MAXIMUM_JSON_INPUT_BYTES) -> dict[str, Any]:
    value = strict_json_bytes(_read_regular_file(path, maximum_bytes=maximum_bytes))
    if not isinstance(value, dict):
        raise ValueError("distributed CPU preflight JSON root is not an object")
    return value


def _repository_path(repository_root: Path, relative: str) -> Path:
    candidate = repository_root / relative
    if Path(relative).is_absolute():
        raise ValueError("distributed CPU preflight path is absolute")
    absolute = Path(os.path.abspath(candidate))
    if not absolute.is_relative_to(repository_root):
        raise ValueError("distributed CPU preflight path escaped repository")
    return absolute


def _require_safe_parent_chain(repository_root: Path, path: Path) -> None:
    current = repository_root
    for part in path.relative_to(repository_root).parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("distributed CPU preflight parent chain is unsafe")


def _verify_binding(
    repository_root: Path,
    name: str,
    binding: Mapping[str, Any],
) -> None:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"distributed CPU preflight binding shape changed: {name}")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"distributed CPU preflight binding is invalid: {name}")
    path = _repository_path(repository_root, relative)
    _require_safe_parent_chain(repository_root, path)
    if sha256_file(path) != digest:
        raise ValueError(f"distributed CPU preflight binding changed: {name}")


def validate_plan(plan: Mapping[str, Any], *, repository_root: Path) -> None:
    expected_top = {
        "authorization_boundary",
        "authorizations",
        "bindings",
        "execution",
        "historical_v0_1",
        "input",
        "limits",
        "outputs",
        "production_eligible",
        "raw_training_text_retained",
        "schema_version",
        "status",
        "training",
    }
    if (
        set(plan) != expected_top
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorizations") != EXPECTED_AUTHORIZATIONS
        or plan.get("authorization_boundary") != EXPECTED_AUTHORIZATION_BOUNDARY
        or plan.get("input") != EXPECTED_INPUT
        or plan.get("training") != EXPECTED_TRAINING
        or plan.get("execution") != EXPECTED_EXECUTION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("outputs") != EXPECTED_OUTPUTS
        or plan.get("historical_v0_1") != EXPECTED_HISTORICAL_V0_1
        or plan.get("production_eligible") is not False
        or plan.get("raw_training_text_retained") is not False
    ):
        raise ValueError("distributed CPU preflight plan contract changed")
    for authorization in EXPECTED_AUTHORIZATIONS:
        payload = authorization["text"].encode("utf-8")
        if (
            len(payload) != authorization["bytes"]
            or hashlib.sha256(payload).hexdigest() != authorization["sha256"]
        ):
            raise RuntimeError("compiled distributed CPU authorization changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != (
        set(EXPECTED_FIXED_BINDINGS) | set(EXPECTED_CODE_PATHS)
    ):
        raise ValueError("distributed CPU preflight binding inventory changed")
    for name, expected in EXPECTED_FIXED_BINDINGS.items():
        if bindings.get(name) != expected:
            raise ValueError(f"fixed distributed CPU binding changed: {name}")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"distributed CPU code path changed: {name}")
    for name, binding in bindings.items():
        assert isinstance(binding, Mapping)
        _verify_binding(repository_root, str(name), binding)


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(os.path.abspath(path))
    expected = Path(os.path.abspath(root / EXPECTED_PLAN_PATH))
    if plan_path != expected:
        raise ValueError("distributed CPU preflight plan path changed")
    _require_safe_parent_chain(root, plan_path)
    plan = _read_json(plan_path)
    validate_plan(plan, repository_root=root)
    return plan


def _historical_snapshot(
    root: Path, *, maximum_bytes: int
) -> tuple[dict[str, Any], int]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("historical distributed CPU root is unsafe")
    entries: list[dict[str, Any]] = []
    bytes_read = 0
    initial_paths = sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )
    for path in initial_paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("historical distributed CPU tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("historical distributed CPU tree contains a special file")
        digest, file_bytes = _sha256_regular_file_bounded(
            path, maximum_bytes=maximum_bytes - bytes_read
        )
        bytes_read += file_bytes
        entries.append(
            {
                "bytes": file_bytes,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
            }
        )
    final_names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    initial_names = [path.relative_to(root).as_posix() for path in initial_paths]
    if final_names != initial_names:
        raise ValueError("historical distributed CPU inventory changed during scan")
    return (
        {
            "file_bytes": sum(item["bytes"] for item in entries),
            "file_count": len(entries),
            "inventory_sha256": sha256_value(entries),
        },
        bytes_read,
    )


def _historical_state_delta(
    reference: Mapping[str, Any], resumed: Mapping[str, Any]
) -> dict[str, Any]:
    reference_keys = set(reference)
    resumed_keys = set(resumed)
    if reference_keys != EXPECTED_HISTORICAL_STATE_KEYS:
        raise ValueError("historical reference top-state key inventory changed")
    if resumed_keys != (
        EXPECTED_HISTORICAL_STATE_KEYS
        | {EXPECTED_HISTORICAL_V0_1["resume_extra_key"]}
    ):
        raise ValueError("historical resumed top-state failure signature changed")
    extra_key = EXPECTED_HISTORICAL_V0_1["resume_extra_key"]
    if resumed.get(extra_key) != EXPECTED_HISTORICAL_V0_1["resume_extra_value"]:
        raise ValueError("historical resumed checkpoint schema value changed")
    common_differences = sorted(
        key for key in reference_keys if reference[key] != resumed[key]
    )
    if common_differences != EXPECTED_HISTORICAL_V0_1[
        "common_value_difference_keys"
    ]:
        raise ValueError("historical distributed state differences changed")
    return {
        "common_key_count": len(reference_keys),
        "common_value_difference_keys": common_differences,
        "reference_key_count": len(reference_keys),
        "resume_extra_keys": [extra_key],
        "resume_key_count": len(resumed_keys),
        "resume_missing_keys": [],
        "resume_schema_value": resumed[extra_key],
    }


def diagnose_historical_v0_1(
    *, repository_root: Path, limits: Mapping[str, Any]
) -> dict[str, Any]:
    root = _repository_path(
        repository_root, EXPECTED_HISTORICAL_V0_1["private_root"]
    )
    snapshot, bytes_read = _historical_snapshot(
        root,
        maximum_bytes=limits["maximum_historical_v0_1_private_bytes_read"],
    )
    expected_snapshot = {
        key: EXPECTED_HISTORICAL_V0_1[key]
        for key in ("file_bytes", "file_count", "inventory_sha256")
    }
    if snapshot != expected_snapshot:
        raise ValueError("historical distributed CPU snapshot changed")
    strategies: dict[str, Any] = {}
    for strategy in EXPECTED_HISTORICAL_V0_1["strategies"]:
        reference_path = root / f"{strategy}-reference" / DISTRIBUTED_TRAINING_STATE_FILENAME
        resumed_path = root / f"{strategy}-resume" / DISTRIBUTED_TRAINING_STATE_FILENAME
        reference_payload = _read_regular_file(
            reference_path, maximum_bytes=_MAXIMUM_JSON_INPUT_BYTES
        )
        resumed_payload = _read_regular_file(
            resumed_path, maximum_bytes=_MAXIMUM_JSON_INPUT_BYTES
        )
        bytes_read += len(reference_payload) + len(resumed_payload)
        reference = strict_json_bytes(reference_payload)
        resumed = strict_json_bytes(resumed_payload)
        if not isinstance(reference, dict) or not isinstance(resumed, dict):
            raise ValueError("historical distributed state is not an object")
        strategies[strategy] = _historical_state_delta(reference, resumed)
    if bytes_read > limits["maximum_historical_v0_1_private_bytes_read"]:
        raise ValueError("historical distributed diagnostic read cap exceeded")
    result = {
        "diagnostic_bytes_read": bytes_read,
        "diagnostic_passed": True,
        "historical_claim_incomplete": True,
        "historical_file_bytes": snapshot["file_bytes"],
        "historical_file_count": snapshot["file_count"],
        "historical_inventory_sha256": snapshot["inventory_sha256"],
        "historical_v0_1_audit_complete": False,
        "incompleteness_code": "top_state_key_inventory_not_compared_v0_1",
        "raw_training_text_retained": False,
        "replacement_required": True,
        "strategies": strategies,
    }
    assert_raw_text_free(result)
    return result


def _mkdir_private(path: Path) -> None:
    path.mkdir(mode=0o700)
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
    ):
        raise ValueError("distributed CPU private directory is unsafe")


def _create_attempt_root(repository_root: Path, plan: Mapping[str, Any]) -> Path:
    root = _repository_path(repository_root, str(plan["outputs"]["private_root"]))
    _require_safe_parent_chain(repository_root, root)
    parent = root.parent
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError("distributed CPU attempt parent is unsafe")
    for path in reversed(missing):
        _mkdir_private(path)
    if root.exists() or root.is_symlink():
        raise FileExistsError("distributed CPU preflight attempt already exists")
    _mkdir_private(root)
    return root


def _write_exclusive_private_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    maximum_bytes: int,
) -> int:
    assert_raw_text_free(payload)
    data = json_file_bytes(payload)
    if len(data) > maximum_bytes:
        raise ValueError("distributed CPU private JSON exceeds its byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if (
            stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size != len(data)
        ):
            raise ValueError("distributed CPU private JSON metadata changed")
    finally:
        os.close(descriptor)
    return len(data)


def _publish_no_retry_marker(
    *, repository_root: Path, plan: Mapping[str, Any], plan_file_sha256: str
) -> tuple[Path, int]:
    marker = _repository_path(
        repository_root, str(plan["outputs"]["no_retry_marker"])
    )
    private_root = _repository_path(
        repository_root, str(plan["outputs"]["private_root"])
    )
    if marker.parent != private_root:
        raise ValueError("distributed CPU no-retry marker left private root")
    payload = {
        "external_network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "launches_authorized": len(LAUNCH_SPECS),
        "plan_file_sha256": plan_file_sha256,
        "raw_training_text_retained": False,
        "retries_after_this_marker": 0,
        "schema_version": "lumi.sft-distributed-cpu-preflight-started.v0.2",
        "status": "STARTED_no_retry",
    }
    return marker, _write_exclusive_private_json(
        marker,
        payload,
        maximum_bytes=plan["limits"]["maximum_launch_log_bytes_each"],
    )


def _require_fresh_outputs(repository_root: Path, plan: Mapping[str, Any]) -> None:
    for key in ("private_root", "report", "verification"):
        path = _repository_path(repository_root, str(plan["outputs"][key]))
        _require_safe_parent_chain(repository_root, path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "distributed CPU preflight output already exists before start"
            )


def _launch_command(
    *,
    python_executable: Path,
    plan: Mapping[str, Any],
    repository_root: Path,
    spec: LaunchSpec,
    port: int,
) -> list[str]:
    training = plan["training"]
    input_specification = plan["input"]
    output = _repository_path(repository_root, str(plan["outputs"][spec.output_key]))
    command = [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "--local-addr=127.0.0.1",
        EXPECTED_CODE_PATHS["worker_runner"],
        "--encoded-shards",
        str(_repository_path(repository_root, input_specification["encoded_directory"])),
        "--encoded-shards-manifest-file-sha256",
        input_specification["manifest_file_sha256"],
        "--encoded-shards-manifest-sha256",
        input_specification["manifest_internal_sha256"],
        "--encoded-shards-inventory-sha256",
        input_specification["inventory_sha256"],
        "--encoded-shards-metadata-inventory-sha256",
        input_specification["metadata_inventory_sha256"],
        "--encoded-shards-seal-sha256",
        input_specification["seal_internal_sha256"],
        "--encoded-shards-record-limit",
        str(input_specification["record_limit"]),
        "--source-checkpoint",
        str(_repository_path(repository_root, input_specification["source_checkpoint"])),
        "--source-checkpoint-manifest-sha256",
        input_specification["source_checkpoint_manifest_sha256"],
        "--output",
        str(output),
        "--target-steps",
        str(training["target_steps"]),
        "--stop-after-step",
        str(spec.stop_after_step),
        "--strategy",
        spec.strategy,
        "--per-rank-batch-size",
        str(training["per_rank_batch_size"]),
        "--gradient-accumulation-steps",
        str(training["gradient_accumulation_steps"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--max-grad-norm",
        str(training["maximum_gradient_norm"]),
        "--checkpoint-steps",
        str(training["checkpoint_steps"]),
        "--keep-checkpoints",
        str(training["target_steps"]),
        "--seed",
        str(training["seed"]),
        "--lr-schedule",
        training["lr_schedule"],
        "--warmup-steps",
        str(training["warmup_steps"]),
        "--min-learning-rate-ratio",
        str(training["minimum_learning_rate_ratio"]),
        "--attention-implementation",
        training["attention_implementation"],
        "--optimizer-profile",
        training["optimizer_profile"],
        "--precision",
        training["precision"],
        "--gradient-sync-mode",
        training["gradient_sync_mode"],
        "--pad-to-multiple-of",
        str(training["pad_to_multiple_of"]),
        "--device",
        "cpu",
        "--backend",
        "gloo",
        "--timeout-seconds",
        str(training["collective_timeout_seconds"]),
    ]
    if spec.resume:
        command.append("--resume")
    return command


def _offline_environment(repository_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.lower() in {
            "all_proxy",
            "ftp_proxy",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        }:
            environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "GLOO_SOCKET_IFNAME": "lo0" if platform.system() == "Darwin" else "lo",
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "NO_PROXY": "localhost,127.0.0.1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repository_root / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _terminate_process_group(process: subprocess.Popen[bytes]) -> int | None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def _run_launch(
    *,
    command: list[str],
    environment: Mapping[str, str],
    repository_root: Path,
    spec: LaunchSpec,
    log_path: Path,
    timeout_seconds: float,
    maximum_log_bytes: int,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise TimeoutError("distributed CPU preflight runtime cap exhausted")
    status = "completed"
    return_code: int | None = None
    process: subprocess.Popen[bytes] | None = None
    interrupted = False
    try:
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return_code = process.wait(timeout=timeout_seconds)
        if return_code != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        assert process is not None
        return_code = _terminate_process_group(process)
    except OSError:
        status = "launch_error"
    except BaseException:  # ensure torchrun children cannot outlive this bounded attempt
        status = "interrupted"
        interrupted = True
        if process is not None:
            return_code = _terminate_process_group(process)
    log = {
        "child_output_bytes_retained": 0,
        "external_network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
        "processes": 2,
        "raw_training_text_retained": False,
        "return_code": return_code,
        "schema_version": "lumi.sft-distributed-cpu-launch-log.v0.2",
        "stage_code": spec.code,
        "status": status,
    }
    log_bytes = _write_exclusive_private_json(
        log_path, log, maximum_bytes=maximum_log_bytes
    )
    if interrupted:
        raise RuntimeError(
            f"distributed CPU preflight interrupted at stage code {spec.code}"
        ) from None
    if status != "completed":
        raise RuntimeError(
            f"distributed CPU preflight launch failed at stage code {spec.code}"
        )
    return {"log_bytes": log_bytes, **log}


def _validate_read_audit(state: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    run_specification = state.get("run_specification")
    if not isinstance(run_specification, Mapping):
        raise ValueError("distributed CPU run specification is missing")
    distributed = run_specification.get("distributed")
    data = run_specification.get("data")
    if not isinstance(distributed, Mapping) or not isinstance(data, Mapping):
        raise ValueError("distributed CPU run identity is incomplete")
    if (
        distributed.get("world_size") != 2
        or distributed.get("backend") != "gloo"
        or distributed.get("strategy") not in {"ddp", "fsdp_full_shard"}
        or distributed.get("precision") != "fp32"
    ):
        raise ValueError("distributed CPU topology changed")
    expected_data = {
        "inventory_sha256": plan["input"]["inventory_sha256"],
        "manifest_file_sha256": plan["input"]["manifest_file_sha256"],
        "manifest_sha256": plan["input"]["manifest_internal_sha256"],
        "metadata_inventory_sha256": plan["input"]["metadata_inventory_sha256"],
        "record_limit": plan["input"]["record_limit"],
        "seal_sha256": plan["input"]["seal_internal_sha256"],
        "selection_scope": "contiguous_prefix_v0.1",
        "sequence_order": "cached_epoch_randperm_v0.1",
        "split": "train",
        "verification_profile": "token_consistent_v0.2",
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise ValueError("distributed CPU selected input identity changed")
    expected_keys = set(expected_data) | {
        "read_audit",
        "read_audit_accounting",
        "selected_prefix_sha256",
    }
    if set(data) != expected_keys:
        raise ValueError("distributed CPU data identity key inventory changed")
    selected_prefix = data.get("selected_prefix_sha256")
    if not isinstance(selected_prefix, str) or _SHA256_PATTERN.fullmatch(selected_prefix) is None:
        raise ValueError("distributed CPU selected-prefix digest is invalid")
    read_audit = data.get("read_audit")
    accounting = data.get("read_audit_accounting")
    if read_audit != EXPECTED_PER_RANK_LOAD_AUDIT or accounting != {
        "all_rank_read_audits_equal": True,
        "rank_local_loads": 2,
    }:
        raise ValueError("distributed CPU collective physical read audit changed")
    return {
        "binary_bytes_read": read_audit["binary_bytes_read"] * 2,
        "encoded_index_rows_read": read_audit["encoded_index_row_reads"] * 2,
        "index_bytes_read": read_audit["index_bytes_read"] * 2,
        "rank_local_loads": 2,
        "selected_prefix_sha256": selected_prefix,
    }


def _stable_metrics(path: Path) -> list[dict[str, Any]]:
    payload = _read_regular_file(path, maximum_bytes=_MAXIMUM_JSON_INPUT_BYTES)
    metrics: list[dict[str, Any]] = []
    for line in payload.splitlines():
        value = strict_json_bytes(line)
        if not isinstance(value, dict):
            raise ValueError("distributed CPU metric row is not an object")
        stable = {
            key: child for key, child in value.items() if key not in _TIMING_METRIC_FIELDS
        }
        assert_raw_text_free(stable)
        metrics.append(stable)
    return metrics


def _state(path: Path) -> dict[str, Any]:
    value = _read_json(path / DISTRIBUTED_TRAINING_STATE_FILENAME)
    if set(value) != EXPECTED_FINAL_STATE_KEYS:
        raise ValueError("distributed CPU top-state field inventory changed")
    assert_raw_text_free(value)
    return value


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_exact(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _nested_exact(a, b) for a, b in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _semantic_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(child: Any) -> None:
        if isinstance(child, torch.Tensor):
            tensor = child.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(canonical_json(list(tensor.shape)))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(child, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(child, key=lambda item: str(item)):
                digest.update(canonical_json(str(key)))
                update(child[key])
        elif isinstance(child, (list, tuple)):
            digest.update(f"{type(child).__name__}\0".encode("ascii"))
            for item in child:
                update(item)
        else:
            digest.update(b"scalar\0")
            digest.update(canonical_json(child))

    update(value)
    return digest.hexdigest()


def _checkpoint_closed_inventory(checkpoint: Path) -> dict[str, Any]:
    state = verify_distributed_sft_checkpoint(checkpoint)
    manifest_path = checkpoint / "manifest.sha256"
    manifest_payload = _read_regular_file(
        manifest_path, maximum_bytes=_MAXIMUM_JSON_INPUT_BYTES
    )
    declared: set[str] = set()
    for line in manifest_payload.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or _SHA256_PATTERN.fullmatch(parts[0]) is None:
            raise ValueError("distributed checkpoint manifest framing changed")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("distributed checkpoint manifest path is unsafe")
        declared.add(relative.as_posix())
    actual: set[str] = set()
    for path in checkpoint.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("distributed checkpoint contains a symlink")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise ValueError("distributed checkpoint contains a linked file")
            actual.add(path.relative_to(checkpoint).as_posix())
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("distributed checkpoint contains a special file")
    if actual != declared | {"manifest.sha256"}:
        raise ValueError("distributed checkpoint has an open file inventory")
    return {
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_entries": len(declared),
        "state_step": state.get("step"),
    }


def _trainer_pair_audit(reference: Path, resumed: Path, *, step: int) -> dict[str, Any]:
    checkpoint_name = f"checkpoint-{step:08d}"
    rank_digests: list[dict[str, Any]] = []
    for rank in range(2):
        filename = f"trainer-rank-{rank:05d}.pt"
        left = torch.load(
            reference / checkpoint_name / filename,
            map_location="cpu",
            weights_only=True,
        )
        right = torch.load(
            resumed / checkpoint_name / filename,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(left, Mapping)
            or set(left) != {"optimizer", "rng_state"}
            or not _nested_exact(left, right)
        ):
            raise ValueError("distributed CPU optimizer or RNG resume state differs")
        rank_digests.append(
            {
                "optimizer_semantic_sha256": _semantic_digest(left["optimizer"]),
                "rank": rank,
                "rng_semantic_sha256": _semantic_digest(left["rng_state"]),
            }
        )
    return {"rank_digests": rank_digests, "step": step}


def _final_model_digest(path: Path) -> dict[str, Any]:
    value = _read_json(path / "final-model-digest.json")
    expected = {
        "algorithm",
        "checkpoint",
        "checkpoint_manifest_sha256",
        "parameter_values_with_tied_aliases",
        "sha256",
        "tensor_count",
    }
    if set(value) != expected or _SHA256_PATTERN.fullmatch(str(value.get("sha256"))) is None:
        raise ValueError("distributed CPU final model digest shape changed")
    assert_raw_text_free(value)
    return value


def _validate_intermediate_state(
    *, output: Path, plan: Mapping[str, Any], spec: LaunchSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_entries = {
        DISTRIBUTED_METRICS_FILENAME,
        DISTRIBUTED_TRAINING_STATE_FILENAME,
        SFT_DISTRIBUTED_STARTED_FILENAME,
        "final-model-digest.json",
        "checkpoint-00000001",
    }
    if spec.stop_after_step == 2:
        expected_entries.add("checkpoint-00000002")
    if spec.resume:
        expected_entries.add(SFT_DISTRIBUTED_RESUME_STARTED_FILENAME)
    observed_entries = {path.name for path in output.iterdir()}
    if observed_entries != expected_entries:
        raise ValueError("distributed CPU output inventory changed")
    if any(path.is_symlink() for path in output.iterdir()):
        raise ValueError("distributed CPU output contains a symlink")
    state = _state(output)
    expected_status = (
        "complete" if spec.stop_after_step == plan["training"]["target_steps"] else "paused"
    )
    expected_records = spec.stop_after_step * 2
    if (
        state.get("status") != expected_status
        or state.get("step") != spec.stop_after_step
        or state.get("records_seen") != expected_records
        or state.get("sequence_cursor") != expected_records
    ):
        raise ValueError("distributed CPU launch stopped at an unexpected cursor")
    distributed = state["run_specification"]["distributed"]
    if distributed.get("strategy") != spec.strategy:
        raise ValueError("distributed CPU launch strategy changed")
    metrics = _stable_metrics(output / DISTRIBUTED_METRICS_FILENAME)
    if [item.get("step") for item in metrics] != list(
        range(1, spec.stop_after_step + 1)
    ):
        raise ValueError("distributed CPU durable metric steps changed")
    return state, _validate_read_audit(state, plan)


def _audit_strategy(
    *, strategy: str, reference: Path, resumed: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    reference_state = _state(reference)
    resumed_state = _state(resumed)
    if set(reference_state) != set(resumed_state):
        raise ValueError("distributed CPU reference/resume top-state keys differ")
    stable_reference_state = {
        key: value
        for key, value in reference_state.items()
        if key not in _TIMING_STATE_FIELDS
    }
    stable_resumed_state = {
        key: value
        for key, value in resumed_state.items()
        if key not in _TIMING_STATE_FIELDS
    }
    if stable_reference_state != stable_resumed_state:
        raise ValueError("distributed CPU stable final states differ")
    if (
        reference_state.get("status") != "complete"
        or reference_state.get("step") != 2
        or reference_state.get("records_seen") != 4
        or reference_state.get("sequence_cursor") != 4
    ):
        raise ValueError("distributed CPU final state totals changed")
    expected_strategy = "ddp" if strategy == "ddp" else "fsdp_full_shard"
    if (
        reference_state["run_specification"]["distributed"].get("strategy")
        != expected_strategy
    ):
        raise ValueError("distributed CPU final strategy changed")
    reference_metrics = _stable_metrics(reference / DISTRIBUTED_METRICS_FILENAME)
    resumed_metrics = _stable_metrics(resumed / DISTRIBUTED_METRICS_FILENAME)
    if (
        reference_metrics != resumed_metrics
        or [item.get("step") for item in reference_metrics] != [1, 2]
    ):
        raise ValueError("distributed CPU stable resume metrics differ")
    reference_digest = _final_model_digest(reference)
    resumed_digest = _final_model_digest(resumed)
    for output, digest in ((reference, reference_digest), (resumed, resumed_digest)):
        if (
            digest["checkpoint"] != "checkpoint-00000002"
            or digest["checkpoint_manifest_sha256"]
            != sha256_file(output / "checkpoint-00000002" / "manifest.sha256")
        ):
            raise ValueError("distributed CPU model digest checkpoint binding changed")
    semantic_model_fields = {
        "algorithm",
        "parameter_values_with_tied_aliases",
        "sha256",
        "tensor_count",
    }
    if {
        key: reference_digest[key] for key in semantic_model_fields
    } != {key: resumed_digest[key] for key in semantic_model_fields}:
        raise ValueError("distributed CPU final model differs after resume")
    checkpoint_audits: dict[str, Any] = {}
    trainer_audits: dict[str, Any] = {}
    for step in (1, 2):
        name = f"checkpoint-{step:08d}"
        left = _checkpoint_closed_inventory(reference / name)
        right = _checkpoint_closed_inventory(resumed / name)
        if left["state_step"] != step or right["state_step"] != step:
            raise ValueError("distributed CPU checkpoint step changed")
        checkpoint_audits[str(step)] = {
            "reference_manifest_entries": left["manifest_entries"],
            "reference_manifest_sha256": left["checkpoint_manifest_sha256"],
            "resumed_manifest_entries": right["manifest_entries"],
            "resumed_manifest_sha256": right["checkpoint_manifest_sha256"],
            "verified": True,
        }
        trainer_audits[str(step)] = _trainer_pair_audit(
            reference, resumed, step=step
        )
    return {
        "checkpoint_audits": checkpoint_audits,
        "final_model_semantic_sha256": reference_digest["sha256"],
        "final_state": {
            key: reference_state[key]
            for key in (
                "records_seen",
                "sequence_cursor",
                "status",
                "step",
                "supervised_tokens_seen",
                "tokens_seen",
            )
        },
        "loss_sequence": [item["loss"] for item in reference_metrics],
        "stable_metrics_sha256": sha256_value(reference_metrics),
        "top_state_key_count": len(reference_state),
        "trainer_audits": trainer_audits,
    }


def _private_tree_bytes(root: Path) -> int:
    info = root.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
    ):
        raise ValueError("distributed CPU private root metadata changed")
    total = 0
    for path in root.rglob("*"):
        child = path.lstat()
        if stat.S_ISLNK(child.st_mode):
            raise ValueError("distributed CPU private tree contains a symlink")
        if stat.S_ISDIR(child.st_mode):
            if stat.S_IMODE(child.st_mode) != 0o700 or child.st_uid != os.geteuid():
                raise ValueError("distributed CPU private directory mode changed")
        elif stat.S_ISREG(child.st_mode):
            if (
                stat.S_IMODE(child.st_mode) != 0o600
                or child.st_uid != os.geteuid()
                or child.st_nlink != 1
            ):
                raise ValueError("distributed CPU private file metadata changed")
            total += child.st_size
        else:
            raise ValueError("distributed CPU private tree contains a special file")
    return total


def _require_final_private_inventory(
    root: Path, logs_root: Path, plan: Mapping[str, Any]
) -> None:
    expected_root_names = {
        Path(plan["outputs"]["ddp_reference"]).name,
        Path(plan["outputs"]["ddp_resumed"]).name,
        Path(plan["outputs"]["fsdp_reference"]).name,
        Path(plan["outputs"]["fsdp_resumed"]).name,
        Path(plan["outputs"]["launch_logs"]).name,
        Path(plan["outputs"]["no_retry_marker"]).name,
    }
    if {path.name for path in root.iterdir()} != expected_root_names:
        raise ValueError("distributed CPU private-root inventory changed")
    expected_logs = {
        f"{index:02d}-{spec.code}.json"
        for index, spec in enumerate(LAUNCH_SPECS, 1)
    }
    if {path.name for path in logs_root.iterdir()} != expected_logs:
        raise ValueError("distributed CPU launch-log inventory changed")


def _aggregate_load_audits(
    audits: Sequence[Mapping[str, Any]], limits: Mapping[str, Any]
) -> dict[str, Any]:
    if len(audits) != 6:
        raise ValueError("distributed CPU launch audit count changed")
    selected_prefixes = {item["selected_prefix_sha256"] for item in audits}
    result = {
        "encoded_binary_bytes_read": sum(item["binary_bytes_read"] for item in audits),
        "encoded_index_bytes_read": sum(item["index_bytes_read"] for item in audits),
        "encoded_index_rows_read": sum(
            item["encoded_index_rows_read"] for item in audits
        ),
        "rank_local_dataset_loads": sum(item["rank_local_loads"] for item in audits),
        "selected_prefix_sha256": next(iter(selected_prefixes))
        if len(selected_prefixes) == 1
        else None,
    }
    if (
        len(selected_prefixes) != 1
        or result["encoded_binary_bytes_read"]
        != limits["maximum_encoded_binary_bytes_read"]
        or result["encoded_index_bytes_read"]
        != limits["maximum_encoded_index_bytes_read"]
        or result["encoded_index_rows_read"]
        != limits["maximum_encoded_index_rows_read"]
        or result["rank_local_dataset_loads"]
        != limits["maximum_rank_local_dataset_loads"]
    ):
        raise ValueError("distributed CPU aggregate physical read accounting changed")
    return result


def _runtime_remaining_seconds(started: float, maximum_minutes: int) -> float:
    return maximum_minutes * 60 - (time.monotonic() - started)


def _report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    historical: Mapping[str, Any],
    launch_logs: Sequence[Mapping[str, Any]],
    load_accounting: Mapping[str, Any],
    strategy_audits: Mapping[str, Any],
    private_write_bytes: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "authorization_boundary": {
            "dataset_or_shards_mutated": False,
            "external_network_bytes": 0,
            "full_4b_checkpoint_restored": False,
            "gpu_used": False,
            "inference_calls": 0,
            "localhost_rendezvous_only": True,
            "production_sft_started": False,
            "remote_or_instance_action": False,
        },
        "checks": {
            "all_checkpoint_inventories_closed_and_hash_verified": True,
            "all_rank_physical_read_audits_collectively_equal": True,
            "ddp_final_model_optimizer_rng_and_metrics_exact": True,
            "fsdp_final_model_optimizer_rng_and_metrics_exact": True,
            "historical_v0_1_singular_state_leak_confirmed": True,
            "no_retry_marker_preceded_selected_content_reads": True,
            "raw_child_output_not_retained": True,
            "top_state_key_inventories_exact": True,
        },
        "execution": {
            "global_optimizer_steps": plan["execution"]["global_optimizer_steps"],
            "launch_stage_codes": [item["stage_code"] for item in launch_logs],
            "launches": len(launch_logs),
            "logical_training_samples": plan["execution"]["logical_training_samples"],
            "processes_per_launch": 2,
            "rank_local_optimizer_steps": plan["execution"][
                "rank_local_optimizer_steps"
            ],
            "sequential": True,
        },
        "historical_v0_1_diagnostic": dict(historical),
        "input_identity": {
            "inventory_sha256": plan["input"]["inventory_sha256"],
            "manifest_file_sha256": plan["input"]["manifest_file_sha256"],
            "manifest_internal_sha256": plan["input"]["manifest_internal_sha256"],
            "metadata_inventory_sha256": plan["input"]["metadata_inventory_sha256"],
            "record_limit": plan["input"]["record_limit"],
            "seal_internal_sha256": plan["input"]["seal_internal_sha256"],
            "source_checkpoint_manifest_sha256": plan["input"][
                "source_checkpoint_manifest_sha256"
            ],
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "resource_accounting": {
            **dict(load_accounting),
            "external_network_bytes": 0,
            "gpu_use": 0,
            "inference_calls": 0,
            "private_write_bytes_before_report": private_write_bytes,
            "runtime_seconds": runtime_seconds,
        },
        "results": dict(strategy_audits),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    assert_raw_text_free(report)
    report["report_sha256"] = sha256_value(report)
    return report


def run_distributed_cpu_preflight(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    base_port: int = 29_650,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the single authorized six-launch local CPU attempt."""

    started = time.monotonic()
    repository = Path(repository_root).resolve()
    if not 1_024 <= base_port <= 65_529:
        raise ValueError("distributed CPU base port cannot fit six localhost launches")
    plan = load_plan(plan_path, repository_root=repository)
    lexical_plan_path = Path(os.path.abspath(plan_path))
    plan_file_sha256 = sha256_file(lexical_plan_path)
    historical = diagnose_historical_v0_1(
        repository_root=repository, limits=plan["limits"]
    )
    if (
        _runtime_remaining_seconds(
            started, plan["limits"]["maximum_local_runtime_minutes"]
        )
        <= 0
    ):
        raise TimeoutError("distributed CPU preflight runtime cap exhausted")
    _require_fresh_outputs(repository, plan)
    private_root = _create_attempt_root(repository, plan)
    marker_path, _marker_bytes = _publish_no_retry_marker(
        repository_root=repository,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    logs_root = _repository_path(repository, str(plan["outputs"]["launch_logs"]))
    if logs_root.parent != private_root:
        raise ValueError("distributed CPU launch logs left private root")
    _mkdir_private(logs_root)
    if marker_path.parent != private_root:
        raise RuntimeError("distributed CPU marker placement changed")

    executable = Path(sys.executable if python_executable is None else python_executable)
    if (
        not executable.is_file()
        or executable.resolve() != Path(sys.executable).resolve()
    ):
        raise ValueError("distributed CPU Python executable is not a file")
    environment = _offline_environment(repository)
    launch_logs: list[dict[str, Any]] = []
    load_audits: list[dict[str, Any]] = []
    paused_states: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(LAUNCH_SPECS, 1):
        remaining = _runtime_remaining_seconds(
            started, plan["limits"]["maximum_local_runtime_minutes"]
        )
        command = _launch_command(
            python_executable=executable,
            plan=plan,
            repository_root=repository,
            spec=spec,
            port=base_port + index,
        )
        log = _run_launch(
            command=command,
            environment=environment,
            repository_root=repository,
            spec=spec,
            log_path=logs_root / f"{index:02d}-{spec.code}.json",
            timeout_seconds=remaining,
            maximum_log_bytes=plan["limits"]["maximum_launch_log_bytes_each"],
        )
        launch_logs.append(log)
        output = _repository_path(repository, str(plan["outputs"][spec.output_key]))
        state, read_audit = _validate_intermediate_state(
            output=output, plan=plan, spec=spec
        )
        load_audits.append(read_audit)
        if spec.code.endswith("_pause"):
            paused_states[spec.strategy] = {
                "records_seen": state["records_seen"],
                "sequence_cursor": state["sequence_cursor"],
                "status": state["status"],
                "step": state["step"],
            }
        reserved = (
            plan["limits"]["maximum_report_bytes"]
            + plan["limits"]["verification_write_bytes_reserved"]
        )
        if _private_tree_bytes(private_root) + reserved > plan["limits"][
            "maximum_private_write_bytes"
        ]:
            raise ValueError("distributed CPU private-write cap exceeded")

    if paused_states != {
        "ddp": {"records_seen": 2, "sequence_cursor": 2, "status": "paused", "step": 1},
        "fsdp_full_shard": {
            "records_seen": 2,
            "sequence_cursor": 2,
            "status": "paused",
            "step": 1,
        },
    }:
        raise ValueError("distributed CPU paused boundaries changed")
    load_accounting = _aggregate_load_audits(load_audits, plan["limits"])
    strategy_audits = {
        "ddp": _audit_strategy(
            strategy="ddp",
            reference=_repository_path(repository, plan["outputs"]["ddp_reference"]),
            resumed=_repository_path(repository, plan["outputs"]["ddp_resumed"]),
            plan=plan,
        ),
        "fsdp_full_shard": _audit_strategy(
            strategy="fsdp_full_shard",
            reference=_repository_path(repository, plan["outputs"]["fsdp_reference"]),
            resumed=_repository_path(repository, plan["outputs"]["fsdp_resumed"]),
            plan=plan,
        ),
    }
    _require_final_private_inventory(private_root, logs_root, plan)
    runtime_seconds = time.monotonic() - started
    if runtime_seconds > plan["limits"]["maximum_local_runtime_minutes"] * 60:
        raise TimeoutError("distributed CPU preflight runtime cap exceeded")
    private_write_bytes = _private_tree_bytes(private_root)
    report = _report(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        historical=historical,
        launch_logs=launch_logs,
        load_accounting=load_accounting,
        strategy_audits=strategy_audits,
        private_write_bytes=private_write_bytes,
        runtime_seconds=runtime_seconds,
    )
    report_path = _repository_path(repository, str(plan["outputs"]["report"]))
    _require_safe_parent_chain(repository, report_path)
    report_bytes = _write_exclusive_private_json(
        report_path,
        report,
        maximum_bytes=plan["limits"]["maximum_report_bytes"],
    )
    total_reserved = (
        private_write_bytes
        + report_bytes
        + plan["limits"]["verification_write_bytes_reserved"]
    )
    if total_reserved > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("distributed CPU final private-write cap exceeded")
    return report


__all__ = [
    "EXPECTED_AUTHORIZATIONS",
    "EXPECTED_AUTHORIZATION_BOUNDARY",
    "EXPECTED_CODE_PATHS",
    "EXPECTED_EXECUTION",
    "EXPECTED_FIXED_BINDINGS",
    "EXPECTED_HISTORICAL_V0_1",
    "EXPECTED_INPUT",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUTS",
    "EXPECTED_PER_RANK_LOAD_AUDIT",
    "EXPECTED_PLAN_PATH",
    "EXPECTED_TRAINING",
    "LAUNCH_SPECS",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "REPORT_SCHEMA_VERSION",
    "REPORT_STATUS",
    "assert_raw_text_free",
    "canonical_json",
    "diagnose_historical_v0_1",
    "json_file_bytes",
    "load_plan",
    "run_distributed_cpu_preflight",
    "sha256_file",
    "sha256_value",
    "strict_json_bytes",
    "validate_plan",
]
