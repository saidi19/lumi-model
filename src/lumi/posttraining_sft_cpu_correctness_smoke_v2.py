"""Bounded v0.2 tiny-CPU SFT correctness smoke orchestration and audit.

This successor consumes only the four-record prefix approved by the completed
metadata preflight.  It runs one uninterrupted two-step path and one one-step
pause plus fresh-process resume path, then requires exact equality of every
stable result.  It cannot authorize production SFT.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import posttraining_sft_runtime_v3 as runtime
from .configuration_lumi import LumiConfig
from .modeling_lumi import LumiForCausalLM

PLAN_SCHEMA_VERSION = "lumi.sft-cpu-correctness-smoke-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-cpu-correctness-smoke-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-cpu-correctness-smoke-verification.v0.2"
PLAN_STATUS = "authorized_v0_2_tiny_cpu_correctness_smoke"
REPORT_STATUS = "PASS_v0_2_tiny_cpu_sft_uninterrupted_equals_fresh_resume"
VERIFICATION_STATUS = "PASS_verified_v0_2_tiny_cpu_sft_fresh_resume_exact"

EXPECTED_AUTHORIZATION_TEXT = "next step shuru de"
EXPECTED_AUTHORIZATION_BYTES = 18
EXPECTED_AUTHORIZATION_SHA256 = (
    "8b57ac832e9b8ce0b1970371aa2e06d4053d9ad24a6168ffa8bdc3e94d1720f4"
)
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-cpu-correctness-smoke-v0.2.json")

EXPECTED_AUTHORIZATION = {
    "cpu_sft_correctness_smoke_authorized": True,
    "dataset_or_shard_mutation_authorized": False,
    "encoded_bounded_prefix_read_authorized": True,
    "external_api_authorized": False,
    "full_4b_checkpoint_restore_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "production_sft_authorized": False,
    "promotion_or_seal_authorized": False,
    "registry_or_contract_mutation_authorized": False,
    "remote_or_instance_action_authorized": False,
    "tiny_cpu_training_authorized": True,
    "unrelated_action_authorized": False,
}

EXPECTED_LIMITS = {
    "maximum_additional_private_write_bytes": 268_435_456,
    "maximum_candidate_row_reads": 0,
    "maximum_complete_encoded_shard_hash_reads": 0,
    "maximum_dataset_loads": 3,
    "maximum_encoded_binary_bytes_read": 393_120,
    "maximum_encoded_index_bytes_read": 2_100,
    "maximum_encoded_index_rows_parsed": 12,
    "maximum_encoded_record_file_reads": 12,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_jsonl_row_reads": 0,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 30,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_tiny_optimizer_updates": 4,
    "maximum_training_sample_consumptions": 8,
}

EXPECTED_MODEL = {
    "head_dimension": 8,
    "hidden_size": 16,
    "intermediate_size": 32,
    "maximum_positions": 4096,
    "number_attention_heads": 2,
    "number_key_value_heads": 1,
    "number_layers": 1,
    "torch_seed": 11,
    "vocabulary_size": 65_536,
}

EXPECTED_TRAINING = {
    "attention": "cpu_eager",
    "batch_size": 1,
    "checkpoint_steps": 1,
    "contiguous_train_prefix_records": 4,
    "cursor_seed": 2027,
    "dataset_loads": 3,
    "gradient_accumulation_steps": 2,
    "learning_rate": 0.0001,
    "maximum_gradient_norm": 1.0,
    "minimum_learning_rate_ratio": 0.1,
    "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
    "pad_to_multiple_of": 8,
    "schedule": "cosine",
    "target_optimizer_steps": 2,
    "warmup_steps": 0,
    "weight_decay": 0.01,
}

EXPECTED_SOURCE_CHECKPOINT = {
    "config.json": "669ad570d0f3c06613bd9348f93bfcf00ea593f93be3024ce4a924ad01155469",
    "configuration_lumi.py": (
        "ab56700d837e314cad4a4493039f699d4f0a0239dfea16e893e1279b261c05aa"
    ),
    "generation_config.json": (
        "057225d453a6f0d3fe22d9e4de300a3775eb45b96bf00bc245aaf7c9973299ff"
    ),
    "manifest.sha256": (
        "2aeb1e3352b7b31c3ab16babee514ba69166626c4292ae5ed594e7a42fe3fa91"
    ),
    "model.safetensors": (
        "a1779301e8b0eef3e648672baf55a8e090b2f1d4e5a232ebe0be93863a2f596a"
    ),
    "modeling_lumi.py": (
        "bf7b908d282a752132918dddad648fd8435d3f8ad9db41eac1c2e64715926fad"
    ),
}

EXPECTED_SOURCE_CHECKPOINT_CONTRACT = {
    "config_semantics": {
        "attention_bias": False,
        "bos_token_id": 1,
        "doc_token_id": 3,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "max_position_embeddings": 4096,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "use_cache": False,
        "vocab_size": 65_536,
    },
    "exact_file_sha256": EXPECTED_SOURCE_CHECKPOINT,
    "fresh_per_attempt": True,
    "inventory": sorted(EXPECTED_SOURCE_CHECKPOINT),
    "private_directory_mode": "0700",
    "private_file_mode": "0600",
    "private_files_nlink_one_and_owned": True,
}

EXPECTED_INPUT = {
    "encoded_directory": (
        "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/encoded"
    ),
    "encoded_dataset_sha256": (
        "fcfc9e227f0d0b067062293e1adcd99d9f85406dcd50eb1363d408e287abf7c0"
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
    "record_hash_sequence_sha256": (
        "8bd0fed685a663b048169a83e86e645806c67545ce1f08974351624ad5e64070"
    ),
    "records": 329_200,
    "seal_internal_sha256": (
        "8c575ab8f6acafe326ec5f19c3c750c86ebe42c5f851513eaee28da8751e05b7"
    ),
    "supervised_tokens": 12_794_510,
    "tokens": 35_892_391,
}

EXPECTED_OUTPUTS = {
    "attempt_marker": (
        "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/"
        "cpu-correctness-smoke-started.json"
    ),
    "private_root": "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private",
    "report": "evidence/lumi-4b-sft-cpu-correctness-smoke-v0.2.json",
    "resumed": (
        "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/resumed"
    ),
    "source_checkpoint": (
        "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/source"
    ),
    "uninterrupted": (
        "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private/uninterrupted"
    ),
    "verification": (
        "evidence/lumi-4b-sft-cpu-correctness-smoke-verification-v0.2.json"
    ),
}

EXPECTED_RUNTIME_CONTRACT = {
    "active_runtime_profile": "unbuffered_descriptor_bound_prefix_v0.3",
    "direct_successor_of_rejected_v2": True,
    "encoded_selected_stream_buffer_bytes": 0,
    "metadata_inventory_checked_before_selected_file_open": True,
    "preflight_v2_code_and_evidence_preserved_byte_for_byte": True,
    "preflight_v2_runtime_selection_superseded_only_for_actual_smoke": True,
    "rejected_runtime_profile": "buffered_descriptor_bound_prefix_v0.2",
    "rejection_reason_code": "buffered_reader_readahead_invalidates_read_caps",
    "resume_marker_created_before_resume_input_reads": True,
    "same_descriptor_selected_bytes_retained": True,
}

EXPECTED_FIXED_BINDINGS = {
    "active_cpu_runner_v3": {
        "path": "scripts/run_sft_cpu_canary_v3.py",
        "sha256": (
            "f4d3b9bf19fe8d1ef9a6feadf38157b4675e4d772a2b7b112641e95238388bf3"
        ),
    },
    "active_runtime_test_v3": {
        "path": "tests/test_posttraining_sft_runtime_v3.py",
        "sha256": (
            "accbd634011576211b4df790606de50ce90f631b9ff69385d581f47b4ed9f809"
        ),
    },
    "active_runtime_v3": {
        "path": "src/lumi/posttraining_sft_runtime_v3.py",
        "sha256": (
            "335bf09b5ca2df962848beb8dd50cf59114924deb44564c9fcf65d7d0f7b0b8d"
        ),
    },
    "encoded_manifest": {
        "path": (
            "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/"
            "encoded/manifest.json"
        ),
        "sha256": EXPECTED_INPUT["manifest_file_sha256"],
    },
    "encoded_shard_receipt": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-v0.2.json",
        "sha256": (
            "3daf1a936177cbd9092c842fdbdd171e61c06a09e12cdf54b715230bb58040cb"
        ),
    },
    "encoded_shard_verification": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-verification-v0.2.json",
        "sha256": (
            "701c8b2e9659526ec3f0a780f099bff32d22c83cb6b8d0f404460934d623c86a"
        ),
    },
    "final_base_binding": {
        "path": "configs/lumi-4b-final-base-binding-v0.1.json",
        "sha256": (
            "1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172"
        ),
    },
    "rejected_buffered_runtime_v2": {
        "path": "src/lumi/posttraining_sft_runtime_v2.py",
        "sha256": (
            "eefdf416fc88461f5986104de75447f1b372759faa17176614e69bf157308bd5"
        ),
    },
    "rejected_buffered_runtime_test_v2": {
        "path": "tests/test_posttraining_sft_runtime_v2.py",
        "sha256": (
            "4db2f98a67de8b36d9dc8361278118c80578a9fcee871ab0cd2567a746311075"
        ),
    },
    "model_implementation": {
        "path": "src/lumi/modeling_lumi.py",
        "sha256": (
            "bf7b908d282a752132918dddad648fd8435d3f8ad9db41eac1c2e64715926fad"
        ),
    },
    "preflight_implementation_v2": {
        "path": "src/lumi/posttraining_sft_cpu_smoke_preflight_v2.py",
        "sha256": (
            "a09b30258a90899e24e58abce34ee24acdef792639758ad2751787546e2a3c8b"
        ),
    },
    "preflight_plan": {
        "path": "configs/lumi-4b-sft-cpu-smoke-preflight-v0.2.json",
        "sha256": (
            "60d6e4aff29d0ad17ab8cb62196f45ba76dbb70560efc727414b278d617377f4"
        ),
    },
    "preflight_report": {
        "path": "evidence/lumi-4b-sft-cpu-smoke-preflight-v0.2.json",
        "sha256": (
            "56b9fa2a8d92084df29a944963f125798dc21e22697f161c60258458a2289e83"
        ),
    },
    "preflight_runner_v2": {
        "path": "scripts/preflight_sft_cpu_smoke_v2.py",
        "sha256": (
            "38b0b6602806109ff29c29b51668671b547b3c6422410894bf54f81184721d78"
        ),
    },
    "preflight_test_v2": {
        "path": "tests/test_sft_cpu_smoke_preflight_v2.py",
        "sha256": (
            "4ac28bd895903c7416a5dcec9ea5e3ed2c6dd57c1c7baa13188ea5c90d20dc7e"
        ),
    },
    "preflight_verification": {
        "path": "evidence/lumi-4b-sft-cpu-smoke-preflight-verification-v0.2.json",
        "sha256": (
            "1f45ddae6917c4cf6486204982faf26eafe732362b224a14571db8c1b6df41a6"
        ),
    },
    "preflight_verifier_v2": {
        "path": "scripts/verify_sft_cpu_smoke_preflight_v2.py",
        "sha256": (
            "572ab308cb4f1609c6461b99e28932fcc55652338512c0e246a4e6d795e8e6ef"
        ),
    },
    "successor_seal": {
        "path": "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.2.json",
        "sha256": (
            "1c78bcea6016b4903c4b6262f24cfcc718b6a88b68a2d1c1e3fc13bbb8ba0ce9"
        ),
    },
    "transitive_configuration": {
        "path": "src/lumi/configuration_lumi.py",
        "sha256": (
            "ab56700d837e314cad4a4493039f699d4f0a0239dfea16e893e1279b261c05aa"
        ),
    },
    "transitive_pilot_training": {
        "path": "src/lumi/pilot_training.py",
        "sha256": (
            "d59707436f50775152f841d3c1290f4d6edb9abfdbe65eaa7263063f0d18102a"
        ),
    },
    "transitive_posttraining_shards": {
        "path": "src/lumi/posttraining_shards.py",
        "sha256": (
            "d8480fc71c061b0c3ae5a4228d754044da8417ead86bd0033b997566f2645504"
        ),
    },
    "rejected_buffered_cpu_runner_v2": {
        "path": "scripts/run_sft_cpu_canary_v2.py",
        "sha256": (
            "f72bda4d74a7b9b9a84423ce381cdca83c0928a7bfa0262e34c427d75db374f8"
        ),
    },
}

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_sft_cpu_correctness_smoke_v2.py",
    "orchestrator": "scripts/run_sft_cpu_correctness_smoke_v2.py",
    "test": "tests/test_sft_cpu_correctness_smoke_v2.py",
    "verifier": "scripts/verify_sft_cpu_correctness_smoke_v2.py",
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_MARKER = {
    "schema_version": "lumi.sft-cpu-correctness-smoke-started.v0.2",
    "status": "started_no_retry",
}
_STABLE_STATE_FIELDS = (
    "best_loss",
    "last_loss",
    "latest_checkpoint",
    "records_seen",
    "run_fingerprint",
    "run_specification",
    "schema_version",
    "status",
    "step",
    "supervised_tokens_seen",
    "tokens_seen",
)
_TIMING_METRIC_FIELDS = frozenset({"elapsed_seconds", "tokens_per_second"})
_MAX_EVIDENCE_FILE_BYTES = 65_536
_TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES = 1_048_576
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
_CHECKPOINT_FILE_INVENTORY = {
    "config.json",
    "configuration_lumi.py",
    "generation_config.json",
    "manifest.sha256",
    "model.safetensors",
    "modeling_lumi.py",
    runtime.SFT_CANARY_STATE_FILENAME,
    "trainer.pt",
}


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in CPU correctness-smoke input")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("CPU correctness-smoke JSON is not UTF-8") from error


def _strict_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) is float and actual == expected
    return type(actual) is type(expected) and actual == expected


def assert_raw_text_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError("raw training text field found in CPU-smoke evidence")
            assert_raw_text_free(child)
    elif isinstance(value, list):
        for child in value:
            assert_raw_text_free(child)


def _read_json(path: Path) -> dict[str, Any]:
    payload = strict_json_bytes(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_bound_file(
    root: Path,
    name: str,
    binding: Mapping[str, Any],
) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"CPU-smoke binding shape changed: {name}")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"CPU-smoke binding is invalid: {name}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"CPU-smoke binding path is invalid: {name}")
    if sha256_file(candidate) != digest:
        raise ValueError(f"CPU-smoke binding changed: {name}")
    return candidate


def validate_plan(plan: Mapping[str, Any], *, repository_root: Path) -> None:
    expected_top = {
        "authorization",
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
        "bindings",
        "input",
        "limits",
        "model",
        "observed_at",
        "outputs",
        "production_eligible",
        "raw_training_text_retained",
        "runtime_contract",
        "schema_version",
        "source_checkpoint_contract",
        "status",
        "training",
    }
    if (
        set(plan) != expected_top
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("observed_at") != EXPECTED_OBSERVED_AT
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or not _strict_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION)
        or not _strict_equal(plan.get("limits"), EXPECTED_LIMITS)
        or not _strict_equal(plan.get("model"), EXPECTED_MODEL)
        or not _strict_equal(plan.get("training"), EXPECTED_TRAINING)
        or not _strict_equal(plan.get("input"), EXPECTED_INPUT)
        or not _strict_equal(plan.get("outputs"), EXPECTED_OUTPUTS)
        or not _strict_equal(plan.get("runtime_contract"), EXPECTED_RUNTIME_CONTRACT)
        or not _strict_equal(
            plan.get("source_checkpoint_contract"),
            EXPECTED_SOURCE_CHECKPOINT_CONTRACT,
        )
        or plan.get("production_eligible") is not False
        or plan.get("raw_training_text_retained") is not False
    ):
        raise ValueError("CPU correctness-smoke plan contract changed")
    authorization_payload = EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
    if (
        len(authorization_payload) != EXPECTED_AUTHORIZATION_BYTES
        or hashlib.sha256(authorization_payload).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError("compiled CPU-smoke authorization binding changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != (
        set(EXPECTED_FIXED_BINDINGS) | set(EXPECTED_CODE_PATHS)
    ):
        raise ValueError("CPU correctness-smoke binding inventory changed")
    for name, expected in EXPECTED_FIXED_BINDINGS.items():
        if not _strict_equal(bindings.get(name), expected):
            raise ValueError(f"CPU correctness-smoke fixed binding changed: {name}")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"CPU correctness-smoke code binding changed: {name}")
    for name, binding in bindings.items():
        assert isinstance(binding, Mapping)
        _resolve_bound_file(repository_root, str(name), binding)


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    if plan_path != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("CPU correctness-smoke plan path changed")
    plan = _read_json(plan_path)
    validate_plan(plan, repository_root=root)
    return plan


def _private_write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_file_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short CPU-smoke evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
    ):
        raise ValueError("CPU-smoke evidence metadata is unsafe")


def _private_evidence_size(path: Path) -> int:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
    ):
        raise ValueError("CPU-smoke evidence metadata is unsafe")
    return info.st_size


def _create_tiny_source_checkpoint(path: Path) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to reuse CPU-smoke source checkpoint: {path}")
    path.mkdir(mode=0o700)
    torch.manual_seed(EXPECTED_MODEL["torch_seed"])
    model = LumiForCausalLM(
        LumiConfig(
            vocab_size=EXPECTED_MODEL["vocabulary_size"],
            hidden_size=EXPECTED_MODEL["hidden_size"],
            intermediate_size=EXPECTED_MODEL["intermediate_size"],
            num_hidden_layers=EXPECTED_MODEL["number_layers"],
            num_attention_heads=EXPECTED_MODEL["number_attention_heads"],
            num_key_value_heads=EXPECTED_MODEL["number_key_value_heads"],
            head_dim=EXPECTED_MODEL["head_dimension"],
            max_position_embeddings=EXPECTED_MODEL["maximum_positions"],
        )
    )
    model.save_pretrained(path, safe_serialization=True)
    runtime._harden_private_tree(path)  # noqa: SLF001
    manifest_sha256 = runtime._write_checksum_manifest(path)  # noqa: SLF001
    runtime._harden_private_tree(path)  # noqa: SLF001
    runtime._fsync_tree(path)  # noqa: SLF001
    runtime._verify_checksum_manifest(  # noqa: SLF001
        path,
        expected_manifest_sha256=manifest_sha256,
        require_private=True,
    )
    observed_names = {child.name for child in path.iterdir()}
    if observed_names != set(EXPECTED_SOURCE_CHECKPOINT):
        raise ValueError("fresh CPU-smoke source checkpoint inventory changed")
    for name, expected_sha256 in EXPECTED_SOURCE_CHECKPOINT.items():
        candidate = path / name
        info = candidate.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise ValueError(f"fresh CPU-smoke source metadata changed: {name}")
        if sha256_file(candidate) != expected_sha256:
            raise ValueError(f"fresh CPU-smoke source checkpoint changed: {name}")
    if manifest_sha256 != EXPECTED_SOURCE_CHECKPOINT["manifest.sha256"]:
        raise ValueError("fresh CPU-smoke source manifest oracle changed")
    root_info = path.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != 0o700
        or root_info.st_uid != os.geteuid()
    ):
        raise ValueError("fresh CPU-smoke source directory metadata changed")
    source_config = _read_json(path / "config.json")
    semantics = EXPECTED_SOURCE_CHECKPOINT_CONTRACT["config_semantics"]
    if any(source_config.get(key) != value for key, value in semantics.items()):
        raise ValueError("fresh CPU-smoke source config semantics changed")
    return manifest_sha256


def runner_commands(
    *,
    plan: Mapping[str, Any],
    repository_root: Path,
    source_manifest_sha256: str,
) -> list[list[str]]:
    bindings = plan["bindings"]
    runner = repository_root / bindings["active_cpu_runner_v3"]["path"]
    encoded = repository_root / plan["input"]["encoded_directory"]
    source = repository_root / plan["outputs"]["source_checkpoint"]
    common = [
        sys.executable,
        str(runner),
        "--encoded-shards",
        str(encoded),
        "--encoded-shards-inventory-sha256",
        str(plan["input"]["inventory_sha256"]),
        "--encoded-shards-manifest-file-sha256",
        str(plan["input"]["manifest_file_sha256"]),
        "--encoded-shards-manifest-sha256",
        str(plan["input"]["manifest_internal_sha256"]),
        "--encoded-shards-metadata-inventory-sha256",
        str(plan["input"]["metadata_inventory_sha256"]),
        "--encoded-shards-seal-sha256",
        str(plan["input"]["seal_internal_sha256"]),
        "--encoded-shards-record-limit",
        str(plan["training"]["contiguous_train_prefix_records"]),
        "--source-checkpoint",
        str(source),
        "--source-checkpoint-manifest-sha256",
        source_manifest_sha256,
        "--target-steps",
        str(plan["training"]["target_optimizer_steps"]),
        "--batch-size",
        str(plan["training"]["batch_size"]),
        "--gradient-accumulation-steps",
        str(plan["training"]["gradient_accumulation_steps"]),
        "--learning-rate",
        str(plan["training"]["learning_rate"]),
        "--weight-decay",
        str(plan["training"]["weight_decay"]),
        "--maximum-gradient-norm",
        str(plan["training"]["maximum_gradient_norm"]),
        "--minimum-learning-rate-ratio",
        str(plan["training"]["minimum_learning_rate_ratio"]),
        "--optimizer-profile",
        str(plan["training"]["optimizer_profile"]),
        "--pad-to-multiple-of",
        str(plan["training"]["pad_to_multiple_of"]),
        "--checkpoint-steps",
        str(plan["training"]["checkpoint_steps"]),
        "--warmup-steps",
        str(plan["training"]["warmup_steps"]),
        "--seed",
        str(plan["training"]["cursor_seed"]),
    ]
    uninterrupted = repository_root / plan["outputs"]["uninterrupted"]
    resumed = repository_root / plan["outputs"]["resumed"]
    return [
        [*common, "--output", str(uninterrupted), "--stop-after-step", "2"],
        [*common, "--output", str(resumed), "--stop-after-step", "1"],
        [
            *common,
            "--output",
            str(resumed),
            "--stop-after-step",
            "2",
            "--resume",
        ],
    ]


def _stable_metrics(path: Path) -> list[dict[str, Any]]:
    metrics = []
    for line in path.read_bytes().splitlines():
        value = strict_json_bytes(line)
        if not isinstance(value, dict):
            raise ValueError("CPU-smoke metric row is not an object")
        metrics.append(
            {key: child for key, child in value.items() if key not in _TIMING_METRIC_FIELDS}
        )
    return metrics


def _state(path: Path) -> dict[str, Any]:
    value = _read_json(path / runtime.SFT_CANARY_STATE_FILENAME)
    if set(value) != set(_STABLE_STATE_FIELDS):
        raise ValueError("CPU-smoke state field inventory changed")
    return {key: value[key] for key in _STABLE_STATE_FIELDS}


def _model_state(path: Path) -> dict[str, torch.Tensor]:
    model = LumiForCausalLM.from_pretrained(path, local_files_only=True)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


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
            and all(_nested_exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _tensor_mapping_exact(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return set(left) == set(right) and all(
        _nested_exact(left[name], right[name]) for name in left
    )


def _semantic_tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(child: Any) -> None:
        if isinstance(child, torch.Tensor):
            tensor = child.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
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
            digest.update(f"{type(child).__name__}\0".encode())
            for item in child:
                update(item)
        else:
            digest.update(b"scalar\0")
            digest.update(canonical_json(child))

    update(value)
    return digest.hexdigest()


def _checkpoint_pair_audit(
    uninterrupted: Path,
    resumed: Path,
) -> dict[str, Any]:
    step_manifest_hashes: dict[str, str] = {}
    for step in (1, 2):
        left = uninterrupted / f"checkpoint-{step:08d}"
        right = resumed / f"checkpoint-{step:08d}"
        _require_exact_checkpoint_inventory(left)
        _require_exact_checkpoint_inventory(right)
        runtime.verify_sft_canary_checkpoint(left)
        runtime.verify_sft_canary_checkpoint(right)
        left_hash = sha256_file(left / "manifest.sha256")
        right_hash = sha256_file(right / "manifest.sha256")
        if left_hash != right_hash:
            raise ValueError(f"CPU-smoke checkpoint-{step} manifest differs")
        step_manifest_hashes[str(step)] = left_hash

    left_final = uninterrupted / "checkpoint-00000002"
    right_final = resumed / "checkpoint-00000002"
    left_model = _model_state(left_final)
    right_model = _model_state(right_final)
    if not _tensor_mapping_exact(left_model, right_model):
        raise ValueError("CPU-smoke final model tensors differ")
    left_trainer = torch.load(left_final / "trainer.pt", map_location="cpu", weights_only=True)
    right_trainer = torch.load(right_final / "trainer.pt", map_location="cpu", weights_only=True)
    if not _nested_exact(left_trainer, right_trainer):
        raise ValueError("CPU-smoke final optimizer or RNG state differs")
    return {
        "final_model_semantic_sha256": _semantic_tensor_digest(left_model),
        "final_optimizer_semantic_sha256": _semantic_tensor_digest(
            left_trainer["optimizer"]
        ),
        "final_rng_semantic_sha256": _semantic_tensor_digest(left_trainer["rng_state"]),
        "step_checkpoint_manifest_sha256": step_manifest_hashes,
    }


def _require_exact_checkpoint_inventory(path: Path) -> None:
    entries = list(path.iterdir())
    if {entry.name for entry in entries} != _CHECKPOINT_FILE_INVENTORY:
        raise ValueError("CPU-smoke checkpoint inventory changed")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("CPU-smoke checkpoint contains a non-regular entry")


def _private_tree_bytes(root: Path) -> int:
    root_info = root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != 0o700
        or root_info.st_uid != os.geteuid()
    ):
        raise ValueError("CPU-smoke private-root metadata changed")
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("CPU-smoke private tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
                raise ValueError("CPU-smoke private directory metadata changed")
        elif stat.S_ISREG(info.st_mode):
            if (
                stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
            ):
                raise ValueError("CPU-smoke private file metadata changed")
            total += info.st_size
        else:
            raise ValueError("CPU-smoke private tree contains a special file")
    return total


def _require_exact_output_inventory(private_root: Path) -> None:
    expected_root = {
        "cpu-correctness-smoke-started.json",
        "resumed",
        "source",
        "uninterrupted",
    }
    if {path.name for path in private_root.iterdir()} != expected_root:
        raise ValueError("CPU-smoke private-root inventory changed")
    common = {
        "checkpoint-00000001",
        "checkpoint-00000002",
        runtime.SFT_CANARY_METRICS_FILENAME,
        runtime.SFT_CANARY_STARTED_FILENAME,
        runtime.SFT_CANARY_STATE_FILENAME,
    }
    uninterrupted = private_root / "uninterrupted"
    resumed = private_root / "resumed"
    source = private_root / "source"
    if {path.name for path in source.iterdir()} != set(EXPECTED_SOURCE_CHECKPOINT):
        raise ValueError("CPU-smoke source inventory changed")
    if any(not path.is_file() or path.is_symlink() for path in source.iterdir()):
        raise ValueError("CPU-smoke source inventory is not exact flat regular files")
    if {path.name for path in uninterrupted.iterdir()} != common:
        raise ValueError("CPU-smoke uninterrupted inventory changed")
    if {path.name for path in resumed.iterdir()} != (
        common | {runtime.SFT_CANARY_RESUME_STARTED_FILENAME}
    ):
        raise ValueError("CPU-smoke resumed inventory changed")
    if _read_json(private_root / "cpu-correctness-smoke-started.json") != _ATTEMPT_MARKER:
        raise ValueError("CPU-smoke global started marker changed")
    runtime_started = {
        "schema_version": "lumi.sft-cpu-canary-started.v0.3",
        "status": "started_no_automatic_retry",
    }
    if _read_json(uninterrupted / runtime.SFT_CANARY_STARTED_FILENAME) != runtime_started:
        raise ValueError("CPU-smoke uninterrupted started marker changed")
    if _read_json(resumed / runtime.SFT_CANARY_STARTED_FILENAME) != runtime_started:
        raise ValueError("CPU-smoke resumed started marker changed")
    resume_started = {
        "schema_version": "lumi.sft-cpu-canary-resume-started.v0.1",
        "status": "started_no_automatic_retry",
        "stop_after_step": 2,
        "target_steps": 2,
    }
    if _read_json(resumed / runtime.SFT_CANARY_RESUME_STARTED_FILENAME) != resume_started:
        raise ValueError("CPU-smoke resume-started marker changed")


def audit_outputs(
    *,
    plan: Mapping[str, Any],
    repository_root: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    uninterrupted = repository_root / plan["outputs"]["uninterrupted"]
    resumed = repository_root / plan["outputs"]["resumed"]
    left_state = _state(uninterrupted)
    right_state = _state(resumed)
    if not _strict_equal(left_state, right_state):
        raise ValueError("CPU-smoke uninterrupted and resumed final states differ")
    if (
        left_state["status"] != "complete"
        or left_state["step"] != 2
        or left_state["records_seen"] != 4
    ):
        raise ValueError("CPU-smoke final state is not the exact two-step contract")
    left_metrics = _stable_metrics(
        uninterrupted / runtime.SFT_CANARY_METRICS_FILENAME
    )
    right_metrics = _stable_metrics(resumed / runtime.SFT_CANARY_METRICS_FILENAME)
    if not _strict_equal(left_metrics, right_metrics) or len(left_metrics) != 2:
        raise ValueError("CPU-smoke stable metric sequences differ")
    checkpoint_audit = _checkpoint_pair_audit(uninterrupted, resumed)
    selected_prefix = left_state["run_specification"]["data"]
    read_audit = selected_prefix.get("read_audit")
    if (
        selected_prefix.get("record_limit") != 4
        or selected_prefix.get("selection_scope") != "contiguous_prefix_v0.1"
        or selected_prefix.get("metadata_inventory_sha256")
        != plan["input"]["metadata_inventory_sha256"]
        or selected_prefix.get("selected_prefix_sha256") in {None, "", "0" * 64}
        or not isinstance(read_audit, Mapping)
        or set(read_audit)
        != {
            "binary_bytes_read",
            "encoded_index_row_reads",
            "file_opens",
            "fully_hashed_shards",
            "index_bytes_read",
            "metadata_inventory_verified_before_selected_content_read",
            "opened_shards",
            "records_loaded",
            "unbuffered_selected_content_reads",
        }
        or read_audit.get("encoded_index_row_reads") != 4
        or read_audit.get("file_opens") != 4
        or read_audit.get("fully_hashed_shards") != 0
        or read_audit.get("metadata_inventory_verified_before_selected_content_read")
        is not True
        or read_audit.get("opened_shards") != 1
        or read_audit.get("records_loaded") != 4
        or read_audit.get("unbuffered_selected_content_reads") is not True
        or type(read_audit.get("binary_bytes_read")) is not int
        or not 1 <= read_audit["binary_bytes_read"] <= 131_040
        or type(read_audit.get("index_bytes_read")) is not int
        or not 1 <= read_audit["index_bytes_read"] <= 700
    ):
        raise ValueError("CPU-smoke selected-prefix run binding changed")
    private_root = repository_root / plan["outputs"]["private_root"]
    _require_exact_output_inventory(private_root)
    private_bytes = _private_tree_bytes(private_root)
    if (
        private_bytes > plan["limits"]["maximum_additional_private_write_bytes"]
    ):
        raise ValueError("CPU-smoke resource cap exceeded")
    return {
        "checkpoint_audit": checkpoint_audit,
        "final_state": left_state,
        "loss_sequence": [metric["loss"] for metric in left_metrics],
        "resource_accounting": {
            "dataset_loads": 3,
            "encoded_binary_bytes_read": read_audit["binary_bytes_read"] * 3,
            "encoded_binary_bytes_read_upper_bound": 393_120,
            "encoded_index_bytes_read": read_audit["index_bytes_read"] * 3,
            "encoded_index_bytes_read_upper_bound": 2_100,
            "encoded_index_rows_parsed": 12,
            "encoded_record_file_reads": 12,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "private_write_bytes_before_report": private_bytes,
            "retries_after_started_pass": 0,
            "tiny_optimizer_updates": 4,
            "training_sample_consumptions": 8,
        },
        "selected_prefix_sha256": selected_prefix["selected_prefix_sha256"],
        "source_checkpoint_manifest_sha256": source_manifest_sha256,
        "stable_metrics_sha256": sha256_value(left_metrics),
    }


def _build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    audit: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_state = audit["final_state"]
    report = {
        "authorization_boundary": {
            "dataset_or_shards_mutated": False,
            "external_api_or_download_used": False,
            "full_4b_checkpoint_restored": False,
            "gpu_used": False,
            "inference_calls": 0,
            "network_bytes": 0,
            "production_sft_started": False,
            "promotion_or_seal_performed": False,
            "remote_or_instance_action": False,
        },
        "bindings": {
            "encoded_manifest_file_sha256": plan["input"]["manifest_file_sha256"],
            "encoded_manifest_internal_sha256": plan["input"]["manifest_internal_sha256"],
            "encoded_metadata_inventory_sha256": plan["input"]["metadata_inventory_sha256"],
            "encoded_inventory_sha256": plan["input"]["inventory_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "seal_internal_sha256": plan["input"]["seal_internal_sha256"],
            "source_checkpoint_manifest_sha256": audit[
                "source_checkpoint_manifest_sha256"
            ],
        },
        "checks": {
            "assistant_and_tool_only_masking_checked_on_each_load": True,
            "bounded_prefix_digest_bound_in_run_fingerprint": True,
            "data_cursor_exact": True,
            "final_model_tensors_exact": True,
            "final_optimizer_state_exact": True,
            "final_rng_state_exact": True,
            "loss_sequence_exact": True,
            "metadata_inventory_rechecked_before_each_content_load": True,
            "private_output_modes_links_and_owner_exact": True,
            "resume_run_fingerprint_exact": True,
            "stable_metric_sequence_exact": True,
            "step_1_and_step_2_checkpoint_manifests_exact": True,
        },
        "execution": {
            "fresh_resume_processes": 2,
            "runtime_seconds": elapsed_seconds,
            "uninterrupted_processes": 1,
        },
        "final_state": {
            "last_loss": final_state["last_loss"],
            "loss_sequence": audit["loss_sequence"],
            "records_seen": final_state["records_seen"],
            "run_fingerprint": final_state["run_fingerprint"],
            "status": final_state["status"],
            "step": final_state["step"],
            "supervised_tokens_seen": final_state["supervised_tokens_seen"],
            "tokens_seen": final_state["tokens_seen"],
        },
        "model": dict(plan["model"]),
        "outputs": {
            **dict(audit["checkpoint_audit"]),
            "selected_prefix_sha256": audit["selected_prefix_sha256"],
            "stable_metrics_sha256": audit["stable_metrics_sha256"],
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
        "resource_accounting": dict(audit["resource_accounting"]),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "timing_fields_excluded_from_deterministic_comparison": sorted(
            _TIMING_METRIC_FIELDS
        ),
        "training": dict(plan["training"]),
    }
    assert_raw_text_free(report)
    unsigned = dict(report)
    report["report_sha256"] = sha256_value(unsigned)
    return report


def _finalize_report_size(
    report: dict[str, Any],
    *,
    private_root_bytes: int,
) -> int:
    resources = report["resource_accounting"]
    report_bytes = 0
    for _ in range(16):
        resources["private_root_write_bytes"] = private_root_bytes
        resources["report_file_bytes"] = report_bytes
        resources["verification_file_bytes_reserved"] = _MAX_EVIDENCE_FILE_BYTES
        resources["transient_atomic_rewrite_reserve_bytes"] = (
            _TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES
        )
        resources["cumulative_private_write_bytes_upper_bound"] = (
            private_root_bytes
            + report_bytes
            + _MAX_EVIDENCE_FILE_BYTES
            + _TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES
        )
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        report["report_sha256"] = sha256_value(unsigned)
        observed = len(json_file_bytes(report))
        if observed == report_bytes:
            if observed > _MAX_EVIDENCE_FILE_BYTES:
                raise ValueError("CPU-smoke report exceeds its evidence-file bound")
            return observed
        report_bytes = observed
    raise RuntimeError("CPU-smoke report byte accounting did not converge")


def run_cpu_correctness_smoke(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, repository_root=root)
    started = time.monotonic()
    private_root = root / plan["outputs"]["private_root"]
    runtime._ensure_private_directory(private_root)  # noqa: SLF001
    if any(private_root.iterdir()):
        raise FileExistsError("CPU-smoke private attempt already started; retry forbidden")
    marker = root / plan["outputs"]["attempt_marker"]
    runtime._write_json_exclusive(marker, _ATTEMPT_MARKER)  # noqa: SLF001
    source = root / plan["outputs"]["source_checkpoint"]
    source_manifest_sha256 = _create_tiny_source_checkpoint(source)
    commands = runner_commands(
        plan=plan,
        repository_root=root,
        source_manifest_sha256=source_manifest_sha256,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(root / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    deadline = started + plan["limits"]["maximum_runtime_minutes"] * 60
    for command in commands:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("CPU-smoke global runtime cap exhausted")
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=remaining_seconds,
        )
    audit = audit_outputs(
        plan=plan,
        repository_root=root,
        source_manifest_sha256=source_manifest_sha256,
    )
    elapsed_seconds = time.monotonic() - started
    if elapsed_seconds > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("CPU-smoke global runtime cap exceeded")
    report = _build_report(
        plan=plan,
        plan_file_sha256=sha256_file(plan_path),
        audit=audit,
        elapsed_seconds=elapsed_seconds,
    )
    report_path = root / plan["outputs"]["report"]
    private_root_bytes = audit["resource_accounting"][
        "private_write_bytes_before_report"
    ]
    _finalize_report_size(report, private_root_bytes=private_root_bytes)
    projected_writes = report["resource_accounting"][
        "cumulative_private_write_bytes_upper_bound"
    ]
    if projected_writes > plan["limits"]["maximum_additional_private_write_bytes"]:
        raise ValueError("CPU-smoke cumulative private-write cap would be exceeded")
    if time.monotonic() > deadline:
        raise TimeoutError("CPU-smoke global runtime cap exceeded before report write")
    _private_write_exclusive(report_path, report)
    return report


def _validate_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> None:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    resources = report.get("resource_accounting")
    checks = report.get("checks")
    boundary = report.get("authorization_boundary")
    if (
        not isinstance(internal, str)
        or internal != sha256_value(unsigned)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("bindings", {}).get("plan_file_sha256") != plan_file_sha256
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(boundary, Mapping)
        or boundary.get("gpu_used") is not False
        or boundary.get("network_bytes") != 0
        or boundary.get("inference_calls") != 0
        or boundary.get("production_sft_started") is not False
        or not isinstance(resources, Mapping)
        or resources.get("dataset_loads") != 3
        or resources.get("encoded_index_rows_parsed") != 12
        or resources.get("encoded_record_file_reads") != 12
        or resources.get("encoded_binary_bytes_read", 10**18)
        > plan["limits"]["maximum_encoded_binary_bytes_read"]
        or resources.get("encoded_binary_bytes_read_upper_bound", 10**18)
        > plan["limits"]["maximum_encoded_binary_bytes_read"]
        or resources.get("encoded_index_bytes_read", 10**18)
        > plan["limits"]["maximum_encoded_index_bytes_read"]
        or resources.get("encoded_index_bytes_read_upper_bound", 10**18)
        > plan["limits"]["maximum_encoded_index_bytes_read"]
        or resources.get("report_file_bytes", 10**18) > _MAX_EVIDENCE_FILE_BYTES
        or resources.get("verification_file_bytes_reserved")
        != _MAX_EVIDENCE_FILE_BYTES
        or resources.get("transient_atomic_rewrite_reserve_bytes")
        != _TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES
        or resources.get("cumulative_private_write_bytes_upper_bound", 10**18)
        > plan["limits"]["maximum_additional_private_write_bytes"]
        or resources.get("tiny_optimizer_updates") != 4
        or resources.get("training_sample_consumptions") != 8
    ):
        raise ValueError("CPU correctness-smoke report changed or failed")
    assert_raw_text_free(report)


def _finalize_verification_size(
    verification: dict[str, Any],
    *,
    private_root_bytes: int,
    report_file_bytes: int,
) -> int:
    verification_bytes = 0
    for _ in range(16):
        verification["private_write_audit"] = {
            "cumulative_private_write_bytes_upper_bound": (
                private_root_bytes
                + report_file_bytes
                + verification_bytes
                + _TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES
            ),
            "private_root_write_bytes": private_root_bytes,
            "report_file_bytes": report_file_bytes,
            "transient_atomic_rewrite_reserve_bytes": (
                _TRANSIENT_ATOMIC_REWRITE_RESERVE_BYTES
            ),
            "verification_file_bytes": verification_bytes,
        }
        unsigned = dict(verification)
        unsigned.pop("verification_sha256", None)
        verification["verification_sha256"] = sha256_value(unsigned)
        observed = len(json_file_bytes(verification))
        if observed == verification_bytes:
            if observed > _MAX_EVIDENCE_FILE_BYTES:
                raise ValueError("CPU-smoke verification exceeds its file bound")
            return observed
        verification_bytes = observed
    raise RuntimeError("CPU-smoke verification byte accounting did not converge")


def verify_cpu_correctness_smoke(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, repository_root=root)
    plan_hash = sha256_file(plan_path)
    report_path = root / plan["outputs"]["report"]
    report_file_bytes = _private_evidence_size(report_path)
    report = _read_json(report_path)
    _validate_report(report, plan=plan, plan_file_sha256=plan_hash)
    if report_file_bytes != report["resource_accounting"]["report_file_bytes"]:
        raise ValueError("CPU-smoke report byte accounting changed")
    source = root / plan["outputs"]["source_checkpoint"]
    source_manifest = runtime._verify_checksum_manifest(  # noqa: SLF001
        source,
        expected_manifest_sha256=report["bindings"][
            "source_checkpoint_manifest_sha256"
        ],
        require_private=True,
    )
    audit = audit_outputs(
        plan=plan,
        repository_root=root,
        source_manifest_sha256=source_manifest,
    )
    expected_outputs = {
        **audit["checkpoint_audit"],
        "selected_prefix_sha256": audit["selected_prefix_sha256"],
        "stable_metrics_sha256": audit["stable_metrics_sha256"],
    }
    if not _strict_equal(report["outputs"], expected_outputs):
        raise ValueError("CPU-smoke report output digests changed")
    verification = {
        "checks": {
            "all_bound_input_and_code_hashes_match": True,
            "assistant_and_tool_only_masks_valid_on_all_loads": True,
            "checkpoint_manifests_verify_exact_inventory": True,
            "encoded_binary_and_index_caps_met": True,
            "final_model_tensor_names_and_values_exact": True,
            "final_optimizer_nested_state_exact": True,
            "final_rng_tensor_exact": True,
            "fresh_process_resume_used": True,
            "loss_and_stable_metric_sequences_exact": True,
            "no_4b_restore_inference_or_production_sft": True,
            "no_network_gpu_remote_or_instance_action": True,
            "private_modes_owner_link_and_write_cap_met": True,
            "raw_training_text_absent_from_new_evidence": True,
            "selected_prefix_and_metadata_identity_exact": True,
            "step_1_and_step_2_manifests_exact_across_paths": True,
        },
        "comparison": {
            **dict(audit["checkpoint_audit"]),
            "records_seen": audit["final_state"]["records_seen"],
            "run_fingerprint": audit["final_state"]["run_fingerprint"],
            "selected_prefix_sha256": audit["selected_prefix_sha256"],
            "stable_metrics_sha256": audit["stable_metrics_sha256"],
            "supervised_tokens_seen": audit["final_state"][
                "supervised_tokens_seen"
            ],
        },
        "plan_file_sha256": plan_hash,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": sha256_file(report_path),
        "resource_audit": dict(report["resource_accounting"]),
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    private_root_bytes = audit["resource_accounting"][
        "private_write_bytes_before_report"
    ]
    _finalize_verification_size(
        verification,
        private_root_bytes=private_root_bytes,
        report_file_bytes=report_file_bytes,
    )
    assert_raw_text_free(verification)
    verification_path = root / plan["outputs"]["verification"]
    projected = verification["private_write_audit"][
        "cumulative_private_write_bytes_upper_bound"
    ]
    if projected > plan["limits"]["maximum_additional_private_write_bytes"]:
        raise ValueError("CPU-smoke verification would exceed private-write cap")
    _private_write_exclusive(verification_path, verification)
    if _private_evidence_size(verification_path) != verification[
        "private_write_audit"
    ]["verification_file_bytes"]:
        raise ValueError("CPU-smoke verification byte accounting changed")
    return verification
