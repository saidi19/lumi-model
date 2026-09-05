"""Metadata-only preflight for the v0.2 private CPU SFT correctness smoke.

The preflight deliberately does not open an encoded index or binary shard.  It
binds the already-completed v0.2 fused audit, checks the current filesystem
identity with ``lstat``, and makes the weaker trust boundary explicit: this is
an inventory snapshot, not a second cryptographic attestation of shard bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-cpu-smoke-preflight-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-cpu-smoke-preflight-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-cpu-smoke-preflight-verification.v0.2"
PLAN_STATUS = "authorized_metadata_only_v0_2_cpu_smoke_preflight"
REPORT_STATUS = "PASS_v0_2_cpu_smoke_successor_ready_no_encoded_content_read"
VERIFICATION_STATUS = "PASS_verified_v0_2_cpu_smoke_metadata_only_preflight"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-cpu-smoke-preflight-v0.2.json")
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_AUTHORIZATION_TEXT = "ok start kor"
EXPECTED_AUTHORIZATION_BYTES = 12
EXPECTED_AUTHORIZATION_SHA256 = "e0869d2f304fbab4a7365ad51c28e0aac9d0240599db57827cfa67fbd349c555"
PENDING_SHA256 = "0" * 64

EXPECTED_AUTHORIZATION = {
    "candidate_or_source_row_access_authorized": False,
    "checkpoint_or_training_action_authorized": False,
    "encoded_binary_content_access_authorized": False,
    "encoded_index_row_access_authorized": False,
    "evidence_writes_authorized": True,
    "gpu_authorized": False,
    "inference_authorized": False,
    "metadata_only_cpu_smoke_preflight_authorized": True,
    "network_authorized": False,
    "private_jsonl_row_access_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_promotion_authorized": False,
    "registry_or_contract_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_or_dataset_mutation_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}

EXPECTED_LIMITS = {
    "maximum_candidate_row_reads": 0,
    "maximum_checkpoint_actions": 0,
    "maximum_encoded_binary_bytes_read": 0,
    "maximum_encoded_index_row_reads": 0,
    "maximum_evidence_writes": 2,
    "maximum_factual_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_jsonl_row_reads": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 15,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_training_actions": 0,
}

EXPECTED_EXTERNAL_BINDINGS = {
    "encoded_manifest": {
        "path": (
            "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/"
            "encoded/manifest.json"
        ),
        "sha256": "b2ec92c7e4f75a770680c794cad1e3053cc1c8a92d288a2ce2bb5a07e86941bb",
    },
    "encoded_shard_plan": {
        "path": "configs/lumi-4b-sft-token-consistent-shards-v0.2.json",
        "sha256": "9fe5d902f5f4ab4e54cbd188b35726a9b9ec3f770cc18bcf9409ed06f587fca0",
    },
    "encoded_shard_receipt": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-v0.2.json",
        "sha256": "3daf1a936177cbd9092c842fdbdd171e61c06a09e12cdf54b715230bb58040cb",
    },
    "encoded_shard_verification": {
        "path": "evidence/lumi-4b-sft-token-consistent-shards-verification-v0.2.json",
        "sha256": "701c8b2e9659526ec3f0a780f099bff32d22c83cb6b8d0f404460934d623c86a",
    },
    "final_base_binding": {
        "path": "configs/lumi-4b-final-base-binding-v0.1.json",
        "sha256": "1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172",
    },
    "legacy_cpu_runner": {
        "path": "scripts/run_sft_cpu_canary.py",
        "sha256": "b0b8edd3c77eb544d19d152c622f4977e1f4d8797f95ab999f9b320ae2b54d91",
    },
    "legacy_runtime": {
        "path": "src/lumi/posttraining_sft_runtime.py",
        "sha256": "6940238db74f7c79c06b2b4d8098314e10879601242b3987932dfda90ba0cced",
    },
    "legacy_runtime_test": {
        "path": "tests/test_posttraining_sft_runtime.py",
        "sha256": "723bde4b40c23371d18ed13bd1679623b6b7114a34104bd3ba3855fec523f7f4",
    },
    "model_implementation": {
        "path": "src/lumi/modeling_lumi.py",
        "sha256": "bf7b908d282a752132918dddad648fd8435d3f8ad9db41eac1c2e64715926fad",
    },
    "successor_seal": {
        "path": "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.2.json",
        "sha256": "1c78bcea6016b4903c4b6262f24cfcc718b6a88b68a2d1c1e3fc13bbb8ba0ce9",
    },
    "token_consistent_implementation": {
        "path": "src/lumi/posttraining_token_consistent_shards_v2.py",
        "sha256": "e5a9c7eb5ef01c35d0f4ba0c79d9e7d6ccd50d040189c104b5b2492e763a8276",
    },
    "token_consistent_runner": {
        "path": "scripts/materialize_sft_token_consistent_shards_v2.py",
        "sha256": "95be5a0c795229e1cd682402148fcf065b9c342e201d425b73db5bd71e4607cc",
    },
    "token_consistent_test": {
        "path": "tests/test_token_consistent_shards_v2.py",
        "sha256": "616f259969f6d6ce0af9ce4e98248d6dd2b3b2357e18ca0b7a0965e3fe9fa8f3",
    },
    "token_consistent_verifier": {
        "path": "scripts/verify_sft_token_consistent_shards_v2.py",
        "sha256": "0a85a839377e197b1b7c92b2dec97de01daf838112665c80601f35a7371e235b",
    },
}

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_sft_cpu_smoke_preflight_v2.py",
    "runner": "scripts/preflight_sft_cpu_smoke_v2.py",
    "test": "tests/test_sft_cpu_smoke_preflight_v2.py",
    "verifier": "scripts/verify_sft_cpu_smoke_preflight_v2.py",
}
EXPECTED_SUCCESSOR_CODE_PATHS = {
    "successor_runner": "scripts/run_sft_cpu_canary_v2.py",
    "successor_runtime": "src/lumi/posttraining_sft_runtime_v2.py",
    "successor_test": "tests/test_posttraining_sft_runtime_v2.py",
}
EXPECTED_BINDING_NAMES = (
    frozenset(EXPECTED_EXTERNAL_BINDINGS)
    | frozenset(EXPECTED_CODE_PATHS)
    | frozenset(EXPECTED_SUCCESSOR_CODE_PATHS)
)

EXPECTED_INPUT = {
    "encoded_dataset_sha256": "fcfc9e227f0d0b067062293e1adcd99d9f85406dcd50eb1363d408e287abf7c0",
    "encoded_directory": (
        "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/encoded"
    ),
    "manifest_file_sha256": EXPECTED_EXTERNAL_BINDINGS["encoded_manifest"]["sha256"],
    "manifest_internal_sha256": (
        "a75a1cb8ef1853aad519fefb3c24b0179828deb315b1d7d14b8013f2771febed"
    ),
    "manifest_inventory_sha256": (
        "61ce3f4915810f39c7a6b53eb70d3eee3f19e7027c4975babdf4160677a39588"
    ),
    "record_hash_sequence_sha256": (
        "8bd0fed685a663b048169a83e86e645806c67545ce1f08974351624ad5e64070"
    ),
    "records": 329_200,
    "receipt_file_sha256": EXPECTED_EXTERNAL_BINDINGS["encoded_shard_receipt"]["sha256"],
    "receipt_internal_sha256": ("e6ff806921fea4105f26d6d8abe0147cbd05cf8dc4ba90086f8f47368e4d36c8"),
    "seal_file_sha256": EXPECTED_EXTERNAL_BINDINGS["successor_seal"]["sha256"],
    "seal_internal_sha256": ("8c575ab8f6acafe326ec5f19c3c750c86ebe42c5f851513eaee28da8751e05b7"),
    "splits": {
        "train": {"records": 309_643, "supervised_tokens": 11_855_217, "tokens": 33_596_489},
        "validation": {"records": 19_557, "supervised_tokens": 939_293, "tokens": 2_295_902},
    },
    "supervised_tokens": 12_794_510,
    "tokens": 35_892_391,
    "verification_file_sha256": EXPECTED_EXTERNAL_BINDINGS["encoded_shard_verification"]["sha256"],
    "verification_internal_sha256": (
        "f79d65233d25515994b72471aede84fe36eb19fa9ff7df9aa3d7cbddf9e1a5d7"
    ),
}

EXPECTED_OUTPUT = {
    "report_path": "evidence/lumi-4b-sft-cpu-smoke-preflight-v0.2.json",
    "verification_path": ("evidence/lumi-4b-sft-cpu-smoke-preflight-verification-v0.2.json"),
}

EXPECTED_SUCCESSOR = {
    "actual_cpu_smoke_authorized": False,
    "code_hashes_frozen": True,
    "first_four_record_lengths_predeclared": False,
    "legacy_runtime_reusable": False,
    "limits": {
        "maximum_additional_private_write_bytes": 268_435_456,
        "maximum_candidate_row_reads": 0,
        "maximum_complete_encoded_shard_hash_reads": 0,
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
    },
    "model": {
        "head_dimension": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "maximum_positions": 4096,
        "number_attention_heads": 2,
        "number_key_value_heads": 1,
        "number_layers": 1,
        "torch_seed": 11,
        "vocabulary_size": 65536,
    },
    "outputs": {
        "plan": "configs/lumi-4b-sft-cpu-correctness-smoke-v0.2.json",
        "private_root": "artifacts/lumi-sft-cpu-smoke-v0.2/attempt-001/private",
        "report": "evidence/lumi-4b-sft-cpu-correctness-smoke-v0.2.json",
        "verification": ("evidence/lumi-4b-sft-cpu-correctness-smoke-verification-v0.2.json"),
    },
    "bounded_prefix_runtime": {
        "contiguous_train_prefix_records": 4,
        "cursor_seed": 2027,
        "dataset_loads": 3,
        "fresh_resume_sample_consumptions": 4,
        "index_rows_parsed_per_load": 4,
        "raw_text_retained": False,
        "selected_prefix_digest_bound_in_run_fingerprint": True,
        "shuffle_domain_records": 4,
        "uninterrupted_sample_consumptions": 4,
    },
    "required_hardening": {
        "complete_selected_shard_content_hash": False,
        "durable_no_retry_markers": True,
        "encoded_inputs_opened_nofollow": True,
        "exact_external_manifest_inventory_and_seal_bindings": True,
        "exact_inventory_metadata_digest_rechecked": True,
        "exclusive_nofollow_private_outputs": True,
        "fresh_private_tiny_source_checkpoint": True,
        "output_directories_mode_0700": True,
        "output_files_mode_0600_nlink_one_owner": True,
        "parent_chain_without_symlinks": True,
        "same_descriptor_selected_bytes_retained": True,
        "selected_prefix_digest_in_run_fingerprint": True,
        "trust_limited_to_prior_full_audit_plus_stable_private_metadata": True,
        "whole_dataset_binary_rehash": False,
    },
    "training": {
        "attention": "cpu_eager",
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.0001,
        "maximum_gradient_norm": 1.0,
        "target_optimizer_steps": 2,
        "warmup_steps": 0,
        "weight_decay": 0.01,
    },
}

EXPECTED_TRUST = {
    "current_content_trust_source": "completed_v0_2_fused_audit",
    "inventory_fields": [
        "relative_path",
        "st_dev",
        "st_ino",
        "st_size",
        "mode",
        "st_uid",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    ],
    "metadata_digest_is_content_rehash": False,
    "preflight_encoded_content_reads": 0,
    "successor_must_recheck_exact_metadata_digest_before_first_content_read": True,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in CPU smoke preflight input")
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
    except UnicodeDecodeError as exc:
        raise ValueError("CPU smoke preflight JSON is not UTF-8") from exc


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
                _strict_equal(left, right) for left, right in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    return type(actual) is type(expected) and actual == expected


def assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError(f"raw training text key forbidden at {path}.{key}")
            assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_raw_text_free(item, f"{path}[{index}]")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _require_under(root: Path, path: Path) -> Path:
    root = _absolute(root)
    path = _absolute(path)
    if not path.is_relative_to(root):
        raise ValueError("CPU smoke preflight path escaped repository")
    return path


def _require_parent_chain_without_symlinks(root: Path, path: Path) -> None:
    root = _absolute(root)
    path = _require_under(root, path)
    current = root
    for part in path.relative_to(root).parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("CPU smoke preflight parent chain is unsafe")


def _read_secure_file(path: Path, *, private: bool = False) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest_payload = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("CPU smoke preflight binding is not regular")
        if private and (
            stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
        ):
            raise ValueError("private CPU smoke preflight binding metadata changed")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest_payload.extend(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    ):
        raise ValueError("CPU smoke preflight binding identity changed while reading")
    return bytes(digest_payload)


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(_read_secure_file(Path(path)))


def _load_object(path: Path, *, private: bool = False) -> dict[str, Any]:
    value = strict_json_bytes(_read_secure_file(path, private=private))
    if not isinstance(value, dict):
        raise ValueError("CPU smoke preflight binding must contain an object")
    return value


def _verify_internal(value: Mapping[str, Any], key: str) -> str:
    expected = value.get(key)
    unsigned = dict(value)
    unsigned.pop(key, None)
    if not isinstance(expected, str) or expected != sha256_value(unsigned):
        raise ValueError(f"CPU smoke preflight internal hash changed: {key}")
    return expected


def _resolve_binding(root: Path, name: str, binding: Mapping[str, Any]) -> Path:
    expected = EXPECTED_EXTERNAL_BINDINGS.get(name)
    expected_path = (
        expected["path"]
        if expected is not None
        else EXPECTED_CODE_PATHS.get(name, EXPECTED_SUCCESSOR_CODE_PATHS.get(name))
    )
    if (
        expected_path is None
        or not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256"}
        or binding.get("path") != expected_path
        or not isinstance(binding.get("sha256"), str)
        or _SHA256.fullmatch(binding["sha256"]) is None
        or (expected is not None and not _strict_equal(binding, expected))
    ):
        raise ValueError(f"CPU smoke preflight binding changed: {name}")
    if name in EXPECTED_SUCCESSOR_CODE_PATHS and binding["sha256"] == PENDING_SHA256:
        raise ValueError(f"CPU smoke successor code hash is not frozen: {name}")
    path = _require_under(root, root / expected_path)
    _require_parent_chain_without_symlinks(root, path)
    payload = _read_secure_file(path, private=name == "encoded_manifest")
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError(f"CPU smoke preflight binding hash changed: {name}")
    return path


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("observed_at") != EXPECTED_OBSERVED_AT
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or not _strict_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION)
        or not _strict_equal(plan.get("input"), EXPECTED_INPUT)
        or not _strict_equal(plan.get("limits"), EXPECTED_LIMITS)
        or not _strict_equal(plan.get("output"), EXPECTED_OUTPUT)
        or not _strict_equal(plan.get("successor"), EXPECTED_SUCCESSOR)
        or not _strict_equal(plan.get("trust_boundary"), EXPECTED_TRUST)
        or not isinstance(plan.get("bindings"), Mapping)
        or set(plan["bindings"]) != EXPECTED_BINDING_NAMES
    ):
        raise ValueError("CPU smoke preflight plan contract changed")
    if len(plan["authorization_text"].encode()) != EXPECTED_AUTHORIZATION_BYTES:
        raise ValueError("CPU smoke preflight authorization byte count changed")
    if sha256_bytes(plan["authorization_text"].encode()) != EXPECTED_AUTHORIZATION_SHA256:
        raise ValueError("CPU smoke preflight authorization digest changed")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = _absolute(Path(repository_root))
    path = _absolute(Path(path))
    if path != root / EXPECTED_PLAN_PATH:
        raise ValueError("unexpected CPU smoke preflight plan path")
    _require_parent_chain_without_symlinks(root, path)
    plan = _load_object(path)
    _validate_plan_shape(plan)
    for name, binding in plan["bindings"].items():
        _resolve_binding(root, name, binding)
    return plan


def _validate_evidence_chain(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    paths = {
        name: root / plan["bindings"][name]["path"]
        for name in (
            "encoded_manifest",
            "encoded_shard_receipt",
            "encoded_shard_verification",
            "successor_seal",
        )
    }
    manifest = _load_object(paths["encoded_manifest"], private=True)
    receipt = _load_object(paths["encoded_shard_receipt"])
    verification = _load_object(paths["encoded_shard_verification"])
    seal = _load_object(paths["successor_seal"])
    assert_raw_text_free(manifest)
    assert_raw_text_free(receipt)
    assert_raw_text_free(verification)
    assert_raw_text_free(seal)
    if _verify_internal(manifest, "manifest_sha256") != EXPECTED_INPUT["manifest_internal_sha256"]:
        raise ValueError("v0.2 encoded manifest internal identity changed")
    if _verify_internal(receipt, "receipt_sha256") != EXPECTED_INPUT["receipt_internal_sha256"]:
        raise ValueError("v0.2 receipt internal identity changed")
    if (
        _verify_internal(verification, "verification_sha256")
        != EXPECTED_INPUT["verification_internal_sha256"]
    ):
        raise ValueError("v0.2 verification internal identity changed")
    if _verify_internal(seal, "seal_sha256") != EXPECTED_INPUT["seal_internal_sha256"]:
        raise ValueError("v0.2 seal internal identity changed")
    expected_manifest_projection = {
        "encoded_dataset_sha256": EXPECTED_INPUT["encoded_dataset_sha256"],
        "manifest_sha256": EXPECTED_INPUT["manifest_internal_sha256"],
        "record_hash_sequence_sha256": EXPECTED_INPUT["record_hash_sequence_sha256"],
        "records": EXPECTED_INPUT["records"],
        "seal_sha256": EXPECTED_INPUT["seal_internal_sha256"],
        "splits": EXPECTED_INPUT["splits"],
        "supervised_tokens": EXPECTED_INPUT["supervised_tokens"],
        "tokens": EXPECTED_INPUT["tokens"],
    }
    if any(manifest.get(key) != value for key, value in expected_manifest_projection.items()):
        raise ValueError("v0.2 encoded manifest totals changed")
    if manifest.get("production_eligible") is not False or manifest.get("format") != {
        "index": "canonical-jsonl-record-hash-only-v0.1",
        "labels": "little-endian-int32-ignore-index-minus-100",
        "raw_text_retained": False,
        "tokens": "little-endian-uint32",
    }:
        raise ValueError("v0.2 encoded manifest format or eligibility changed")
    shards = manifest.get("shards")
    expected_identities = [("train", index) for index in range(9)] + [("validation", 0)]
    if (
        not isinstance(shards, list)
        or [(item.get("split"), item.get("index")) for item in shards if isinstance(item, Mapping)]
        != expected_identities
    ):
        raise ValueError("v0.2 shard identity inventory changed")
    if sha256_value(shards) != EXPECTED_INPUT["manifest_inventory_sha256"]:
        raise ValueError("v0.2 manifest shard inventory digest changed")
    if (
        receipt.get("encoded_shards", {}).get("manifest_file_sha256")
        != EXPECTED_INPUT["manifest_file_sha256"]
        or receipt.get("encoded_shards", {}).get("manifest_internal_sha256")
        != EXPECTED_INPUT["manifest_internal_sha256"]
        or receipt.get("successor_seal_internal_sha256") != EXPECTED_INPUT["seal_internal_sha256"]
        or verification.get("encoded_manifest_file_sha256")
        != EXPECTED_INPUT["manifest_file_sha256"]
        or verification.get("encoded_manifest_internal_sha256")
        != EXPECTED_INPUT["manifest_internal_sha256"]
        or verification.get("receipt_file_sha256") != EXPECTED_INPUT["receipt_file_sha256"]
        or verification.get("receipt_internal_sha256") != EXPECTED_INPUT["receipt_internal_sha256"]
        or verification.get("successor_seal_file_sha256") != EXPECTED_INPUT["seal_file_sha256"]
        or verification.get("successor_seal_internal_sha256")
        != EXPECTED_INPUT["seal_internal_sha256"]
        or seal.get("encoded_dataset_sha256") != EXPECTED_INPUT["encoded_dataset_sha256"]
        or seal.get("records") != EXPECTED_INPUT["records"]
        or receipt.get("production_eligible") is not False
        or verification.get("production_eligible") is not False
        or seal.get("production_eligible") is not False
    ):
        raise ValueError("v0.2 evidence binding chain changed")
    return manifest


def inventory_metadata(root: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return an lstat-only private inventory; never open index or binary files."""

    encoded = _require_under(root, root / EXPECTED_INPUT["encoded_directory"])
    private_chain = [
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2",
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001",
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private",
        encoded,
    ]
    for directory in private_chain:
        info = directory.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid()
        ):
            raise ValueError("v0.2 private encoded directory metadata changed")

    manifest_info = (encoded / "manifest.json").lstat()
    if stat.S_ISLNK(manifest_info.st_mode) or not stat.S_ISREG(manifest_info.st_mode):
        raise ValueError("v0.2 encoded manifest metadata changed")
    declared: dict[str, int] = {"manifest.json": manifest_info.st_size}
    for shard in manifest["shards"]:
        files = shard.get("files")
        if not isinstance(files, list) or len(files) != 3:
            raise ValueError("v0.2 shard must declare exactly three files")
        for item in files:
            relative = item.get("path")
            size = item.get("bytes")
            if (
                not isinstance(relative, str)
                or Path(relative).name != relative
                or not isinstance(size, int)
                or size < 1
                or relative in declared
            ):
                raise ValueError("v0.2 shard file declaration changed")
            declared[relative] = size
    if set(os.listdir(encoded)) != set(declared):
        raise ValueError("v0.2 encoded directory inventory changed")

    records: list[dict[str, Any]] = []
    for relative in sorted(declared):
        path = encoded / relative
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size != declared[relative]
        ):
            raise ValueError(f"v0.2 encoded file metadata changed: {relative}")
        records.append(
            {
                "mode": "0600",
                "relative_path": relative,
                "st_ctime_ns": info.st_ctime_ns,
                "st_dev": info.st_dev,
                "st_ino": info.st_ino,
                "st_mtime_ns": info.st_mtime_ns,
                "st_nlink": info.st_nlink,
                "st_size": info.st_size,
                "st_uid": info.st_uid,
            }
        )
    return records, sha256_value(records)


def output_paths(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = _require_under(root, root / plan["output"]["report_path"])
    verification = _require_under(root, root / plan["output"]["verification_path"])
    if (report, verification) != (
        root / EXPECTED_OUTPUT["report_path"],
        root / EXPECTED_OUTPUT["verification_path"],
    ):
        raise ValueError("CPU smoke preflight output path changed")
    _require_parent_chain_without_symlinks(root, report)
    _require_parent_chain_without_symlinks(root, verification)
    return report, verification


def _successor_paths(root: Path) -> list[Path]:
    output = EXPECTED_SUCCESSOR["outputs"]
    paths = [
        root / output["plan"],
        root / output["private_root"],
        root / output["report"],
        root / output["verification"],
    ]
    for path in paths:
        _require_under(root, path)
        _require_parent_chain_without_symlinks(root, path)
    return paths


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    payload = json_file_bytes(value)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_uid != os.geteuid():
            raise ValueError("CPU smoke preflight evidence metadata changed")
    finally:
        os.close(descriptor)
    return len(payload)


def build_report(
    plan: Mapping[str, Any], *, repository_root: str | Path, plan_file_sha256: str
) -> dict[str, Any]:
    root = _absolute(Path(repository_root))
    manifest = _validate_evidence_chain(plan, root)
    inventory, inventory_sha256 = inventory_metadata(root, manifest)
    if any(path.exists() or path.is_symlink() for path in _successor_paths(root)):
        raise FileExistsError("v0.2 CPU smoke successor output already exists")
    report = {
        "authorization_boundary": {
            "encoded_binary_content_read": False,
            "encoded_index_rows_read": 0,
            "gpu_inference_or_training_used": False,
            "network_or_remote_access_used": False,
            "production_eligible": False,
            "source_candidate_or_private_jsonl_rows_read": 0,
        },
        "bindings": {
            "encoded_manifest_file_sha256": EXPECTED_INPUT["manifest_file_sha256"],
            "encoded_manifest_internal_sha256": EXPECTED_INPUT["manifest_internal_sha256"],
            "encoded_manifest_inventory_sha256": EXPECTED_INPUT[
                "manifest_inventory_sha256"
            ],
            "encoded_shard_verification_file_sha256": EXPECTED_INPUT["verification_file_sha256"],
            "encoded_shard_verification_internal_sha256": EXPECTED_INPUT[
                "verification_internal_sha256"
            ],
            "plan_file_sha256": plan_file_sha256,
            "successor_seal_file_sha256": EXPECTED_INPUT["seal_file_sha256"],
            "successor_seal_internal_sha256": EXPECTED_INPUT["seal_internal_sha256"],
        },
        "decision": {
            "actual_cpu_smoke_authorized": False,
            "first_four_record_lengths_known": False,
            "legacy_runtime_reusable": False,
            "reason": "fresh_hardened_bounded_prefix_runtime_required",
            "successor_ready_for_fresh_authorization": True,
        },
        "encoded_inventory": {
            "files": len(inventory),
            "metadata_sha256": inventory_sha256,
            "private_directories_mode_0700": True,
            "private_files_mode_0600_owner_nlink_one": True,
        },
        "input_totals": {
            "records": EXPECTED_INPUT["records"],
            "splits": EXPECTED_INPUT["splits"],
            "supervised_tokens": EXPECTED_INPUT["supervised_tokens"],
            "tokens": EXPECTED_INPUT["tokens"],
        },
        "next_gate": {
            "action": "authorize_v0_2_hardened_tiny_cpu_correctness_smoke",
            "authorization_required": True,
            "complete_encoded_shard_hash_reads": 0,
            "encoded_binary_bytes_read_cap": 393_120,
            "encoded_index_bytes_read_cap": 2_100,
            "encoded_index_rows_parsed": 12,
            "encoded_record_file_reads": 12,
            "gpu_required": False,
            "production_sft": False,
            "training_sample_consumptions": 8,
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
        "resource_accounting": {
            "encoded_binary_bytes_read": 0,
            "encoded_index_row_reads": 0,
            "factual_passes": 1,
            "gpu_use": 0,
            "network_bytes": 0,
            "private_jsonl_candidate_source_or_test_rows_read": 0,
            "training_actions": 0,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "trust_boundary": EXPECTED_TRUST,
    }
    assert_raw_text_free(report)
    return report


def run_preflight(*, plan_path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = _absolute(Path(repository_root))
    plan_path = _absolute(Path(plan_path))
    plan = load_plan(plan_path, repository_root=root)
    report_path, verification_path = output_paths(plan, root)
    if (
        report_path.exists()
        or report_path.is_symlink()
        or verification_path.exists()
        or verification_path.is_symlink()
    ):
        raise FileExistsError("CPU smoke preflight evidence output already exists")
    report = build_report(plan, repository_root=root, plan_file_sha256=file_sha256(plan_path))
    if _write_json_exclusive(report_path, report) > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("CPU smoke preflight evidence exceeded write cap")
    return report


def validate_report(report: Mapping[str, Any], *, plan_file_sha256: str) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("bindings", {}).get("plan_file_sha256") != plan_file_sha256
        or report.get("decision", {}).get("legacy_runtime_reusable") is not False
        or report.get("decision", {}).get("first_four_record_lengths_known") is not False
        or report.get("resource_accounting", {}).get("encoded_binary_bytes_read") != 0
        or report.get("resource_accounting", {}).get("encoded_index_row_reads") != 0
        or report.get("resource_accounting", {}).get("training_actions") != 0
        or report.get("trust_boundary") != EXPECTED_TRUST
    ):
        raise ValueError("CPU smoke preflight report changed")
    assert_raw_text_free(report)


def run_verification(
    *, plan_path: str | Path, report_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    root = _absolute(Path(repository_root))
    plan_path = _absolute(Path(plan_path))
    report_path = _absolute(Path(report_path))
    plan = load_plan(plan_path, repository_root=root)
    expected_report, verification_path = output_paths(plan, root)
    if report_path != expected_report:
        raise ValueError("unexpected CPU smoke preflight report path")
    if verification_path.exists() or verification_path.is_symlink():
        raise FileExistsError("CPU smoke preflight verification already exists")
    report_payload = _read_secure_file(report_path, private=True)
    report = strict_json_bytes(report_payload)
    if not isinstance(report, dict):
        raise ValueError("CPU smoke preflight report must be an object")
    plan_sha256 = file_sha256(plan_path)
    validate_report(report, plan_file_sha256=plan_sha256)
    manifest = _validate_evidence_chain(plan, root)
    inventory, inventory_sha256 = inventory_metadata(root, manifest)
    if report.get("encoded_inventory", {}).get("metadata_sha256") != inventory_sha256:
        raise ValueError("encoded lstat inventory changed between preflight passes")
    verification = {
        "checks": {
            "all_v0_2_evidence_file_and_internal_hashes_match": True,
            "encoded_content_not_opened": True,
            "exact_9_train_and_1_validation_shards_bound": True,
            "exact_totals_bound": True,
            "future_first_four_lengths_not_predeclared": True,
            "future_bounded_prefix_is_limited_to_12_file_record_reads": True,
            "inventory_metadata_digest_matches_both_passes": True,
            "legacy_runtime_rejected": True,
            "no_network_gpu_inference_training_or_remote_action": True,
            "nonproduction_boundary_preserved": True,
            "private_modes_owner_and_link_counts_match": True,
            "successor_requires_fresh_authorization": True,
        },
        "encoded_inventory_metadata_sha256": inventory_sha256,
        "plan_file_sha256": plan_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": sha256_bytes(report_payload),
        "resource_accounting": {
            "encoded_binary_bytes_read": 0,
            "encoded_index_row_reads": 0,
            "factual_passes": 2,
            "gpu_use": 0,
            "network_bytes": 0,
            "private_jsonl_candidate_source_or_test_rows_read": 0,
            "training_actions": 0,
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "trust_boundary": EXPECTED_TRUST,
    }
    verification["verification_sha256"] = sha256_value(verification)
    assert_raw_text_free(verification)
    total = len(report_payload) + _write_json_exclusive(verification_path, verification)
    if total > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("CPU smoke preflight evidence exceeded write cap")
    return verification
