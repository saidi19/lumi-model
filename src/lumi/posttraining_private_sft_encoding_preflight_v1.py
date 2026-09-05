"""Metadata-only preflight for the registry-v0.24 private SFT encoder."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit_v18 as mix_v18
from . import posttraining_learning_candidate_seal_v3 as seal_v3
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)

PLAN_SCHEMA_VERSION = "lumi.sft-private-encoding-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-private-encoding-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-private-encoding-preflight-verification.v0.1"
PLAN_STATUS = "authorized_registry_v0_24_metadata_only_private_encoding_preflight"
REPORT_STATUS = "PASS_private_encoding_successor_ready_no_rows_or_shards_read"
VERIFICATION_STATUS = "PASS_verified_private_encoding_metadata_only_preflight"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-private-nonproduction-encoding-preflight-v0.1.json")
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_AUTHORIZATION_TEXT = "ok shuru kor next step"
EXPECTED_AUTHORIZATION_BYTES = 22
EXPECTED_AUTHORIZATION_SHA256 = "21942a4369f358c42fa420f16090400c6f8b6a96e10b8640d4766a9d0a874126"
EXPECTED_AUTHORIZATION = {
    "candidate_mutation_authorized": False,
    "candidate_row_access_authorized": False,
    "encoded_shard_materialization_authorized": False,
    "evidence_writes_authorized": True,
    "gpu_authorized": False,
    "inference_authorized": False,
    "metadata_only_encoding_preflight_authorized": True,
    "network_authorized": False,
    "private_jsonl_row_access_authorized": False,
    "private_input_or_predecessor_mutation_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_promotion_authorized": False,
    "registry_or_contract_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_row_access_authorized": False,
    "source_mutation_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_or_source_mutations": 0,
    "maximum_candidate_row_reads": 0,
    "maximum_encoded_index_row_reads": 0,
    "maximum_encoded_shard_writes": 0,
    "maximum_evidence_writes": 2,
    "maximum_factual_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_jsonl_row_reads": 0,
    "maximum_private_input_or_predecessor_mutations": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_registry_or_contract_writes": 0,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 15,
    "maximum_runtime_seconds_per_pass": 450,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_training_actions": 0,
}
EXPECTED_OUTPUT = {
    "encoded_shards_created": False,
    "report_path": "evidence/lumi-4b-sft-private-nonproduction-encoding-preflight-v0.1.json",
    "successor_seal_created": False,
    "verification_path": (
        "evidence/lumi-4b-sft-private-nonproduction-encoding-preflight-verification-v0.1.json"
    ),
}
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_private_sft_encoding_preflight_v1.py",
    "runner": "scripts/preflight_private_sft_encoding_v1.py",
    "test": "tests/test_private_sft_encoding_preflight_v1.py",
    "verifier": "scripts/verify_private_sft_encoding_preflight_v1.py",
}
EXPECTED_EXTERNAL_BINDINGS = {
    "candidate_mix_implementation": {
        "path": "src/lumi/posttraining_candidate_mix_reaudit_v18.py",
        "sha256": "d27061fe5a078789858a1b7c5ba7977ce7cc3738a187df8285004cb9e909f9f0",
    },
    "candidate_mix_plan": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
        "sha256": "14e737fa9768c4527eb2663d5fda32d38121459591de44c83f899066034acb26",
    },
    "candidate_mix_report": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
        "sha256": "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000",
    },
    "candidate_mix_verification": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.18.json",
        "sha256": "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651",
    },
    "chat_template": {
        "path": "configs/lumi-4b-chat-template-v0.1.json",
        "sha256": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    },
    "data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.2.json",
        "sha256": "281fba028c58f5bc0fd489c1d70694e54d37d5dffd09dcc79676d42029497d33",
    },
    "data_contract_activation_plan": {
        "path": "configs/lumi-4b-sft-private-research-data-contract-v0.2.json",
        "sha256": "ad24966e5c6a5ad196843f45ec23ccc147e6a3f5826d19a3a78a3ddda593961a",
    },
    "data_contract_activation_report": {
        "path": "evidence/lumi-4b-sft-private-research-data-contract-v0.2.json",
        "sha256": "3434220a9cb669edc971889101a0547ff19d6d4e8118e12643139a435b48daf4",
    },
    "data_contract_activation_verification": {
        "path": "evidence/lumi-4b-sft-private-research-data-contract-verification-v0.2.json",
        "sha256": "8a5f2cbb3de53eaefd13069d0b114ca3cd1cf6060d9c2b187754dae81b631aba",
    },
    "encoding_format_implementation": {
        "path": "src/lumi/posttraining_format.py",
        "sha256": "32df8a7b29d73354f21b73c79442f7c44f437c24330662b308a381326ad0a9d5",
    },
    "final_base_binding": {
        "path": "configs/lumi-4b-final-base-binding-v0.1.json",
        "sha256": "1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172",
    },
    "learning_seal": {
        "path": "evidence/lumi-4b-sft-learning-candidate-seal-v0.3.json",
        "sha256": "c034e08f7349778c371798bd24f96336bcaab4ff74baca2c07e7fb1c991b1576",
    },
    "learning_seal_implementation": {
        "path": "src/lumi/posttraining_learning_candidate_seal_v3.py",
        "sha256": "61369ba5d1a0602b30f200a56d79a21ff81ca3ec57110010cb05a1b882b6068b",
    },
    "learning_seal_plan": {
        "path": "configs/lumi-4b-sft-learning-candidate-seal-v0.3.json",
        "sha256": "c1d478db8b8e334e00c899857a92c289b4c418606266f91dac122c381e0041dc",
    },
    "learning_seal_runner": {
        "path": "scripts/seal_sft_learning_candidate_v3.py",
        "sha256": "78fddab88696a0dc538df390b8388d7b2d2f0b9a6045c4e46877c2736d3ca382",
    },
    "learning_seal_test": {
        "path": "tests/test_learning_candidate_seal_v3.py",
        "sha256": "771d02e1031f4ca2f9538a9e1552c5b66bfc3345a3f0da8b692829d825a1118b",
    },
    "learning_seal_verification": {
        "path": "evidence/lumi-4b-sft-learning-candidate-seal-verification-v0.3.json",
        "sha256": "9f7a5dfd556632a2186951efcb7c9e9531bed2bea3b69b9904001dc280170da1",
    },
    "learning_seal_verifier": {
        "path": "scripts/verify_sft_learning_candidate_seal_v3.py",
        "sha256": "095cbf4994b9175092ca441ddec5a03cf56bc9682fa583f829cdf2aa0b2ba20f",
    },
    "policy_loader": {
        "path": "src/lumi/posttraining_data.py",
        "sha256": "e520c3771869b7221848a456e59c3b15a279fb1ebdf700646600c774e25b74b5",
    },
    "shard_format_reference": {
        "path": "src/lumi/posttraining_shards.py",
        "sha256": "d8480fc71c061b0c3ae5a4228d754044da8417ead86bd0033b997566f2645504",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.24.json",
        "sha256": "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    },
    "tokenizer": {
        "path": (
            "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
            "staging/input/tokenizer/tokenizer.json"
        ),
        "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    },
    "tokenizer_implementation": {
        "path": "src/lumi/posttraining_gpt_oss_teacher_source_promotion_preflight_v16.py",
        "sha256": "c1e1bacbaf3bbdc5a0bdc1720f0063fa79330b1ada95f5a304feed2385f389e8",
    },
}
EXPECTED_INPUT = {
    "accepted_dataset_sha256": "1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e",
    "context_window": 4_096,
    "learning_seal_file_sha256": EXPECTED_EXTERNAL_BINDINGS["learning_seal"]["sha256"],
    "learning_seal_internal_sha256": (
        "ce31de12a23a0ee1fd02963e3c431397fd3ccfb8de85f142f016089992d3f6cf"
    ),
    "learning_seal_verification_file_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "learning_seal_verification"
    ]["sha256"],
    "learning_seal_verification_internal_sha256": (
        "2a3996012baa5df5d29e0c1d4e47e13f59a784432da98d07d8ab64fbbf86681e"
    ),
    "maximum_record_tokens": 4_095,
    "records": 329_200,
    "splits": {
        "train": {
            "bytes": 493_034_265,
            "file_sha256": "cfc80186ba520573c451efd00ed302a5af06bbee84ae247018caa2674986969e",
            "path": (
                "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private/train.jsonl"
            ),
            "record_id_sequence_sha256": (
                "123c0f4e658f44d6a0d6b6876ab42445ec5b949796cd0b0e92b4589c5e2e4d12"
            ),
            "records": 309_643,
            "serialized_tokens": 33_596_489,
            "supervised_tokens": 11_855_217,
        },
        "validation": {
            "bytes": 33_168_094,
            "file_sha256": "bd63394bc4e2c40919ce086c1eb6ac47d91341018b4b2b19c5cfb391435bdc93",
            "path": (
                "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private/"
                "validation.jsonl"
            ),
            "record_id_sequence_sha256": (
                "1b5f0e5b9cfbcca3d5b4719933dcbfc555e7b3a3a1a6a9c15e4affe65817e8e2"
            ),
            "records": 19_557,
            "serialized_tokens": 2_295_902,
            "supervised_tokens": 939_293,
        },
    },
}
EXPECTED_ENCODING_CONTRACT = {
    "append_eos": True,
    "context_window": 4_096,
    "encoded_directory_mode": "0700",
    "encoded_file_mode": "0600",
    "index_format": "canonical-jsonl-record-hash-only-v0.1",
    "label_format": "little-endian-int32-ignore-index-minus-100",
    "label_ignore_index": -100,
    "loss_roles": ["assistant", "tool"],
    "maximum_record_tokens": 4_095,
    "maximum_tokens_per_shard": 4_194_304,
    "ordering": "train_then_validation_existing_file_order_v0.1",
    "overlength_policy": "reject",
    "raw_text_in_encoded_outputs": False,
    "token_format": "little-endian-uint32",
    "token_ids_fit_uint32_and_int32": True,
    "tokenizer_added_token_count": 11,
    "tokenizer_id_maximum": 65_535,
    "tokenizer_id_minimum": 0,
    "tokenizer_model": "BPE",
    "tokenizer_sha256": EXPECTED_EXTERNAL_BINDINGS["tokenizer"]["sha256"],
    "tokenizer_unique_id_count": 65_536,
    "tokenizer_vocab_size": 65_536,
}
EXPECTED_PROJECTION = {
    "binary_token_and_label_bytes": 287_139_128,
    "encoded_index_maximum_bytes": 57_610_000,
    "encoded_index_maximum_bytes_per_row": 175,
    "evidence_and_manifest_reserve_bytes": 4_194_304,
    "maximum_projected_write_bytes": 348_943_432,
    "shards": 10,
    "train_shards": 9,
    "validation_shards": 1,
}
EXPECTED_SUCCESSOR_LIMITS = {
    "maximum_candidate_row_reads": 0,
    "maximum_encoded_index_row_reads": 329_200,
    "maximum_full_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_input_jsonl_row_reads": 329_200,
    "maximum_input_or_predecessor_mutations": 0,
    "maximum_network_bytes": 0,
    "maximum_output_files": 34,
    "maximum_private_write_bytes": 536_870_912,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 180,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_total_row_reads": 658_400,
    "maximum_training_actions": 0,
    "maximum_tokens_per_shard": 4_194_304,
}
EXPECTED_SUCCESSOR_OUTPUT = {
    "encoded_shards_path": (
        "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/encoded"
    ),
    "receipt_path": "evidence/lumi-4b-sft-token-consistent-shards-v0.2.json",
    "successor_seal_path": (
        "evidence/lumi-4b-sft-learning-candidate-token-consistent-seal-v0.2.json"
    ),
    "verification_path": ("evidence/lumi-4b-sft-token-consistent-shards-verification-v0.2.json"),
}
EXPECTED_SUCCESSOR = {
    "fresh_versioned_executor_required": True,
    "generic_shard_writer_reusable": False,
    "historical_registry_v0_17_executor_reusable": False,
    "limits": EXPECTED_SUCCESSOR_LIMITS,
    "output": EXPECTED_SUCCESSOR_OUTPUT,
    "production_dataset_seal": False,
    "production_eligible": False,
    "required_hardening": {
        "candidate_or_source_files_opened": False,
        "durable_no_retry_markers_before_data_reads": True,
        "encoded_directory_mode": "0700",
        "encoded_file_mode": "0600",
        "exclusive_nofollow_output_creation": True,
        "fail_before_publish_on_any_mismatch": True,
        "fused_single_pass_encoded_index_and_binary_audit": True,
        "input_file_nlink_one": True,
        "input_hash_verified_on_same_stream_used_for_encoding": True,
        "input_open_nofollow": True,
        "input_owner_matches_process": True,
        "input_record_sequence_and_split_totals_verified": True,
        "input_same_fd_regular_mode_0600_nlink_one": True,
        "output_file_nlink_one": True,
        "private_parent_chain_anchored_without_symlinks": True,
        "raw_text_absent_from_indices_shards_and_evidence": True,
    },
    "sft_training": False,
}
EXPECTED_PRODUCTION_BLOCKERS = [
    "active_production_data_contract_not_updated",
    "dataset_seal_cpu_canary_and_sft_not_authorized",
]
EXPECTED_DECISION = {
    "encoded_shards_created": False,
    "generic_shard_writer_reusable": False,
    "metadata_only_preflight_passed": True,
    "private_nonproduction_encoding_safe_to_authorize": True,
    "production_encoding_or_sft_safe_to_execute": False,
    "reason": "fresh_hardened_registry_v0_24_executor_required_old_and_generic_executors_rejected",
}
EXPECTED_NEXT_GATE = {
    "action": "authorize_v0_2_private_nonproduction_token_consistent_seal_and_shards",
    "authorization_required": True,
    "encoded_index_row_reads": 329_200,
    "gpu_required": False,
    "input_jsonl_row_reads": 329_200,
    "original_candidate_or_source_row_reads": 0,
    "production_dataset_seal": False,
    "sft_training": False,
}
EXPECTED_INPUT_METADATA = {
    "private_root": {"directory": True, "mode": "0700", "owner_matches_process": True},
    **{
        split: {
            "bytes": values["bytes"],
            "mode": "0600",
            "nlink": 1,
            "owner_matches_process": True,
            "regular_file": True,
            "row_reads": 0,
        }
        for split, values in EXPECTED_INPUT["splits"].items()
    },
}
EXPECTED_OUTPUT_ABSENCE = {
    "encoded_shards_absent": True,
    "receipt_absent": True,
    "successor_seal_absent": True,
    "verification_absent": True,
}
EXPECTED_BINDING_CHAIN = {
    "candidate_mix_plan_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_plan"]["sha256"],
    "candidate_mix_report_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"]["sha256"],
    "candidate_mix_report_internal_sha256": (
        "7bce1bd1bcaf5b804ba5f8da5cb2614836fe8a398028079fae498131d310858e"
    ),
    "candidate_mix_verification_internal_sha256": (
        "162f2e8d803426921c1457ae9f5127e978be13120b54f13301cb47d53d6176cc"
    ),
    "candidate_mix_verification_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_verification"][
        "sha256"
    ],
    "data_contract_activation_plan_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "data_contract_activation_plan"
    ]["sha256"],
    "data_contract_activation_report_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "data_contract_activation_report"
    ]["sha256"],
    "data_contract_activation_verification_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "data_contract_activation_verification"
    ]["sha256"],
    "data_contract_sha256": EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"],
    "final_base_binding_sha256": EXPECTED_EXTERNAL_BINDINGS["final_base_binding"]["sha256"],
    "learning_seal_file_sha256": EXPECTED_EXTERNAL_BINDINGS["learning_seal"]["sha256"],
    "learning_seal_internal_sha256": EXPECTED_INPUT["learning_seal_internal_sha256"],
    "learning_seal_plan_sha256": EXPECTED_EXTERNAL_BINDINGS["learning_seal_plan"]["sha256"],
    "learning_seal_verification_file_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "learning_seal_verification"
    ]["sha256"],
    "learning_seal_verification_internal_sha256": EXPECTED_INPUT[
        "learning_seal_verification_internal_sha256"
    ],
    "source_registry_sha256": EXPECTED_EXTERNAL_BINDINGS["source_registry"]["sha256"],
}
EXPECTED_RESOURCE_PASS_ONE = {
    "candidate_or_source_mutations": 0,
    "candidate_row_reads": 0,
    "encoded_index_row_reads": 0,
    "encoded_shard_writes": 0,
    "evidence_writes": 1,
    "factual_passes": 1,
    "gpu_use": 0,
    "inference_calls": 0,
    "network_bytes": 0,
    "private_jsonl_row_reads": 0,
    "private_input_or_predecessor_mutations": 0,
    "registry_or_contract_writes": 0,
    "source_row_reads": 0,
    "test_object_reads": 0,
    "training_actions": 0,
}
EXPECTED_RESOURCE_PASS_TWO = {
    **EXPECTED_RESOURCE_PASS_ONE,
    "evidence_writes": 2,
    "factual_passes": 2,
}
EXPECTED_VERIFICATION_CHECKS = {
    "all_private_candidate_source_and_test_row_reads_zero": True,
    "candidate_source_and_private_input_mutation_prohibited": True,
    "chat_template_and_tokenizer_compatibility_verified": True,
    "exact_current_seal_file_and_internal_chain_verified": True,
    "exact_input_file_metadata_verified_without_opening_jsonl": True,
    "exact_nine_train_and_one_validation_shards_proved": True,
    "future_512_mib_write_cap_has_bounded_projection": True,
    "future_private_modes_and_same_stream_input_integrity_required": True,
    "generic_shard_writer_rejected_without_hardening": True,
    "historical_registry_v0_17_executor_rejected": True,
    "no_shards_seal_gpu_inference_network_or_training": True,
    "production_blockers_and_nonproduction_status_preserved": True,
    "raw_training_text_absent_from_evidence": True,
    "successor_requires_fresh_explicit_authorization": True,
}

EXPECTED_REPORT_MARKER = b'{"started":"private_sft_encoding_preflight_v0.1_pass_1"}\n'
EXPECTED_VERIFICATION_MARKER = b'{"started":"private_sft_encoding_preflight_v0.1_pass_2"}\n'
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
_PLAN_KEYS = frozenset(
    {
        "authorization",
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
        "bindings",
        "input",
        "limits",
        "observed_at",
        "output",
        "schema_version",
        "status",
        "successor",
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            and all(_strict_equal(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    return type(actual) is type(expected) and actual == expected


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key in encoding preflight input")
        output[key] = value
    return output


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
        raise ValueError("encoding preflight JSON is not UTF-8") from exc


def _load_object(path: str | Path) -> dict[str, Any]:
    value = strict_json_bytes(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("encoding preflight binding must contain an object")
    return value


def _assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError(f"raw training text key forbidden at {path}.{key}")
            _assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, f"{path}[{index}]")


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    expected = EXPECTED_EXTERNAL_BINDINGS.get(name)
    expected_path = expected["path"] if expected else EXPECTED_CODE_PATHS.get(name)
    if (
        expected_path is None
        or not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256"}
        or binding.get("path") != expected_path
        or not isinstance(binding.get("sha256"), str)
        or _SHA256.fullmatch(binding["sha256"]) is None
        or (expected is not None and not _strict_equal(binding, expected))
    ):
        raise ValueError(f"encoding preflight binding changed: {name}")
    declared = root / expected_path
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(declared, flags)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"missing or unsafe encoding preflight binding: {name}") from exc
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"encoding preflight binding must be regular: {name}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = declared.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino, after.st_size) != (info.st_dev, info.st_ino, info.st_size)
        or digest.hexdigest() != binding["sha256"]
    ):
        raise ValueError(f"encoding preflight binding hash or inode changed: {name}")
    return declared


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _require_no_symlink_parents(root: Path, target: Path) -> None:
    root = _lexical_absolute(root)
    target = _lexical_absolute(target)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("encoding preflight path escaped repository") from exc
    current = root
    root_info = current.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("encoding preflight repository root is unsafe")
    for part in relative.parts[:-1]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("encoding preflight output parent is unsafe")


def output_paths(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = _lexical_absolute(root / str(plan["output"]["report_path"]))
    verification = _lexical_absolute(root / str(plan["output"]["verification_path"]))
    expected_report = _lexical_absolute(root / EXPECTED_OUTPUT["report_path"])
    expected_verification = _lexical_absolute(root / EXPECTED_OUTPUT["verification_path"])
    if (report, verification) != (expected_report, expected_verification):
        raise ValueError("encoding preflight output path changed")
    evidence_root = _lexical_absolute(root / "evidence")
    if not report.is_relative_to(evidence_root) or not verification.is_relative_to(evidence_root):
        raise ValueError("encoding preflight output escaped evidence root")
    _require_no_symlink_parents(root, report)
    _require_no_symlink_parents(root, verification)
    return report, verification


def successor_output_paths(root: Path) -> dict[str, Path]:
    paths = {
        key: _lexical_absolute(root / value) for key, value in EXPECTED_SUCCESSOR_OUTPUT.items()
    }
    encoded_root = _lexical_absolute(root / "artifacts/lumi-sft-token-consistent-shards-v0.2")
    if not paths["encoded_shards_path"].is_relative_to(encoded_root):
        raise ValueError("future encoded shard path escaped private root")
    evidence_root = _lexical_absolute(root / "evidence")
    if any(
        not path.is_relative_to(evidence_root)
        for key, path in paths.items()
        if key != "encoded_shards_path"
    ):
        raise ValueError("future encoding evidence path escaped evidence root")
    for path in paths.values():
        _require_no_symlink_parents(root, path)
    return paths


def _validate_input_metadata(root: Path) -> dict[str, Any]:
    private_root = root / "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private"
    _require_no_symlink_parents(root, private_root / "sentinel")
    private_root_info = private_root.lstat()
    if (
        stat.S_ISLNK(private_root_info.st_mode)
        or not stat.S_ISDIR(private_root_info.st_mode)
        or stat.S_IMODE(private_root_info.st_mode) != 0o700
        or private_root_info.st_uid != os.geteuid()
    ):
        raise ValueError("sealed private input root metadata changed")
    observed: dict[str, Any] = {
        "private_root": {
            "directory": True,
            "mode": "0700",
            "owner_matches_process": True,
        }
    }
    for split, expected in EXPECTED_INPUT["splits"].items():
        path = root / expected["path"]
        _require_no_symlink_parents(root, path)
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"missing sealed private input: {split}") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != expected["bytes"]
            or info.st_uid != os.geteuid()
        ):
            raise ValueError(f"sealed private input metadata changed: {split}")
        observed[split] = {
            "bytes": info.st_size,
            "mode": "0600",
            "nlink": info.st_nlink,
            "owner_matches_process": True,
            "regular_file": True,
            "row_reads": 0,
        }
    if not _strict_equal(observed, EXPECTED_INPUT_METADATA):
        raise ValueError("sealed private input metadata inventory changed")
    return observed


def _project_shards(tokens: int, maximum_record_tokens: int, shard_cap: int) -> tuple[int, int]:
    minimum = (tokens + shard_cap - 1) // shard_cap
    guaranteed_fill = shard_cap - maximum_record_tokens + 1
    maximum = (tokens + guaranteed_fill - 1) // guaranteed_fill
    return minimum, maximum


def _validate_projection() -> None:
    train_min, train_max = _project_shards(33_596_489, 4_095, 4_194_304)
    validation_min, validation_max = _project_shards(2_295_902, 4_095, 4_194_304)
    if (
        (train_min, train_max) != (9, 9)
        or (validation_min, validation_max) != (1, 1)
        or EXPECTED_PROJECTION["binary_token_and_label_bytes"] != (33_596_489 + 2_295_902) * 8
        or EXPECTED_PROJECTION["encoded_index_maximum_bytes"] != 329_200 * 175
        or EXPECTED_PROJECTION["maximum_projected_write_bytes"]
        != 287_139_128 + 57_610_000 + 4_194_304
        or EXPECTED_PROJECTION["maximum_projected_write_bytes"]
        > EXPECTED_SUCCESSOR_LIMITS["maximum_private_write_bytes"]
    ):
        raise ValueError("encoding shard or byte projection changed")


def _validate_chain(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    resolved = {
        name: _resolve_binding(root, binding, name) for name, binding in plan["bindings"].items()
    }
    seal_plan = seal_v3.load_plan(
        resolved["learning_seal_plan"],
        repository_root=root,
        check_candidate_files=False,
    )
    manifest = _load_object(resolved["learning_seal"])
    verification = _load_object(resolved["learning_seal_verification"])
    seal_v3.validate_manifest(manifest, plan=seal_plan)
    seal_v3.validate_verification(verification, plan=seal_plan, manifest=manifest)
    if (
        manifest.get("plan_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["learning_seal_plan"]["sha256"]
        or manifest.get("seal_sha256") != EXPECTED_INPUT["learning_seal_internal_sha256"]
        or verification.get("manifest_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["learning_seal"]["sha256"]
        or verification.get("manifest_internal_sha256")
        != EXPECTED_INPUT["learning_seal_internal_sha256"]
        or verification.get("verification_sha256")
        != EXPECTED_INPUT["learning_seal_verification_internal_sha256"]
        or not _strict_equal(
            manifest.get("production_blockers_preserved"), EXPECTED_PRODUCTION_BLOCKERS
        )
        or manifest.get("production_eligible") is not False
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("learning-candidate seal evidence chain changed")
    for split, expected in EXPECTED_INPUT["splits"].items():
        manifest_split = manifest.get("output", {}).get("splits", {}).get(split)
        audit_split = verification.get("observed_outputs", {}).get("splits", {}).get(split)
        expected_manifest_split = {
            "bytes": expected["bytes"],
            "record_id_sequence_sha256": expected["record_id_sequence_sha256"],
            "records": expected["records"],
            "serialized_tokens": expected["serialized_tokens"],
            "sha256": expected["file_sha256"],
            "supervised_tokens": expected["supervised_tokens"],
        }
        expected_audit_split = {
            key: expected_manifest_split[key]
            for key in ("bytes", "record_id_sequence_sha256", "records", "sha256")
        }
        if not _strict_equal(manifest_split, expected_manifest_split) or not _strict_equal(
            audit_split, expected_audit_split
        ):
            raise ValueError(f"sealed input split evidence changed: {split}")
    if (
        manifest.get("output", {}).get("candidate_row_reads") != EXPECTED_INPUT["records"]
        or manifest.get("output", {}).get("unique_record_ids") != EXPECTED_INPUT["records"]
        or verification.get("observed_outputs", {}).get("candidate_files_opened") != 0
        or verification.get("observed_outputs", {}).get("output_audit_row_reads")
        != EXPECTED_INPUT["records"]
    ):
        raise ValueError("sealed input record accounting changed")

    mix_report = _load_object(resolved["candidate_mix_report"])
    mix_verification = _load_object(resolved["candidate_mix_verification"])
    mix_v18.validate_report(mix_report)
    mix_v18.validate_verification(mix_verification)
    source_metrics = mix_report.get("source_metrics")
    manifest_source_metrics = manifest.get("output", {}).get("source_metrics")
    if (
        not isinstance(source_metrics, Mapping)
        or len(source_metrics) != 20
        or max(item["maximum_record_tokens"] for item in source_metrics.values()) != 4_095
        or sum(item["records"] for item in source_metrics.values()) != EXPECTED_INPUT["records"]
        or mix_report.get("report_sha256")
        != EXPECTED_BINDING_CHAIN["candidate_mix_report_internal_sha256"]
        or mix_verification.get("report_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"]["sha256"]
        or mix_verification.get("report_internal_sha256")
        != EXPECTED_BINDING_CHAIN["candidate_mix_report_internal_sha256"]
        or mix_verification.get("verification_sha256")
        != EXPECTED_BINDING_CHAIN["candidate_mix_verification_internal_sha256"]
        or not isinstance(manifest_source_metrics, Mapping)
        or set(manifest_source_metrics) != set(source_metrics)
    ):
        raise ValueError("frozen tokenizer ledger changed")
    for source_id, metrics in source_metrics.items():
        sealed = manifest_source_metrics[source_id]
        if (
            sealed.get("records") != metrics.get("records")
            or sealed.get("candidate_sha256") != metrics.get("candidate_file_sha256")
            or not _strict_equal(sealed.get("by_split"), metrics.get("by_split"))
        ):
            raise ValueError(f"sealed source tokenizer ledger changed: {source_id}")

    template = load_chat_template(resolved["chat_template"])
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, template)
    tokenizer_json = _load_object(resolved["tokenizer"])
    tokenizer_model = tokenizer_json.get("model")
    tokenizer_vocab = tokenizer_model.get("vocab") if isinstance(tokenizer_model, Mapping) else None
    added_tokens = tokenizer_json.get("added_tokens")
    vocab_ids = list(tokenizer_vocab.values()) if isinstance(tokenizer_vocab, Mapping) else []
    if (
        template.context_window != EXPECTED_ENCODING_CONTRACT["context_window"]
        or template.loss_roles != frozenset(EXPECTED_ENCODING_CONTRACT["loss_roles"])
        or template.ignore_index != EXPECTED_ENCODING_CONTRACT["label_ignore_index"]
        or not isinstance(tokenizer_model, Mapping)
        or tokenizer_model.get("type") != EXPECTED_ENCODING_CONTRACT["tokenizer_model"]
        or len(vocab_ids) != EXPECTED_ENCODING_CONTRACT["tokenizer_vocab_size"]
        or len(set(vocab_ids)) != EXPECTED_ENCODING_CONTRACT["tokenizer_unique_id_count"]
        or min(vocab_ids, default=-1) != EXPECTED_ENCODING_CONTRACT["tokenizer_id_minimum"]
        or max(vocab_ids, default=-1) != EXPECTED_ENCODING_CONTRACT["tokenizer_id_maximum"]
        or not isinstance(added_tokens, list)
        or len(added_tokens) != EXPECTED_ENCODING_CONTRACT["tokenizer_added_token_count"]
        or [item.get("id") for item in added_tokens] != list(range(11))
    ):
        raise ValueError("encoding chat contract changed")
    input_metadata = _validate_input_metadata(root)
    _validate_projection()
    future_paths = successor_output_paths(root)
    output_absence = {
        "encoded_shards_absent": not os.path.lexists(future_paths["encoded_shards_path"]),
        "receipt_absent": not os.path.lexists(future_paths["receipt_path"]),
        "successor_seal_absent": not os.path.lexists(future_paths["successor_seal_path"]),
        "verification_absent": not os.path.lexists(future_paths["verification_path"]),
    }
    if not _strict_equal(output_absence, EXPECTED_OUTPUT_ABSENCE):
        raise FileExistsError("future private encoding output already exists")
    return {"input_metadata": input_metadata, "output_absence": output_absence}


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = root / supplied
    plan_path = _lexical_absolute(supplied)
    if plan_path != _lexical_absolute(root / EXPECTED_PLAN_PATH):
        raise ValueError("encoding preflight plan path changed")
    info = plan_path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("encoding preflight plan must be regular")
    plan = _load_object(plan_path)
    if (
        set(plan) != _PLAN_KEYS
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("observed_at") != EXPECTED_OBSERVED_AT
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
        or not _strict_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION)
        or not _strict_equal(plan.get("input"), EXPECTED_INPUT)
        or not _strict_equal(plan.get("limits"), EXPECTED_LIMITS)
        or not _strict_equal(plan.get("output"), EXPECTED_OUTPUT)
        or not _strict_equal(plan.get("successor"), EXPECTED_SUCCESSOR)
    ):
        raise ValueError("encoding preflight plan boundary changed")
    bindings = plan.get("bindings")
    expected_names = set(EXPECTED_EXTERNAL_BINDINGS) | set(EXPECTED_CODE_PATHS)
    if not isinstance(bindings, Mapping) or set(bindings) != expected_names:
        raise ValueError("encoding preflight binding set changed")
    observed = _validate_chain(plan, root)
    runtime_plan = dict(plan)
    runtime_plan["_runtime_observed"] = observed
    return runtime_plan


def build_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    del repository_root
    observed = plan.get("_runtime_observed")
    if (
        not isinstance(observed, Mapping)
        or not _strict_equal(observed.get("input_metadata"), EXPECTED_INPUT_METADATA)
        or not _strict_equal(observed.get("output_absence"), EXPECTED_OUTPUT_ABSENCE)
    ):
        raise ValueError("encoding preflight runtime observation is absent")
    report: dict[str, Any] = {
        "authorization_boundary": EXPECTED_AUTHORIZATION,
        "binding_chain": EXPECTED_BINDING_CHAIN,
        "decision": EXPECTED_DECISION,
        "encoding_contract": EXPECTED_ENCODING_CONTRACT,
        "input": EXPECTED_INPUT,
        "input_metadata": observed["input_metadata"],
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": EXPECTED_OBSERVED_AT,
        "output_absence": observed["output_absence"],
        "plan_file_sha256": plan_file_sha256,
        "production_blockers_preserved": EXPECTED_PRODUCTION_BLOCKERS,
        "production_eligible": False,
        "projection": EXPECTED_PROJECTION,
        "raw_training_text_retained": False,
        "resource_accounting": EXPECTED_RESOURCE_PASS_ONE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "successor": EXPECTED_SUCCESSOR,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


_REPORT_KEYS = frozenset(
    {
        "authorization_boundary",
        "binding_chain",
        "decision",
        "encoding_contract",
        "input",
        "input_metadata",
        "next_gate",
        "observed_at",
        "output_absence",
        "plan_file_sha256",
        "production_blockers_preserved",
        "production_eligible",
        "projection",
        "raw_training_text_retained",
        "report_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "successor",
    }
)


def validate_report(report: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        set(report) != _REPORT_KEYS
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("observed_at") != EXPECTED_OBSERVED_AT
        or report.get("report_sha256") != sha256_value(unsigned)
        or not isinstance(report.get("plan_file_sha256"), str)
        or _SHA256.fullmatch(report["plan_file_sha256"]) is None
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or not _strict_equal(report.get("authorization_boundary"), EXPECTED_AUTHORIZATION)
        or not _strict_equal(report.get("binding_chain"), EXPECTED_BINDING_CHAIN)
        or not _strict_equal(report.get("decision"), EXPECTED_DECISION)
        or not _strict_equal(report.get("encoding_contract"), EXPECTED_ENCODING_CONTRACT)
        or not _strict_equal(report.get("input"), EXPECTED_INPUT)
        or not _strict_equal(report.get("input_metadata"), EXPECTED_INPUT_METADATA)
        or not _strict_equal(report.get("next_gate"), EXPECTED_NEXT_GATE)
        or not _strict_equal(report.get("output_absence"), EXPECTED_OUTPUT_ABSENCE)
        or not _strict_equal(
            report.get("production_blockers_preserved"), EXPECTED_PRODUCTION_BLOCKERS
        )
        or not _strict_equal(report.get("projection"), EXPECTED_PROJECTION)
        or not _strict_equal(report.get("resource_accounting"), EXPECTED_RESOURCE_PASS_ONE)
        or not _strict_equal(report.get("successor"), EXPECTED_SUCCESSOR)
    ):
        raise ValueError("encoding preflight report boundary changed")
    _assert_raw_text_free(report)


def build_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": EXPECTED_VERIFICATION_CHECKS,
        "next_gate": EXPECTED_NEXT_GATE,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "resource_accounting": EXPECTED_RESOURCE_PASS_TWO,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


_VERIFICATION_KEYS = frozenset(
    {
        "checks",
        "next_gate",
        "production_eligible",
        "raw_training_text_retained",
        "report_file_sha256",
        "report_internal_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "verification_sha256",
    }
)


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in verification.items() if key != "verification_sha256"}
    if (
        set(verification) != _VERIFICATION_KEYS
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or not isinstance(verification.get("report_file_sha256"), str)
        or _SHA256.fullmatch(verification["report_file_sha256"]) is None
        or not isinstance(verification.get("report_internal_sha256"), str)
        or _SHA256.fullmatch(verification["report_internal_sha256"]) is None
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or not _strict_equal(verification.get("checks"), EXPECTED_VERIFICATION_CHECKS)
        or not _strict_equal(verification.get("next_gate"), EXPECTED_NEXT_GATE)
        or not _strict_equal(verification.get("resource_accounting"), EXPECTED_RESOURCE_PASS_TWO)
    ):
        raise ValueError("encoding preflight verification boundary changed")
    _assert_raw_text_free(verification)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_marker(path: Path, marker: bytes) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("encoding preflight marker inode changed")
        if os.write(descriptor, marker) != len(marker):
            raise OSError("short encoding preflight marker write")
        os.fsync(descriptor)
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return identity


def finish_marker(path: Path, marker: bytes, payload: bytes, identity: tuple[int, int]) -> None:
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or (info.st_dev, info.st_ino) != identity
            or os.read(descriptor, len(marker) + 1) != marker
        ):
            raise ValueError("encoding preflight marker changed")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short encoding preflight evidence write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_ENCODING_CONTRACT",
    "EXPECTED_INPUT",
    "EXPECTED_INPUT_METADATA",
    "EXPECTED_LIMITS",
    "EXPECTED_NEXT_GATE",
    "EXPECTED_OUTPUT",
    "EXPECTED_OUTPUT_ABSENCE",
    "EXPECTED_PLAN_PATH",
    "EXPECTED_PROJECTION",
    "EXPECTED_REPORT_MARKER",
    "EXPECTED_RESOURCE_PASS_ONE",
    "EXPECTED_RESOURCE_PASS_TWO",
    "EXPECTED_SUCCESSOR",
    "EXPECTED_VERIFICATION_MARKER",
    "REPORT_SCHEMA_VERSION",
    "REPORT_STATUS",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_STATUS",
    "_assert_raw_text_free",
    "_load_object",
    "_project_shards",
    "build_report",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_file_bytes",
    "load_plan",
    "output_paths",
    "sha256_bytes",
    "sha256_value",
    "start_marker",
    "strict_json_bytes",
    "validate_report",
    "validate_verification",
]
