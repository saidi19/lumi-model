"""Metadata-only preflight for the registry-v0.24 private learning-candidate seal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from . import posttraining_candidate_mix_reaudit_v18 as mix_v18
from . import posttraining_private_research_data_contract_v2 as contract_v2
from .posttraining_data import SFTPolicy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_readiness import validate_final_base_binding
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-preflight.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-preflight-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-preflight-verification.v0.2"
PLAN_STATUS = "authorized_registry_v0_24_metadata_only_seal_preflight"
REPORT_STATUS = "PASS_registry_v0_24_private_seal_successor_ready_no_materialization"
VERIFICATION_STATUS = "PASS_verified_registry_v0_24_metadata_only_seal_preflight"

EXPECTED_AUTHORIZATION_TEXT = "ok step by stpe kor tor kaj"
EXPECTED_AUTHORIZATION_BYTES = 27
EXPECTED_AUTHORIZATION_SHA256 = "c0593c0cb29576069f8bd4a141f3047080b0c0ad8fe287b2e454d48b043f5ee2"
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "candidate_row_access_authorized": False,
    "data_contract_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "encoded_shard_materialization_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "metadata_only_preflight_authorized": True,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_row_access_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_row_reads": 0,
    "maximum_evidence_writes": 2,
    "maximum_factual_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 4_194_304,
    "maximum_registry_or_contract_writes": 0,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 15,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_training_actions": 0,
}
EXPECTED_OUTPUT = {
    "candidate_materialization": False,
    "dataset_seal": False,
    "report_path": "evidence/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json",
    "verification_path": (
        "evidence/lumi-4b-sft-learning-candidate-seal-preflight-verification-v0.2.json"
    ),
}
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_learning_candidate_seal_preflight_v2.py",
    "runner": "scripts/preflight_sft_learning_candidate_seal_v2.py",
    "test": "tests/test_learning_candidate_seal_preflight_v2.py",
    "verifier": "scripts/verify_sft_learning_candidate_seal_preflight_v2.py",
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
        "path": ("evidence/lumi-4b-sft-private-research-data-contract-verification-v0.2.json"),
        "sha256": "8a5f2cbb3de53eaefd13069d0b114ca3cd1cf6060d9c2b187754dae81b631aba",
    },
    "data_contract_implementation": {
        "path": "src/lumi/posttraining_private_research_data_contract_v2.py",
        "sha256": "80a358e092bee51c8851fe5739d1b8acf56e2995dfa50e41faf4cece0d644d81",
    },
    "final_base_binding": {
        "path": "configs/lumi-4b-final-base-binding-v0.1.json",
        "sha256": "1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172",
    },
    "policy_loader": {
        "path": "src/lumi/posttraining_data.py",
        "sha256": "e520c3771869b7221848a456e59c3b15a279fb1ebdf700646600c774e25b74b5",
    },
    "predecessor_data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.1.json",
        "sha256": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    },
    "production_seal": {
        "path": "src/lumi/posttraining_seal.py",
        "sha256": "52b9d93ceaf756497b7ccc4805b51d76e22490269a61cd644ce5a1b12e946155",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.24.json",
        "sha256": "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    },
    "source_registry_loader": {
        "path": "src/lumi/posttraining_sources.py",
        "sha256": "10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50",
    },
    "tokenizer": {
        "path": (
            "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
            "staging/input/tokenizer/tokenizer.json"
        ),
        "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    },
}
EXPECTED_BINDING_NAMES = frozenset(EXPECTED_EXTERNAL_BINDINGS) | frozenset(EXPECTED_CODE_PATHS)
EXPECTED_PLAN_KEYS = frozenset(
    {
        "authorization",
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
        "bindings",
        "limits",
        "observed_at",
        "output",
        "schema_version",
        "status",
    }
)

INCLUDED_SOURCE_IDS = tuple(contract_v2.INCLUDED_SOURCE_IDS)
EXCLUDED_SOURCE_IDS = tuple(contract_v2.EXCLUDED_SOURCE_IDS)
EVALUATION_SPLIT_PROHIBITIONS = dict(contract_v2.EVALUATION_SPLIT_PROHIBITIONS)
AI_SOURCE_CONTRACTS = dict(contract_v2.AI_SOURCE_CONTRACTS)
EXPECTED_ACCEPTED_DATASET_SHA256 = (
    "1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e"
)
EXPECTED_INVENTORY = {
    "accepted_dataset_sha256": EXPECTED_ACCEPTED_DATASET_SHA256,
    "included_sources": 20,
    "records": 329_200,
    "train_records": 309_643,
    "train_supervised_tokens": 11_855_217,
    "validation_records": 19_557,
    "validation_supervised_tokens": 939_293,
}
EXPECTED_SUCCESSOR_LIMITS = {
    "candidate_source_files": 20,
    "maximum_candidate_row_reads": 329_200,
    "maximum_full_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_output_audit_row_reads": 329_200,
    "maximum_private_write_bytes": 2_147_483_648,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 180,
    "maximum_source_line_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_total_row_reads": 658_400,
    "maximum_training_actions": 0,
}
EXPECTED_SUCCESSOR_AUTHORIZATION = (
    "I approve one bounded local-only Lumi registry-v0.24 private nonproduction "
    "learning-candidate seal materialization. Bind candidate-mix v0.18 report SHA-256 "
    "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000 and "
    "verification SHA-256 43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651, "
    "data contract v0.2 SHA-256 281fba028c58f5bc0fd489c1d70694e54d37d5dffd09dcc79676d42029497d33, "
    "source registry v0.24 SHA-256 "
    "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb, "
    "final-base binding SHA-256 1faaf5398f554b4708948b3298405a0451a31fe60027da359b54685a876b9172, "
    "chat template SHA-256 bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12, "
    "and tokenizer SHA-256 1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee. "
    "Read the exact 329,200 already audited candidate records across the exact 20 included "
    "sources once in source-id lexical then existing-file order and materialize private train "
    "and validation JSONL containing exactly 309,643 train records / 11,855,217 train "
    "supervised tokens and 19,557 validation records / 939,293 validation supervised tokens, "
    "plus a raw-text-free manifest. Run one independent output verification pass. Maximum "
    "329,200 candidate-row reads, 329,200 output-audit row reads, 658,400 total row reads, "
    "two sequential passes, 180 local runtime-minutes, 2,147,483,648 private-write bytes, and "
    "zero network, GPU, inference, source-line, test-object, or training actions. Preserve all "
    "candidate sources, registry, contracts, production blockers, evaluation exclusions, and "
    "production_eligible=false. Create no encoded shards, production seal, registry mutation, "
    "SFT run, promotion, remote action, or unrelated output. No retry after a started pass."
)

EXPECTED_HYGIENE = {
    "exact_cross_source_duplicate_groups": 0,
    "exact_duplicate_extra_records": 0,
    "exact_duplicate_groups": 0,
    "generic_rejected_records": 0,
    "near_duplicate_cross_source_components": 0,
    "near_duplicate_edges": 0,
    "near_duplicate_multi_record_components": 0,
    "near_duplicate_train_validation_components": 0,
    "quarantined_records": 0,
    "source_family_cap_violations": 0,
    "strict_targeted_gap_lower_bound": 0,
    "task_floor_gap_total": 0,
    "language_floor_gap_total": 0,
    "train_validation_exact_overlap_groups": 0,
}
EXPECTED_AUTHORIZATION_BOUNDARY = {
    "candidate_materialization_authorized": False,
    "dataset_seal_authorized": False,
    "encoded_shards_authorized": False,
    "metadata_only_preflight_authorized": True,
    "production_sft_authorized": False,
    "successor_authorization_required": True,
}
EXPECTED_BASE_AND_TOKENIZER = {
    "base_checkpoint_manifest_sha256": (
        "73c852095441a4246cea7bb97f7831c364137869e00e29c815b7e22a5972f8de"
    ),
    "base_checkpoint_step": 76_294,
    "base_checkpoint_training_tokens": 80_000_000_000,
    "chat_template_sha256": EXPECTED_EXTERNAL_BINDINGS["chat_template"]["sha256"],
    "tokenizer_sha256": EXPECTED_EXTERNAL_BINDINGS["tokenizer"]["sha256"],
}
EXPECTED_BINDING_CHAIN = {
    "candidate_mix_report_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"]["sha256"],
    "candidate_mix_verification_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_verification"][
        "sha256"
    ],
    "data_contract_activation_report_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "data_contract_activation_report"
    ]["sha256"],
    "data_contract_activation_verification_sha256": EXPECTED_EXTERNAL_BINDINGS[
        "data_contract_activation_verification"
    ]["sha256"],
    "data_contract_sha256": EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"],
    "source_registry_sha256": EXPECTED_EXTERNAL_BINDINGS["source_registry"]["sha256"],
}
EXPECTED_CONTRACT_BOUNDARY = {
    "evaluation_split_prohibitions": EVALUATION_SPLIT_PROHIBITIONS,
    "external_distribution_allowed": False,
    "learning_candidate_seal_allowed": True,
    "machine_gap_source_contracts": AI_SOURCE_CONTRACTS,
    "production_dataset_seal_allowed": False,
    "production_sft_allowed": False,
    "scope": "private_research_nonproduction",
}
EXPECTED_DECISION = {
    "candidate_materialization_performed": False,
    "nonproduction_learning_candidate_seal_safe_to_authorize": True,
    "production_dataset_seal_safe_to_execute": False,
    "recommended_path": ("materialize_exact_registry_v0_24_mix_as_private_nonproduction_seal"),
}
EXPECTED_NEXT_GATE = {
    "action": "authorize_registry_v0_24_private_learning_candidate_seal_materialization",
    "authorization_required": True,
    "candidate_records": 329_200,
    "gpu_required": False,
    "production_dataset_seal": False,
    "sft_training_started": False,
}
EXPECTED_REGISTRY = {
    "excluded_source_ids": list(EXCLUDED_SOURCE_IDS),
    "included_source_ids": list(INCLUDED_SOURCE_IDS),
    "production_eligible_source_count": 0,
    "source_count": 27,
}
EXPECTED_SUCCESSOR = {
    "exact_authorization": EXPECTED_SUCCESSOR_AUTHORIZATION,
    "limits": EXPECTED_SUCCESSOR_LIMITS,
    "output_boundary": {
        "encoded_shards_authorized": False,
        "private_train_and_validation_jsonl": True,
        "production_dataset_seal": False,
        "raw_text_allowed_only_in_private_train_validation_jsonl": True,
        "raw_text_in_manifest_or_verification": False,
        "sft_training_authorized": False,
    },
}
EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE = {
    "candidate_row_reads": 0,
    "dataset_or_shard_writes": 0,
    "evidence_writes": 1,
    "factual_passes": 1,
    "gpu_use": 0,
    "inference_calls": 0,
    "network_bytes": 0,
    "registry_or_contract_writes": 0,
    "source_row_reads": 0,
    "test_object_reads": 0,
    "training_actions": 0,
}
EXPECTED_RESOURCE_ACCOUNTING_PASS_TWO = {
    **EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE,
    "evidence_writes": 2,
    "factual_passes": 2,
}
EXPECTED_VERIFICATION_CHECKS = {
    "accepted_dataset_hash_and_exact_inventory_bound": True,
    "all_candidate_source_and_test_row_reads_zero": True,
    "base_chat_and_tokenizer_bindings_verified": True,
    "candidate_mix_v0_18_report_and_verification_bound": True,
    "data_contract_v0_2_activation_chain_bound": True,
    "exact_twenty_included_and_seven_excluded_sources_bound": True,
    "hygiene_and_all_task_language_floor_gaps_zero": True,
    "no_materialization_seal_shards_gpu_inference_or_training": True,
    "predecessor_v0_17_seal_chain_not_reused": True,
    "production_and_external_distribution_flags_false": True,
    "raw_training_text_absent": True,
    "registry_v0_24_has_zero_production_eligible_sources": True,
    "successor_requires_fresh_explicit_authorization": True,
}
EXPECTED_REPORT_KEYS = frozenset(
    {
        "authorization_boundary",
        "base_and_tokenizer",
        "binding_chain",
        "contract_boundary",
        "decision",
        "hygiene",
        "inventory",
        "next_gate",
        "observed_at",
        "plan_file_sha256",
        "production_eligible",
        "raw_training_text_retained",
        "registry",
        "report_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "successor",
    }
)
EXPECTED_VERIFICATION_KEYS = frozenset(
    {
        "checks",
        "next_gate",
        "production_eligible",
        "report_file_sha256",
        "report_internal_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "verification_sha256",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_TEXT_KEYS = frozenset(
    {
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "response",
        "text",
        "user_prompt",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) is float and actual == expected
    return type(actual) is type(expected) and actual == expected


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("seal preflight binding must contain a JSON object")
    return value


def _assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RAW_TEXT_KEYS:
                raise ValueError(f"raw candidate text key forbidden at {path}.{key}")
            _assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, f"{path}[{index}]")


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    if name in EXPECTED_EXTERNAL_BINDINGS:
        expected = EXPECTED_EXTERNAL_BINDINGS[name]
        if not _strict_json_equal(binding, expected):
            raise ValueError(f"seal preflight external binding changed: {name}")
    else:
        expected_path = EXPECTED_CODE_PATHS.get(name)
        if (
            expected_path is None
            or not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError(f"seal preflight code binding changed: {name}")
    expected_path = (
        EXPECTED_EXTERNAL_BINDINGS[name]["path"]
        if name in EXPECTED_EXTERNAL_BINDINGS
        else EXPECTED_CODE_PATHS[name]
    )
    declared = root / str(binding["path"])
    expected_declared = root / expected_path
    if declared != expected_declared:
        raise ValueError(f"seal preflight binding path changed: {name}")
    try:
        file_stat = declared.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing seal preflight binding: {name}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"seal preflight binding must be a regular file: {name}")
    resolved = declared.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing seal preflight binding: {name}")
    if file_sha256(resolved) != binding["sha256"]:
        raise ValueError(f"seal preflight binding hash changed: {name}")
    return resolved


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    expected_plan_path = (
        root / "configs/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json"
    ).resolve()
    if plan_path != expected_plan_path:
        raise ValueError("seal preflight plan path changed")
    plan = _load_object(plan_path)
    if set(plan) != EXPECTED_PLAN_KEYS:
        raise ValueError("seal preflight plan keys changed")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("observed_at") != EXPECTED_OBSERVED_AT
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or type(plan.get("authorization_bytes")) is not int
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or not _strict_json_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION)
        or not _strict_json_equal(plan.get("limits"), EXPECTED_LIMITS)
        or not _strict_json_equal(plan.get("output"), EXPECTED_OUTPUT)
    ):
        raise ValueError("seal preflight plan boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("seal preflight binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"seal preflight binding must be an object: {name}")
        _resolve_binding(root, binding, name)
    return plan


def output_paths(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    evidence_root = (root / "evidence").resolve()
    if not report.is_relative_to(evidence_root) or not verification.is_relative_to(evidence_root):
        raise ValueError("seal preflight evidence escaped evidence root")
    return report, verification


def _inventory_from_mix(report: Mapping[str, Any]) -> dict[str, Any]:
    source_metrics = report["source_metrics"]
    return {
        "accepted_dataset_sha256": report["generic_sft_validation"]["accepted_dataset_sha256"],
        "included_sources": len(source_metrics),
        "records": sum(item["records"] for item in source_metrics.values()),
        "train_records": sum(
            item["by_split"]["train"]["records"] for item in source_metrics.values()
        ),
        "train_supervised_tokens": sum(
            item["by_split"]["train"]["supervised_tokens"] for item in source_metrics.values()
        ),
        "validation_records": sum(
            item["by_split"]["validation"]["records"] for item in source_metrics.values()
        ),
        "validation_supervised_tokens": sum(
            item["by_split"]["validation"]["supervised_tokens"] for item in source_metrics.values()
        ),
    }


def _hygiene_from_mix(report: Mapping[str, Any]) -> dict[str, int]:
    exact = report["exact_duplicate_audit"]
    near = report["near_duplicate_audit"]
    closure = report["closure"]
    pool = report["candidate_pool"]
    return {
        "exact_cross_source_duplicate_groups": exact["cross_source_duplicate_groups"],
        "exact_duplicate_extra_records": exact["duplicate_extra_records"],
        "exact_duplicate_groups": exact["duplicate_groups"],
        "generic_rejected_records": report["generic_sft_validation"]["rejected"],
        "near_duplicate_cross_source_components": near["cross_source_components"],
        "near_duplicate_edges": near["edge_count"],
        "near_duplicate_multi_record_components": near["multi_record_components"],
        "near_duplicate_train_validation_components": near["train_validation_components"],
        "quarantined_records": report["quarantine"]["total_records"],
        "source_family_cap_violations": len(closure["source_family_cap_violations"]),
        "strict_targeted_gap_lower_bound": pool["strict_targeted_gap_lower_bound"],
        "task_floor_gap_total": sum(closure["task_floor_gaps_behavior_safe"].values()),
        "language_floor_gap_total": sum(closure["language_floor_gaps_behavior_safe"].values()),
        "train_validation_exact_overlap_groups": exact["train_validation_overlap_groups"],
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: _resolve_binding(root, binding, name) for name, binding in plan["bindings"].items()
    }

    mix_report = _load_object(resolved["candidate_mix_report"])
    mix_verification = _load_object(resolved["candidate_mix_verification"])
    mix_v18.validate_report(mix_report)
    mix_v18.validate_verification(mix_verification)
    if mix_verification.get("report_file_sha256") != EXPECTED_EXTERNAL_BINDINGS[
        "candidate_mix_report"
    ]["sha256"] or mix_verification.get("report_internal_sha256") != mix_report.get(
        "report_sha256"
    ):
        raise ValueError("candidate-mix report and verification chain changed")

    predecessor_contract = _load_object(resolved["predecessor_data_contract"])
    data_contract = _load_object(resolved["data_contract"])
    contract_v2.validate_contract(data_contract, predecessor=predecessor_contract)
    contract_report = _load_object(resolved["data_contract_activation_report"])
    contract_verification = _load_object(resolved["data_contract_activation_verification"])
    contract_v2.validate_report(contract_report)
    contract_v2.validate_verification(contract_verification)
    if (
        contract_report["data_contract"]["file_sha256"]
        != EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"]
        or contract_verification["data_contract_file_sha256"]
        != EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"]
        or contract_verification["report_file_sha256"]
        != EXPECTED_EXTERNAL_BINDINGS["data_contract_activation_report"]["sha256"]
        or contract_verification["report_internal_sha256"] != contract_report["report_sha256"]
    ):
        raise ValueError("private contract activation chain changed")

    policy = SFTPolicy.from_mapping(data_contract)
    learning = policy.learning_candidate
    if learning is None:
        raise ValueError("private learning-candidate contract is absent")
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["source_registry"]["sha256"],
    )
    if (
        len(registry.sources) != 27
        or any(item.production_eligible for item in registry.sources)
        or set(registry.by_id) != set(INCLUDED_SOURCE_IDS) | set(EXCLUDED_SOURCE_IDS)
        or learning["included_source_ids"] != frozenset(INCLUDED_SOURCE_IDS)
        or learning["excluded_source_ids"] != frozenset(EXCLUDED_SOURCE_IDS)
        or learning["evaluation_split_prohibitions"]
        != {key: frozenset(value) for key, value in EVALUATION_SPLIT_PROHIBITIONS.items()}
        or not _strict_json_equal(
            data_contract["learning_candidate"]["ai_assisted_source_contracts"],
            AI_SOURCE_CONTRACTS,
        )
    ):
        raise ValueError("private learning-candidate registry or policy changed")

    base_binding = validate_final_base_binding(_load_object(resolved["final_base_binding"]))
    chat_template = load_chat_template(resolved["chat_template"])
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, chat_template)

    inventory = _inventory_from_mix(mix_report)
    hygiene = _hygiene_from_mix(mix_report)
    source_ids = tuple(sorted(mix_report["source_metrics"]))
    excluded_ids = tuple(
        mix_report["excluded_source_gate"]["registry_sources_not_in_candidate_inventory"]
    )
    if (
        not _strict_json_equal(inventory, EXPECTED_INVENTORY)
        or not _strict_json_equal(hygiene, EXPECTED_HYGIENE)
        or source_ids != INCLUDED_SOURCE_IDS
        or excluded_ids != EXCLUDED_SOURCE_IDS
        or mix_report.get("production_eligible") is not False
    ):
        raise ValueError("current candidate-mix aggregate boundary changed")

    report: dict[str, Any] = {
        "authorization_boundary": EXPECTED_AUTHORIZATION_BOUNDARY,
        "base_and_tokenizer": EXPECTED_BASE_AND_TOKENIZER,
        "binding_chain": EXPECTED_BINDING_CHAIN,
        "contract_boundary": EXPECTED_CONTRACT_BOUNDARY,
        "decision": EXPECTED_DECISION,
        "hygiene": hygiene,
        "inventory": inventory,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "registry": EXPECTED_REGISTRY,
        "resource_accounting": EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "successor": EXPECTED_SUCCESSOR,
    }
    if not _strict_json_equal(
        report["base_and_tokenizer"],
        {
            "base_checkpoint_manifest_sha256": base_binding["checkpoint"]["manifest_sha256"],
            "base_checkpoint_step": base_binding["checkpoint"]["step"],
            "base_checkpoint_training_tokens": base_binding["checkpoint"]["training_tokens"],
            "chat_template_sha256": EXPECTED_EXTERNAL_BINDINGS["chat_template"]["sha256"],
            "tokenizer_sha256": EXPECTED_EXTERNAL_BINDINGS["tokenizer"]["sha256"],
        },
    ):
        raise ValueError("final-base checkpoint boundary changed")
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        set(report) != EXPECTED_REPORT_KEYS
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("observed_at") != EXPECTED_OBSERVED_AT
        or report.get("report_sha256") != sha256_value(unsigned)
        or not isinstance(report.get("plan_file_sha256"), str)
        or _SHA256.fullmatch(report["plan_file_sha256"]) is None
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or not _strict_json_equal(
            report.get("authorization_boundary"), EXPECTED_AUTHORIZATION_BOUNDARY
        )
        or not _strict_json_equal(report.get("base_and_tokenizer"), EXPECTED_BASE_AND_TOKENIZER)
        or not _strict_json_equal(report.get("binding_chain"), EXPECTED_BINDING_CHAIN)
        or not _strict_json_equal(report.get("contract_boundary"), EXPECTED_CONTRACT_BOUNDARY)
        or not _strict_json_equal(report.get("decision"), EXPECTED_DECISION)
        or not _strict_json_equal(report.get("inventory"), EXPECTED_INVENTORY)
        or not _strict_json_equal(report.get("hygiene"), EXPECTED_HYGIENE)
        or not _strict_json_equal(
            report.get("resource_accounting"), EXPECTED_RESOURCE_ACCOUNTING_PASS_ONE
        )
        or not _strict_json_equal(report.get("next_gate"), EXPECTED_NEXT_GATE)
        or not _strict_json_equal(report.get("registry"), EXPECTED_REGISTRY)
        or not _strict_json_equal(report.get("successor"), EXPECTED_SUCCESSOR)
    ):
        raise ValueError("registry-v0.24 seal preflight report boundary changed")
    _assert_raw_text_free(report)


def build_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": EXPECTED_VERIFICATION_CHECKS,
        "next_gate": report["next_gate"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "resource_accounting": EXPECTED_RESOURCE_ACCOUNTING_PASS_TWO,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in verification.items() if key != "verification_sha256"}
    checks = verification.get("checks")
    if (
        set(verification) != EXPECTED_VERIFICATION_KEYS
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or not isinstance(verification.get("report_file_sha256"), str)
        or _SHA256.fullmatch(verification["report_file_sha256"]) is None
        or not isinstance(verification.get("report_internal_sha256"), str)
        or _SHA256.fullmatch(verification["report_internal_sha256"]) is None
        or verification.get("production_eligible") is not False
        or not _strict_json_equal(checks, EXPECTED_VERIFICATION_CHECKS)
        or not _strict_json_equal(verification.get("next_gate"), EXPECTED_NEXT_GATE)
        or not _strict_json_equal(
            verification.get("resource_accounting"),
            EXPECTED_RESOURCE_ACCOUNTING_PASS_TWO,
        )
    ):
        raise ValueError("registry-v0.24 seal preflight verification boundary changed")
    _assert_raw_text_free(verification)


def _reserve_output(path: Path) -> BinaryIO:
    if not path.parent.is_dir():
        raise ValueError("registry-v0.24 seal preflight output parent is absent")
    handle = path.open("xb")
    try:
        handle.flush()
        os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        handle.close()
        raise
    return handle


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("registry-v0.24 seal preflight output already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError("stale registry-v0.24 seal preflight temporary exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_HYGIENE",
    "EXPECTED_INVENTORY",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "EXPECTED_SUCCESSOR_AUTHORIZATION",
    "EXPECTED_SUCCESSOR_LIMITS",
    "REPORT_STATUS",
    "VERIFICATION_STATUS",
    "_assert_raw_text_free",
    "_load_object",
    "_reserve_output",
    "_write_exclusive",
    "build_report",
    "build_verification",
    "file_sha256",
    "json_file_bytes",
    "load_plan",
    "output_paths",
    "sha256_value",
    "validate_report",
    "validate_verification",
]
