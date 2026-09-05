"""Registry-v0.24 private nonproduction learning-candidate seal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from . import posttraining_candidate_mix_reaudit_v18 as mix_v18
from . import posttraining_learning_candidate_seal_preflight_v2 as preflight_v2
from . import posttraining_private_research_data_contract_v2 as contract_v2
from .posttraining_data import SFTPolicy
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_readiness import validate_final_base_binding
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-plan.v0.3"
MANIFEST_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal.v0.3"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-learning-candidate-seal-verification.v0.3"
PLAN_STATUS = "authorized_registry_v0_24_private_nonproduction_materialization"
MANIFEST_STATUS = "PASS_registry_v0_24_private_nonproduction_learning_candidate_sealed"
VERIFICATION_STATUS = "PASS_verified_registry_v0_24_private_nonproduction_learning_candidate_seal"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-learning-candidate-seal-v0.3.json")
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_AUTHORIZATION_TEXT = "next shuru kor"
EXPECTED_AUTHORIZATION_BYTES = 14
EXPECTED_AUTHORIZATION_SHA256 = "76f0cac58eacb1c6b3547309557745910152a03ec87cbb7cf9923e8bb9431b66"
EXPECTED_AUTHORIZATION = {
    "candidate_row_access_authorized": True,
    "data_contract_mutation_authorized": False,
    "encoded_shard_materialization_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "output_audit_authorized": True,
    "private_nonproduction_materialization_authorized": True,
    "production_dataset_seal_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_row_access_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_INPUT = {
    "accepted_dataset_sha256": ("1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e"),
    "candidate_input_bytes": 529_207_367,
    "candidate_source_files": 20,
    "records": 329_200,
    "train_records": 309_643,
    "train_serialized_tokens": 33_596_489,
    "train_supervised_tokens": 11_855_217,
    "validation_records": 19_557,
    "validation_serialized_tokens": 2_295_902,
    "validation_supervised_tokens": 939_293,
}
EXPECTED_PREFLIGHT_INVENTORY = {
    "accepted_dataset_sha256": EXPECTED_INPUT["accepted_dataset_sha256"],
    "included_sources": EXPECTED_INPUT["candidate_source_files"],
    "records": EXPECTED_INPUT["records"],
    "train_records": EXPECTED_INPUT["train_records"],
    "train_supervised_tokens": EXPECTED_INPUT["train_supervised_tokens"],
    "validation_records": EXPECTED_INPUT["validation_records"],
    "validation_supervised_tokens": EXPECTED_INPUT["validation_supervised_tokens"],
}
EXPECTED_LIMITS = {
    "candidate_source_files": 20,
    "maximum_candidate_row_reads": 329_200,
    "maximum_evidence_bytes": 4_194_304,
    "maximum_full_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_output_audit_row_reads": 329_200,
    "maximum_output_files": 4,
    "maximum_private_write_bytes": 2_147_483_648,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 180,
    "maximum_runtime_seconds_per_pass": 5_400,
    "maximum_source_line_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_total_row_reads": 658_400,
    "maximum_training_actions": 0,
}
EXPECTED_OUTPUT = {
    "manifest_path": "evidence/lumi-4b-sft-learning-candidate-seal-v0.3.json",
    "private_root": ("artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private"),
    "train_path": (
        "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private/train.jsonl"
    ),
    "validation_path": (
        "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private/validation.jsonl"
    ),
    "verification_path": ("evidence/lumi-4b-sft-learning-candidate-seal-verification-v0.3.json"),
}
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_learning_candidate_seal_v3.py",
    "runner": "scripts/seal_sft_learning_candidate_v3.py",
    "test": "tests/test_learning_candidate_seal_v3.py",
    "verifier": "scripts/verify_sft_learning_candidate_seal_v3.py",
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
    "preflight_implementation": {
        "path": "src/lumi/posttraining_learning_candidate_seal_preflight_v2.py",
        "sha256": "e44a2b9667b9ceb6f5d8f506867cfd7def3dd90499ae1c82f6ba71bedae0ca0a",
    },
    "preflight_plan": {
        "path": "configs/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json",
        "sha256": "bd5005d3b52a141e80f3e71561747c059bf9e1edc3da5fa79389bf6d2cc1ef65",
    },
    "preflight_report": {
        "path": "evidence/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json",
        "sha256": "6db2f31719242e72c17fde0ffb5b9d877309b908a6a22f6c97b28f7e39471418",
    },
    "preflight_verification": {
        "path": ("evidence/lumi-4b-sft-learning-candidate-seal-preflight-verification-v0.2.json"),
        "sha256": "c73387d8ab0bb3db577572fb41510e4a448070b010927a17826a9ea88f7933b1",
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
EXPECTED_SOURCE_PATHS = {
    "allenai_prosocial_dialog_safety_candidate": (
        "artifacts/lumi-sft-prosocial-dialog-safety-v0.1/attempt-001/private/"
        "prosocial-safety-candidate.jsonl"
    ),
    "amazon_massive_bn_bd_tool_canary": (
        "../lumi-artifacts/4b-sft-massive-bn-bd-parser-canary-v0.3/"
        "38be2e88eabf/attempt-001/private/candidate-rows.jsonl"
    ),
    "banglabridge_instructions_closure_canary": (
        "artifacts/lumi-sft-banglabridge-expanded-delta-v0.1/attempt-001/private/"
        "banglabridge-candidate-561.jsonl"
    ),
    "banglatlit_auxiliary_closure_canary": (
        "../lumi-artifacts/4b-sft-closure-canary-acquisition-v0.1/31568114288e/"
        "attempt-001/curation-attempt-002/private/banglatlit-auxiliary-candidate.jsonl"
    ),
    "coherelabs_aya_bengali_closure_canary": (
        "artifacts/lumi-sft-existing-candidate-salvage-v0.1/attempt-001/private/"
        "aya-mit-candidate.jsonl"
    ),
    "coherelabs_aya_dataset": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.1/"
        "58348effb427/attempt-001/aya-clean-candidate.jsonl"
    ),
    "deepmind_mathematics_dataset": (
        "../lumi-artifacts/4b-sft-deepmind-scale-generation-v0.1/427f45075f84/"
        "attempt-001/curated/accepted-records.jsonl"
    ),
    "dmath": (
        "../lumi-artifacts/4b-sft-dmath-acquisition-v0.1/aaba91c90ea8/"
        "attempt-001/curated/accepted-records.jsonl"
    ),
    "fiscalnote_billsum_writing_candidate": (
        "artifacts/lumi-sft-billsum-writing-v0.1/attempt-001/private/"
        "billsum-writing-candidate.jsonl"
    ),
    "lumi_ai_assisted_gap_closure_canary_v0_1": (
        "artifacts/lumi-sft-ai-assisted-gap-closure-canary-v0.1/attempt-001/"
        "private/candidate-rows.jsonl"
    ),
    "lumi_ai_assisted_gap_closure_scaled_v0_2": (
        "artifacts/lumi-sft-ai-assisted-gap-closure-scaled-v0.2/attempt-001/"
        "private/candidate-rows.jsonl"
    ),
    "lumi_ai_assisted_sft_v0_3": (
        "../lumi-artifacts/4b-sft-ai-v3-provenance-v0.1/7961c081dc57/"
        "attempt-001/ai-v3-candidate.jsonl"
    ),
    "lumi_first_party_gpt_oss_teacher_canary_v0_1": (
        "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/05f56935922e/"
        "attempt-001/private/canonical-candidate-v0.16.jsonl"
    ),
    "lumi_first_party_multilingual_safety_tool_v0_6": (
        "artifacts/lumi-gpt-oss-deterministic-tool-repair-v0.6/79db468b8c61/"
        "attempt-001/private/canonical-candidate-v0.17.jsonl"
    ),
    "lumi_first_party_teacher_canary_v0_1": (
        "../lumi-artifacts/4b-sft-first-party-teacher-canary-v0.1/e80b7aa7936b/"
        "attempt-018/private/canonical-candidate-v0.1.jsonl"
    ),
    "lumi_safety_tool_sft_v0_1": (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/8cd8cc0b20cc/"
        "attempt-001/safety-tool-candidate.jsonl"
    ),
    "nvidia_nemotron_post_training_tool_canary": (
        "artifacts/lumi-sft-existing-candidate-salvage-v0.1/attempt-001/private/"
        "nemotron-tool-subset.jsonl"
    ),
    "nvidia_opencodeinstruct": (
        "../lumi-artifacts/4b-sft-opencodeinstruct-single-shard-v0.1/"
        "8f3ba5bafe4d/attempt-002/curated/accepted-records.jsonl"
    ),
    "openassistant_oasst1": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.2/"
        "c07ac2b3c451/attempt-001/oasst1-clean-candidate.jsonl"
    ),
    "openassistant_oasst2": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.2/"
        "c07ac2b3c451/attempt-001/oasst2-clean-candidate.jsonl"
    ),
}

EXPECTED_HYGIENE = dict(preflight_v2.EXPECTED_HYGIENE)
INCLUDED_SOURCE_IDS = tuple(contract_v2.INCLUDED_SOURCE_IDS)
EXCLUDED_SOURCE_IDS = tuple(contract_v2.EXCLUDED_SOURCE_IDS)
EVALUATION_SPLIT_PROHIBITIONS = dict(contract_v2.EVALUATION_SPLIT_PROHIBITIONS)
AI_SOURCE_CONTRACTS = dict(contract_v2.AI_SOURCE_CONTRACTS)
EXPECTED_PRODUCTION_BLOCKERS = [
    "active_production_data_contract_not_updated",
    "dataset_seal_cpu_canary_and_sft_not_authorized",
]
EXPECTED_NEXT_GATE = {
    "action": "authorize_private_nonproduction_sft_encoding_preflight",
    "authorization_required": True,
    "gpu_required": False,
    "production_sft": False,
}
EXPECTED_MANIFEST_MARKER = b'{"started":"learning_candidate_seal_v0.3_pass_1"}\n'
EXPECTED_VERIFICATION_MARKER = b'{"started":"learning_candidate_seal_v0.3_pass_2"}\n'
EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY = {
    "candidate_materialization_performed": True,
    "encoded_shards_created": False,
    "gpu_inference_or_training_used": False,
    "network_or_remote_access_used": False,
    "production_dataset_seal": False,
    "registry_or_contract_changed": False,
}
EXPECTED_MANIFEST_BINDINGS = {
    "candidate_mix_report_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"]["sha256"],
    "candidate_mix_verification_sha256": EXPECTED_EXTERNAL_BINDINGS["candidate_mix_verification"][
        "sha256"
    ],
    "chat_template_sha256": EXPECTED_EXTERNAL_BINDINGS["chat_template"]["sha256"],
    "data_contract_sha256": EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"],
    "final_base_binding_sha256": EXPECTED_EXTERNAL_BINDINGS["final_base_binding"]["sha256"],
    "preflight_report_sha256": EXPECTED_EXTERNAL_BINDINGS["preflight_report"]["sha256"],
    "preflight_verification_sha256": EXPECTED_EXTERNAL_BINDINGS["preflight_verification"]["sha256"],
    "source_registry_sha256": EXPECTED_EXTERNAL_BINDINGS["source_registry"]["sha256"],
    "tokenizer_sha256": EXPECTED_EXTERNAL_BINDINGS["tokenizer"]["sha256"],
}
EXPECTED_CONTRACT_BOUNDARY = {
    "evaluation_split_prohibitions": EVALUATION_SPLIT_PROHIBITIONS,
    "external_distribution_allowed": False,
    "machine_gap_source_contracts": AI_SOURCE_CONTRACTS,
    "private_research_nonproduction": True,
    "production_sft_allowed": False,
}
EXPECTED_MANIFEST_NEXT_GATE = {
    "action": "complete_independent_output_audit",
    "authorization_required": False,
    "candidate_files_reopened": False,
    "output_rows_to_audit": EXPECTED_INPUT["records"],
}
EXPECTED_PASS_ONE_RESOURCES = {
    "candidate_row_reads": EXPECTED_INPUT["records"],
    "data_output_files": 2,
    "full_passes": 1,
    "gpu_use": 0,
    "inference_calls": 0,
    "network_bytes": 0,
    "output_audit_row_reads": 0,
    "source_line_reads": 0,
    "test_object_reads": 0,
    "training_actions": 0,
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
        "sources",
        "status",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], item) for key, item in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
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


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key in learning-candidate input")
        value[key] = item
    return value


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
        raise ValueError("learning-candidate JSON is not UTF-8") from exc


def _load_object(path: str | Path) -> dict[str, Any]:
    value = strict_json_bytes(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("learning-candidate binding must contain a JSON object")
    return value


def _assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError(f"raw candidate key forbidden at {path}.{key}")
            _assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, f"{path}[{index}]")


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
            raise ValueError("learning-candidate pass marker inode changed")
        if os.write(descriptor, marker) != len(marker):
            raise OSError("short learning-candidate marker write")
        os.fsync(descriptor)
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return identity


def finish_marker(
    path: Path,
    marker: bytes,
    payload: bytes,
    identity: tuple[int, int],
) -> None:
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != identity
            or os.read(descriptor, len(marker) + 1) != marker
        ):
            raise ValueError("learning-candidate pass marker changed")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short learning-candidate final evidence write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    if name in EXPECTED_EXTERNAL_BINDINGS:
        expected = EXPECTED_EXTERNAL_BINDINGS[name]
        if not _strict_json_equal(binding, expected):
            raise ValueError(f"learning-candidate external binding changed: {name}")
        expected_path = expected["path"]
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
            raise ValueError(f"learning-candidate code binding changed: {name}")
    declared = root / str(binding["path"])
    if declared != root / expected_path:
        raise ValueError(f"learning-candidate binding path changed: {name}")
    try:
        info = declared.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing learning-candidate binding: {name}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"learning-candidate binding must be regular: {name}")
    if file_sha256(declared) != binding["sha256"]:
        raise ValueError(f"learning-candidate binding hash changed: {name}")
    return declared.resolve()


def _source_ledger_from_mix(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "by_split": {
            split: {
                "records": metrics["by_split"][split]["records"],
                "serialized_tokens": metrics["by_split"][split]["serialized_tokens"],
                "supervised_tokens": metrics["by_split"][split]["supervised_tokens"],
            }
            for split in ("train", "validation")
        },
        "candidate_sha256": metrics["candidate_file_sha256"],
        "records": metrics["records"],
    }


def _validate_sources(
    sources: Any,
    *,
    check_candidate_files: bool,
    mix_report: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(sources, list) or len(sources) != EXPECTED_INPUT["candidate_source_files"]:
        raise ValueError("learning-candidate source table changed")
    source_metrics = mix_report.get("source_metrics")
    if not isinstance(source_metrics, Mapping) or set(source_metrics) != set(EXPECTED_SOURCE_PATHS):
        raise ValueError("learning-candidate mix source metrics changed")
    normalized: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "by_split",
            "bytes",
            "candidate_sha256",
            "path",
            "records",
            "source_id",
        }:
            raise ValueError("learning-candidate source specification changed")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or source_id not in EXPECTED_SOURCE_PATHS:
            raise ValueError("learning-candidate source ID changed")
        expected = {
            **_source_ledger_from_mix(source_metrics[source_id]),
            "path": EXPECTED_SOURCE_PATHS[source_id],
            "source_id": source_id,
        }
        if (
            set(source) != set(expected) | {"bytes"}
            or not _strict_json_equal(
                {key: value for key, value in source.items() if key != "bytes"},
                expected,
            )
            or type(source.get("bytes")) is not int
            or source["bytes"] <= 0
        ):
            raise ValueError(f"learning-candidate source ledger changed: {source_id}")
        declared = root / str(source["path"])
        if declared != root / EXPECTED_SOURCE_PATHS[source_id]:
            raise ValueError(f"learning-candidate source path changed: {source_id}")
        if check_candidate_files:
            try:
                info = declared.lstat()
            except FileNotFoundError as exc:
                raise ValueError(f"missing learning-candidate source: {source_id}") from exc
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_size != source["bytes"]
            ):
                raise ValueError(f"learning-candidate source metadata changed: {source_id}")
        normalized.append(dict(source))
    if tuple(item["source_id"] for item in normalized) != INCLUDED_SOURCE_IDS:
        raise ValueError("learning-candidate source order or partition changed")
    if sum(item["bytes"] for item in normalized) != EXPECTED_INPUT["candidate_input_bytes"]:
        raise ValueError("learning-candidate input byte inventory changed")
    for split in ("train", "validation"):
        if (
            sum(item["by_split"][split]["records"] for item in normalized)
            != EXPECTED_INPUT[f"{split}_records"]
            or sum(item["by_split"][split]["serialized_tokens"] for item in normalized)
            != EXPECTED_INPUT[f"{split}_serialized_tokens"]
            or sum(item["by_split"][split]["supervised_tokens"] for item in normalized)
            != EXPECTED_INPUT[f"{split}_supervised_tokens"]
        ):
            raise ValueError(f"learning-candidate aggregate ledger changed: {split}")
    return tuple(normalized)


def _validate_metadata_chain(
    plan: Mapping[str, Any], root: Path, *, check_candidate_files: bool
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    resolved = {
        name: _resolve_binding(root, binding, name) for name, binding in plan["bindings"].items()
    }
    mix_report = _load_object(resolved["candidate_mix_report"])
    mix_verification = _load_object(resolved["candidate_mix_verification"])
    mix_v18.validate_report(mix_report)
    mix_v18.validate_verification(mix_verification)
    if (
        mix_verification.get("report_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["candidate_mix_report"]["sha256"]
        or mix_verification.get("report_internal_sha256") != mix_report.get("report_sha256")
        or mix_report.get("generic_sft_validation", {}).get("accepted_dataset_sha256")
        != EXPECTED_INPUT["accepted_dataset_sha256"]
    ):
        raise ValueError("learning-candidate mix evidence chain changed")

    preflight_report = _load_object(resolved["preflight_report"])
    preflight_verification = _load_object(resolved["preflight_verification"])
    preflight_v2.validate_report(preflight_report)
    preflight_v2.validate_verification(preflight_verification)
    if (
        preflight_verification.get("report_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["preflight_report"]["sha256"]
        or preflight_verification.get("report_internal_sha256")
        != preflight_report.get("report_sha256")
        or not _strict_json_equal(preflight_report.get("inventory"), EXPECTED_PREFLIGHT_INVENTORY)
    ):
        raise ValueError("learning-candidate preflight chain changed")

    predecessor_contract = _load_object(resolved["predecessor_data_contract"])
    data_contract = _load_object(resolved["data_contract"])
    contract_v2.validate_contract(data_contract, predecessor=predecessor_contract)
    contract_report = _load_object(resolved["data_contract_activation_report"])
    contract_verification = _load_object(resolved["data_contract_activation_verification"])
    contract_v2.validate_report(contract_report)
    contract_v2.validate_verification(contract_verification)
    if (
        contract_report.get("data_contract", {}).get("file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"]
        or contract_verification.get("data_contract_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["data_contract"]["sha256"]
        or contract_verification.get("report_file_sha256")
        != EXPECTED_EXTERNAL_BINDINGS["data_contract_activation_report"]["sha256"]
        or contract_verification.get("report_internal_sha256")
        != contract_report.get("report_sha256")
    ):
        raise ValueError("learning-candidate contract activation chain changed")

    policy = SFTPolicy.from_mapping(data_contract)
    learning = policy.learning_candidate
    if learning is None:
        raise ValueError("learning-candidate contract is absent")
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
        or any(source.production_eligible for source in registry.sources)
        or set(registry.by_id) != set(INCLUDED_SOURCE_IDS) | set(EXCLUDED_SOURCE_IDS)
        or learning["included_source_ids"] != frozenset(INCLUDED_SOURCE_IDS)
        or learning["excluded_source_ids"] != frozenset(EXCLUDED_SOURCE_IDS)
        or learning["evaluation_split_prohibitions"]
        != {
            source_id: frozenset(splits)
            for source_id, splits in EVALUATION_SPLIT_PROHIBITIONS.items()
        }
        or not _strict_json_equal(
            data_contract["learning_candidate"]["ai_assisted_source_contracts"],
            AI_SOURCE_CONTRACTS,
        )
    ):
        raise ValueError("learning-candidate registry or contract changed")

    validate_final_base_binding(_load_object(resolved["final_base_binding"]))
    template = load_chat_template(resolved["chat_template"])
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, template)

    sources = _validate_sources(
        plan["sources"],
        check_candidate_files=check_candidate_files,
        mix_report=mix_report,
        root=root,
    )
    return mix_report, sources


def load_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
    check_candidate_files: bool = True,
) -> dict[str, Any]:
    if type(check_candidate_files) is not bool:
        raise TypeError("check_candidate_files must be bool")
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    if plan_path != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("learning-candidate plan path changed")
    info = plan_path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("learning-candidate plan must be a regular file")
    plan = _load_object(plan_path)
    if (
        set(plan) != _PLAN_KEYS
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("observed_at") != EXPECTED_OBSERVED_AT
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or type(plan.get("authorization_bytes")) is not int
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
        or not _strict_json_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION)
        or not _strict_json_equal(plan.get("input"), EXPECTED_INPUT)
        or not _strict_json_equal(plan.get("limits"), EXPECTED_LIMITS)
        or not _strict_json_equal(plan.get("output"), EXPECTED_OUTPUT)
    ):
        raise ValueError("learning-candidate plan boundary changed")
    bindings = plan.get("bindings")
    expected_names = set(EXPECTED_EXTERNAL_BINDINGS) | set(EXPECTED_CODE_PATHS)
    if not isinstance(bindings, Mapping) or set(bindings) != expected_names:
        raise ValueError("learning-candidate binding set changed")
    _validate_metadata_chain(plan, root, check_candidate_files=check_candidate_files)
    return plan


def output_paths(plan: Mapping[str, Any], root: Path) -> dict[str, Path]:
    paths = {
        key: (root / str(plan["output"][key])).resolve()
        for key in (
            "manifest_path",
            "private_root",
            "train_path",
            "validation_path",
            "verification_path",
        )
    }
    expected = {key: (root / value).resolve() for key, value in EXPECTED_OUTPUT.items()}
    if paths != expected:
        raise ValueError("learning-candidate output paths changed")
    if not paths["train_path"].is_relative_to(paths["private_root"]) or not paths[
        "validation_path"
    ].is_relative_to(paths["private_root"]):
        raise ValueError("learning-candidate private output escaped root")
    evidence_root = (root / "evidence").resolve()
    if not paths["manifest_path"].is_relative_to(evidence_root) or not paths[
        "verification_path"
    ].is_relative_to(evidence_root):
        raise ValueError("learning-candidate evidence output escaped root")
    return paths


def preflight_runner(plan: Mapping[str, Any], root: Path) -> dict[str, Path]:
    paths = output_paths(plan, root)
    paths["private_root"].mkdir(parents=True, mode=0o700, exist_ok=True)
    root_info = paths["private_root"].lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) & 0o077
    ):
        raise ValueError("learning-candidate private root is not private")
    for key in ("manifest_path", "train_path", "validation_path", "verification_path"):
        if paths[key].exists():
            raise FileExistsError("learning-candidate materialization already started")
    return paths


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> dict[str, Path]:
    paths = output_paths(plan, root)
    if paths["verification_path"].exists():
        raise FileExistsError("learning-candidate verification already started")
    for key in ("manifest_path", "train_path", "validation_path"):
        if not paths[key].is_file():
            raise ValueError("learning-candidate pass-one output is missing")
    if paths["manifest_path"].stat().st_size <= len(EXPECTED_MANIFEST_MARKER):
        raise ValueError("learning-candidate manifest is incomplete")
    return paths


def open_private_output(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("learning-candidate output inode changed")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("learning-candidate output is not private")
        _fsync_directory(path.parent)
        return os.fdopen(descriptor, "wb", buffering=1024 * 1024)
    except BaseException:
        os.close(descriptor)
        raise


def _open_candidate_once(root: Path, source: Mapping[str, Any]) -> BinaryIO:
    path = root / str(source["path"])
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != source["bytes"] or info.st_nlink < 1:
            raise ValueError(f"learning-candidate source inode changed: {source['source_id']}")
        return os.fdopen(descriptor, "rb", buffering=1024 * 1024)
    except BaseException:
        os.close(descriptor)
        raise


def _write_checked(handle: BinaryIO, payload: bytes) -> None:
    written = handle.write(payload)
    if written != len(payload):
        raise OSError("short learning-candidate data write")


def _runtime_check(started: float) -> None:
    if time.monotonic() - started > EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"]:
        raise TimeoutError("learning-candidate pass runtime cap reached")


def materialize_sources(
    *,
    sources: Sequence[Mapping[str, Any]],
    repository_root: Path,
    train_handle: BinaryIO,
    validation_handle: BinaryIO,
    started: float,
    maximum_candidate_rows: int,
    maximum_data_write_bytes: int,
) -> dict[str, Any]:
    if tuple(source["source_id"] for source in sources) != tuple(
        sorted(source["source_id"] for source in sources)
    ):
        raise ValueError("learning-candidate sources are not in lexical order")
    handles = {"train": train_handle, "validation": validation_handle}
    split_counts: Counter[str] = Counter()
    split_serialized_tokens: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_bytes: Counter[str] = Counter()
    split_hashers = {split: hashlib.sha256() for split in handles}
    split_record_id_hashers = {split: hashlib.sha256() for split in handles}
    input_record_id_hasher = hashlib.sha256()
    seen_record_ids: set[str] = set()
    observed_sources: dict[str, Any] = {}
    total_candidate_rows = 0
    total_data_bytes = 0
    total_input_bytes = 0

    for source in sources:
        _runtime_check(started)
        source_id = str(source["source_id"])
        source_split_counts: Counter[str] = Counter()
        candidate_hasher = hashlib.sha256()
        candidate_bytes = 0
        source_records = 0
        with _open_candidate_once(repository_root, source) as candidate_handle:
            while source_records < source["records"]:
                line = candidate_handle.readline()
                if not line:
                    break
                candidate_hasher.update(line)
                candidate_bytes += len(line)
                if not line.strip():
                    continue
                source_records += 1
                total_candidate_rows += 1
                if total_candidate_rows > maximum_candidate_rows:
                    raise ValueError("learning-candidate row-read cap exceeded")
                row = strict_json_bytes(line)
                if not isinstance(row, dict):
                    raise ValueError("learning-candidate row must be an object")
                row_source = row.get("source")
                split = row.get("split")
                record_id = row.get("record_id")
                if (
                    not isinstance(row_source, Mapping)
                    or row_source.get("source_id") != source_id
                    or split not in handles
                    or not isinstance(record_id, str)
                    or not record_id
                ):
                    raise ValueError(f"learning-candidate row binding changed: {source_id}")
                if record_id in seen_record_ids:
                    raise ValueError("duplicate learning-candidate record ID")
                seen_record_ids.add(record_id)
                identity_bytes = f"{source_id}\0{record_id}\n".encode()
                input_record_id_hasher.update(identity_bytes)
                split_record_id_hashers[str(split)].update(identity_bytes)
                payload = canonical_json(row) + b"\n"
                if total_data_bytes + len(payload) > maximum_data_write_bytes:
                    raise ValueError("learning-candidate private-write cap reached")
                _write_checked(handles[str(split)], payload)
                split_hashers[str(split)].update(payload)
                split_counts[str(split)] += 1
                split_bytes[str(split)] += len(payload)
                source_split_counts[str(split)] += 1
                total_data_bytes += len(payload)
                if total_candidate_rows % 10_000 == 0:
                    _runtime_check(started)
            if source_records == source["records"] and candidate_bytes != source["bytes"]:
                raise ValueError(
                    f"learning-candidate source has bytes after final bound row: {source_id}"
                )
        if (
            source_records != source["records"]
            or candidate_bytes != source["bytes"]
            or candidate_hasher.hexdigest() != source["candidate_sha256"]
        ):
            raise ValueError(f"learning-candidate source changed: {source_id}")
        for split in handles:
            expected = source["by_split"][split]
            if source_split_counts[split] != expected["records"]:
                raise ValueError(f"learning-candidate split ledger changed: {source_id}:{split}")
            split_serialized_tokens[split] += expected["serialized_tokens"]
            split_tokens[split] += expected["supervised_tokens"]
        total_input_bytes += candidate_bytes
        observed_sources[source_id] = {
            "by_split": {
                split: {
                    "records": source_split_counts[split],
                    "serialized_tokens": source["by_split"][split]["serialized_tokens"],
                    "supervised_tokens": source["by_split"][split]["supervised_tokens"],
                }
                for split in handles
            },
            "candidate_bytes": candidate_bytes,
            "candidate_sha256": candidate_hasher.hexdigest(),
            "records": source_records,
        }

    for handle in handles.values():
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "candidate_input_bytes": total_input_bytes,
        "candidate_row_reads": total_candidate_rows,
        "input_record_id_sequence_sha256": input_record_id_hasher.hexdigest(),
        "source_metrics": observed_sources,
        "splits": {
            split: {
                "bytes": split_bytes[split],
                "record_id_sequence_sha256": split_record_id_hashers[split].hexdigest(),
                "records": split_counts[split],
                "serialized_tokens": split_serialized_tokens[split],
                "sha256": split_hashers[split].hexdigest(),
                "supervised_tokens": split_tokens[split],
            }
            for split in handles
        },
        "unique_record_ids": len(seen_record_ids),
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
        "language_floor_gap_total": sum(closure["language_floor_gaps_behavior_safe"].values()),
        "near_duplicate_cross_source_components": near["cross_source_components"],
        "near_duplicate_edges": near["edge_count"],
        "near_duplicate_multi_record_components": near["multi_record_components"],
        "near_duplicate_train_validation_components": near["train_validation_components"],
        "quarantined_records": report["quarantine"]["total_records"],
        "source_family_cap_violations": len(closure["source_family_cap_violations"]),
        "strict_targeted_gap_lower_bound": pool["strict_targeted_gap_lower_bound"],
        "task_floor_gap_total": sum(closure["task_floor_gaps_behavior_safe"].values()),
        "train_validation_exact_overlap_groups": exact["train_validation_overlap_groups"],
    }


def build_manifest(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    train_handle: BinaryIO,
    validation_handle: BinaryIO,
    started: float,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    mix_report, sources = _validate_metadata_chain(plan, root, check_candidate_files=True)
    maximum_data_bytes = (
        EXPECTED_LIMITS["maximum_private_write_bytes"] - EXPECTED_LIMITS["maximum_evidence_bytes"]
    )
    materialized = materialize_sources(
        sources=sources,
        repository_root=root,
        train_handle=train_handle,
        validation_handle=validation_handle,
        started=started,
        maximum_candidate_rows=EXPECTED_LIMITS["maximum_candidate_row_reads"],
        maximum_data_write_bytes=maximum_data_bytes,
    )
    splits = materialized["splits"]
    if (
        materialized["candidate_input_bytes"] != EXPECTED_INPUT["candidate_input_bytes"]
        or materialized["candidate_row_reads"] != EXPECTED_INPUT["records"]
        or materialized["unique_record_ids"] != EXPECTED_INPUT["records"]
        or splits["train"]["records"] != EXPECTED_INPUT["train_records"]
        or splits["train"]["serialized_tokens"] != EXPECTED_INPUT["train_serialized_tokens"]
        or splits["train"]["supervised_tokens"] != EXPECTED_INPUT["train_supervised_tokens"]
        or splits["validation"]["records"] != EXPECTED_INPUT["validation_records"]
        or splits["validation"]["serialized_tokens"]
        != EXPECTED_INPUT["validation_serialized_tokens"]
        or splits["validation"]["supervised_tokens"]
        != EXPECTED_INPUT["validation_supervised_tokens"]
        or not _strict_json_equal(_hygiene_from_mix(mix_report), EXPECTED_HYGIENE)
        or mix_report.get("closure", {}).get("blockers") != EXPECTED_PRODUCTION_BLOCKERS
    ):
        raise ValueError("learning-candidate materialized totals changed")
    manifest: dict[str, Any] = {
        "authorization_boundary": EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY,
        "bindings": EXPECTED_MANIFEST_BINDINGS,
        "contract_boundary": EXPECTED_CONTRACT_BOUNDARY,
        "hygiene": EXPECTED_HYGIENE,
        "input": EXPECTED_INPUT,
        "next_gate": EXPECTED_MANIFEST_NEXT_GATE,
        "observed_at": EXPECTED_OBSERVED_AT,
        "ordering_contract": "source_id_lexical_then_existing_file_order_v0.1",
        "output": materialized,
        "plan_file_sha256": plan_file_sha256,
        "production_blockers_preserved": EXPECTED_PRODUCTION_BLOCKERS,
        "production_eligible": False,
        "raw_training_text_retained_in_manifest": False,
        "resource_accounting": EXPECTED_PASS_ONE_RESOURCES,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": MANIFEST_STATUS,
        "token_accounting_contract": (
            "exact_candidate_sha256_plus_frozen_v0_18_per_source_token_ledger"
        ),
    }
    manifest["seal_sha256"] = sha256_value(manifest)
    validate_manifest(manifest, plan=plan)
    return manifest


def _expected_manifest_sources(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        source["source_id"]: {
            "by_split": source["by_split"],
            "candidate_bytes": source["bytes"],
            "candidate_sha256": source["candidate_sha256"],
            "records": source["records"],
        }
        for source in plan["sources"]
    }


def validate_manifest(manifest: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    unsigned = {key: item for key, item in manifest.items() if key != "seal_sha256"}
    expected_keys = {
        "authorization_boundary",
        "bindings",
        "contract_boundary",
        "hygiene",
        "input",
        "next_gate",
        "observed_at",
        "ordering_contract",
        "output",
        "plan_file_sha256",
        "production_blockers_preserved",
        "production_eligible",
        "raw_training_text_retained_in_manifest",
        "resource_accounting",
        "schema_version",
        "seal_sha256",
        "status",
        "token_accounting_contract",
    }
    output = manifest.get("output")
    splits = output.get("splits") if isinstance(output, Mapping) else None
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != MANIFEST_STATUS
        or manifest.get("observed_at") != EXPECTED_OBSERVED_AT
        or manifest.get("seal_sha256") != sha256_value(unsigned)
        or not isinstance(manifest.get("plan_file_sha256"), str)
        or _SHA256.fullmatch(manifest["plan_file_sha256"]) is None
        or manifest.get("production_eligible") is not False
        or manifest.get("raw_training_text_retained_in_manifest") is not False
        or manifest.get("ordering_contract") != "source_id_lexical_then_existing_file_order_v0.1"
        or manifest.get("token_accounting_contract")
        != "exact_candidate_sha256_plus_frozen_v0_18_per_source_token_ledger"
        or not _strict_json_equal(
            manifest.get("authorization_boundary"),
            EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY,
        )
        or not _strict_json_equal(manifest.get("bindings"), EXPECTED_MANIFEST_BINDINGS)
        or not _strict_json_equal(manifest.get("contract_boundary"), EXPECTED_CONTRACT_BOUNDARY)
        or not _strict_json_equal(manifest.get("input"), EXPECTED_INPUT)
        or not _strict_json_equal(manifest.get("hygiene"), EXPECTED_HYGIENE)
        or not _strict_json_equal(manifest.get("next_gate"), EXPECTED_MANIFEST_NEXT_GATE)
        or not _strict_json_equal(manifest.get("resource_accounting"), EXPECTED_PASS_ONE_RESOURCES)
        or manifest.get("production_blockers_preserved") != EXPECTED_PRODUCTION_BLOCKERS
        or not isinstance(output, Mapping)
        or set(output)
        != {
            "candidate_input_bytes",
            "candidate_row_reads",
            "input_record_id_sequence_sha256",
            "source_metrics",
            "splits",
            "unique_record_ids",
        }
        or output.get("candidate_input_bytes") != EXPECTED_INPUT["candidate_input_bytes"]
        or output.get("candidate_row_reads") != EXPECTED_INPUT["records"]
        or output.get("unique_record_ids") != EXPECTED_INPUT["records"]
        or not isinstance(output.get("input_record_id_sequence_sha256"), str)
        or _SHA256.fullmatch(output["input_record_id_sequence_sha256"]) is None
        or not _strict_json_equal(output.get("source_metrics"), _expected_manifest_sources(plan))
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "validation"}
    ):
        raise ValueError("learning-candidate manifest boundary changed")
    for split, expected_records, expected_serialized, expected_supervised in (
        (
            "train",
            EXPECTED_INPUT["train_records"],
            EXPECTED_INPUT["train_serialized_tokens"],
            EXPECTED_INPUT["train_supervised_tokens"],
        ),
        (
            "validation",
            EXPECTED_INPUT["validation_records"],
            EXPECTED_INPUT["validation_serialized_tokens"],
            EXPECTED_INPUT["validation_supervised_tokens"],
        ),
    ):
        value = splits[split]
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "bytes",
                "record_id_sequence_sha256",
                "records",
                "serialized_tokens",
                "sha256",
                "supervised_tokens",
            }
            or type(value.get("bytes")) is not int
            or value["bytes"] <= 0
            or value.get("records") != expected_records
            or value.get("serialized_tokens") != expected_serialized
            or value.get("supervised_tokens") != expected_supervised
            or not isinstance(value.get("sha256"), str)
            or _SHA256.fullmatch(value["sha256"]) is None
            or not isinstance(value.get("record_id_sequence_sha256"), str)
            or _SHA256.fullmatch(value["record_id_sequence_sha256"]) is None
        ):
            raise ValueError(f"learning-candidate manifest split changed: {split}")
    _assert_raw_text_free(manifest)


def _open_output_for_audit(path: Path, expected_bytes: int) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_size != expected_bytes
        ):
            raise ValueError("learning-candidate output metadata changed")
        return os.fdopen(descriptor, "rb", buffering=1024 * 1024)
    except BaseException:
        os.close(descriptor)
        raise


def audit_outputs(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repository_root: Path,
    started: float,
) -> dict[str, Any]:
    validate_manifest(manifest, plan=plan)
    paths = output_paths(plan, repository_root)
    expected_sources = tuple(source["source_id"] for source in plan["sources"])
    source_index = {source_id: index for index, source_id in enumerate(expected_sources)}
    expected_by_split = {
        split: {
            source["source_id"]: source["by_split"][split]["records"] for source in plan["sources"]
        }
        for split in ("train", "validation")
    }
    observed_outputs: dict[str, Any] = {}
    observed_source_split_records = {
        source_id: {"train": 0, "validation": 0} for source_id in expected_sources
    }
    seen_record_ids: set[str] = set()
    total_rows = 0
    for split in ("train", "validation"):
        expected_output = manifest["output"]["splits"][split]
        hasher = hashlib.sha256()
        identity_hasher = hashlib.sha256()
        byte_count = 0
        last_source_index = -1
        with _open_output_for_audit(paths[f"{split}_path"], expected_output["bytes"]) as handle:
            for _ in range(expected_output["records"]):
                line = handle.readline()
                if not line or not line.strip():
                    raise ValueError("learning-candidate output row is missing")
                total_rows += 1
                if total_rows > EXPECTED_LIMITS["maximum_output_audit_row_reads"]:
                    raise ValueError("learning-candidate output-audit cap exceeded")
                row = strict_json_bytes(line)
                if not isinstance(row, dict) or canonical_json(row) + b"\n" != line:
                    raise ValueError("learning-candidate output is not canonical JSONL")
                source = row.get("source")
                source_id = source.get("source_id") if isinstance(source, Mapping) else None
                record_id = row.get("record_id")
                if (
                    source_id not in source_index
                    or row.get("split") != split
                    or not isinstance(record_id, str)
                    or not record_id
                ):
                    raise ValueError("learning-candidate output row binding changed")
                current_source_index = source_index[str(source_id)]
                if current_source_index < last_source_index:
                    raise ValueError("learning-candidate output source order changed")
                last_source_index = current_source_index
                if record_id in seen_record_ids:
                    raise ValueError("duplicate learning-candidate output record ID")
                seen_record_ids.add(record_id)
                identity_bytes = f"{source_id}\0{record_id}\n".encode()
                identity_hasher.update(identity_bytes)
                observed_source_split_records[str(source_id)][split] += 1
                hasher.update(line)
                byte_count += len(line)
                if total_rows % 10_000 == 0:
                    _runtime_check(started)
            if handle.read(1):
                raise ValueError("learning-candidate output has extra bytes")
        observed_outputs[split] = {
            "bytes": byte_count,
            "record_id_sequence_sha256": identity_hasher.hexdigest(),
            "records": expected_output["records"],
            "sha256": hasher.hexdigest(),
        }
        for source_id, expected_records in expected_by_split[split].items():
            if observed_source_split_records[source_id][split] != expected_records:
                raise ValueError(f"learning-candidate output ledger changed: {source_id}:{split}")
        if any(
            observed_outputs[split][field] != expected_output[field]
            for field in ("bytes", "record_id_sequence_sha256", "records", "sha256")
        ):
            raise ValueError(f"learning-candidate output hash changed: {split}")
    if total_rows != EXPECTED_INPUT["records"] or len(seen_record_ids) != total_rows:
        raise ValueError("learning-candidate output row total changed")
    _runtime_check(started)
    return {
        "candidate_files_opened": 0,
        "output_audit_row_reads": total_rows,
        "source_split_records": observed_source_split_records,
        "splits": observed_outputs,
        "unique_record_ids": len(seen_record_ids),
    }


EXPECTED_VERIFICATION_CHECKS = {
    "all_329200_outputs_parsed_once": True,
    "candidate_files_not_reopened_in_pass_two": True,
    "canonical_jsonl_and_source_split_order_verified": True,
    "exact_20_source_ledgers_reproduced": True,
    "file_bytes_hashes_counts_and_record_sequences_reproduced": True,
    "frozen_v0_18_supervised_token_ledger_bound": True,
    "no_network_gpu_inference_training_or_encoded_shards": True,
    "nonproduction_and_production_blockers_preserved": True,
    "raw_training_text_absent_from_evidence": True,
    "registry_contract_and_evaluation_exclusions_preserved": True,
    "unique_record_ids_verified": True,
}
EXPECTED_PASS_TWO_RESOURCES = {
    "candidate_row_reads": EXPECTED_INPUT["records"],
    "data_output_files": 2,
    "evidence_output_files": 2,
    "full_passes": 2,
    "gpu_use": 0,
    "inference_calls": 0,
    "network_bytes": 0,
    "output_audit_row_reads": EXPECTED_INPUT["records"],
    "source_line_reads": 0,
    "test_object_reads": 0,
    "total_row_reads": EXPECTED_LIMITS["maximum_total_row_reads"],
    "training_actions": 0,
}


def _validate_audit(
    audit: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected_source_counts = {
        source["source_id"]: {
            split: source["by_split"][split]["records"] for split in ("train", "validation")
        }
        for source in plan["sources"]
    }
    expected_splits = {
        split: {
            field: manifest["output"]["splits"][split][field]
            for field in ("bytes", "record_id_sequence_sha256", "records", "sha256")
        }
        for split in ("train", "validation")
    }
    if (
        not isinstance(audit, Mapping)
        or set(audit)
        != {
            "candidate_files_opened",
            "output_audit_row_reads",
            "source_split_records",
            "splits",
            "unique_record_ids",
        }
        or not _strict_json_equal(audit.get("candidate_files_opened"), 0)
        or not _strict_json_equal(audit.get("output_audit_row_reads"), EXPECTED_INPUT["records"])
        or not _strict_json_equal(audit.get("unique_record_ids"), EXPECTED_INPUT["records"])
        or not _strict_json_equal(audit.get("source_split_records"), expected_source_counts)
        or not _strict_json_equal(audit.get("splits"), expected_splits)
    ):
        raise ValueError("learning-candidate output audit changed")


def build_verification(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    validate_manifest(manifest, plan=plan)
    _validate_audit(audit, plan=plan, manifest=manifest)
    verification: dict[str, Any] = {
        "checks": EXPECTED_VERIFICATION_CHECKS,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["seal_sha256"],
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_outputs": audit,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "resource_accounting": EXPECTED_PASS_TWO_RESOURCES,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification, plan=plan, manifest=manifest)
    return verification


def validate_verification(
    verification: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    validate_manifest(manifest, plan=plan)
    unsigned = {key: item for key, item in verification.items() if key != "verification_sha256"}
    expected_keys = {
        "checks",
        "manifest_file_sha256",
        "manifest_internal_sha256",
        "next_gate",
        "observed_outputs",
        "production_eligible",
        "raw_training_text_retained",
        "resource_accounting",
        "schema_version",
        "status",
        "verification_sha256",
    }
    if (
        set(verification) != expected_keys
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("manifest_internal_sha256") != manifest.get("seal_sha256")
        or not isinstance(verification.get("manifest_file_sha256"), str)
        or _SHA256.fullmatch(verification["manifest_file_sha256"]) is None
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or not _strict_json_equal(verification.get("checks"), EXPECTED_VERIFICATION_CHECKS)
        or not _strict_json_equal(verification.get("next_gate"), EXPECTED_NEXT_GATE)
        or not _strict_json_equal(
            verification.get("resource_accounting"),
            EXPECTED_PASS_TWO_RESOURCES,
        )
    ):
        raise ValueError("learning-candidate verification boundary changed")
    observed_outputs = verification.get("observed_outputs")
    if not isinstance(observed_outputs, Mapping):
        raise ValueError("learning-candidate verification audit is absent")
    _validate_audit(observed_outputs, plan=plan, manifest=manifest)
    _assert_raw_text_free(verification)


def prospective_total_write_bytes(
    *,
    manifest_path: Path,
    train_path: Path,
    validation_path: Path,
    verification_payload: bytes,
) -> int:
    return (
        len(EXPECTED_MANIFEST_MARKER)
        + manifest_path.stat().st_size
        + train_path.stat().st_size
        + validation_path.stat().st_size
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_payload)
    )


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_INPUT",
    "EXPECTED_LIMITS",
    "EXPECTED_MANIFEST_MARKER",
    "EXPECTED_OUTPUT",
    "EXPECTED_SOURCE_PATHS",
    "EXPECTED_VERIFICATION_MARKER",
    "MANIFEST_STATUS",
    "VERIFICATION_STATUS",
    "_assert_raw_text_free",
    "_load_object",
    "audit_outputs",
    "build_manifest",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_file_bytes",
    "load_plan",
    "materialize_sources",
    "open_private_output",
    "output_paths",
    "preflight_runner",
    "preflight_verifier",
    "prospective_total_write_bytes",
    "sha256_bytes",
    "sha256_value",
    "start_marker",
    "strict_json_bytes",
    "validate_manifest",
    "validate_verification",
]
