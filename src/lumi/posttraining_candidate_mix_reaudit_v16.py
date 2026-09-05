"""Registry-v0.22-bound full local candidate-mix re-audit.

The audit replays the exact clean registry-v0.21 candidate inventory plus the
64-row owner-reviewed MASSIVE bn-BD parser candidate.  It runs the frozen
generic, benchmark, exact-duplicate, and MinHash gates without mutating any
candidate, source, registry, data contract, seal, or training artifact.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v13 as v13
from . import posttraining_candidate_mix_reaudit_v15 as v15
from . import posttraining_massive_bn_bd_parser_source_promotion_v6 as promotion_v6
from .posttraining_candidate_mix_preflight import (
    EXPECTED_LANGUAGE_MODES,
    EXPECTED_TASKS,
    _canonical_message_digest,
    _duplicate_snapshot,
    _load_json,
    _load_jsonl,
    _targets,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import (
    encode_sft_messages,
    load_chat_template,
    verify_chat_tokenizer,
)
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.16"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.16"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.16"
PLAN_STATUS = "authorized_registry_v0_22_full_local_reaudit_no_production_action"
REPORT_STATUS = "PASS_registry_v0_22_full_candidate_mix_reaudit_nonproduction"
VERIFICATION_STATUS = (
    "PASS_verified_registry_v0_22_full_candidate_mix_reaudit_nonproduction"
)

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-candidate-mix-reaudit-v0.16.json")
EXPECTED_AUTHORIZATION_TEXT = "ok next step shuru kor"
EXPECTED_AUTHORIZATION_TEXT_SHA256 = (
    "1046cda26a21ff4994e4a452870466e609fe93d825cad3622e9fd069fb18b35e"
)
EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": False,
    "candidate_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "full_local_reaudit_authorized": True,
    "gpu_authorized": False,
    "inference_authorized": False,
    "instance_action_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "s3_action_authorized": False,
    "sft_training_authorized": False,
    "source_promotion_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_per_pass": 314_780,
    "maximum_full_audit_passes": 2,
    "maximum_total_candidate_row_reads": 629_560,
    "maximum_source_line_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_registry_writes": 0,
    "maximum_private_write_bytes": 4_194_304,
    "maximum_runtime_seconds_per_process": 3_600,
    "maximum_runtime_seconds_total": 7_200,
    "maximum_network_bytes": 0,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "maximum_retries_after_started_pass": 0,
}
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v15.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_candidate_mix_reaudit_v16.py",
    "runner": "scripts/reaudit_sft_candidate_mix_v16.py",
    "verifier": "scripts/verify_sft_candidate_mix_reaudit_v16.py",
    "test": "tests/test_candidate_mix_reaudit_v16.py",
}
EXPECTED_INPUTS = {
    "predecessor_plan": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
        "bytes": 7_691,
        "sha256": "7d37369d115b0c666d96d138508f7b31a442cc8c6d990470ec62cf47096282fb",
    },
    "predecessor_report": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.15.json",
        "bytes": 24_444,
        "sha256": "fc81dc5827ebc1a039df7c7692b9f4ccd9144ad1ba1f2a32754ff6f6449d94bf",
        "internal_sha256": (
            "19fc82952653c7e1a022338fbe58bafeae25a6a88d63a2a3e3bc4154e5d99504"
        ),
    },
    "predecessor_verification": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.15.json",
        "bytes": 1_498,
        "sha256": "1af422f2fbb5dcf21b2d4212891ab4d60c9c4cb63a797ac87acfa7598b9be782",
        "internal_sha256": (
            "bf11960dee2f429d11c964ef2e3f8960062796bf950d27f8e55a88576f19b9da"
        ),
    },
    "promotion_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-parser-source-promotion-v0.6.json",
        "bytes": 5_675,
        "sha256": "da2245705a12cffc33f6bebdc21fc56d6fcdafca31b020934e8045c98f7ca030",
    },
    "promotion_receipt": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-v0.6.json",
        "bytes": 5_366,
        "sha256": "641785b91028af77fae55f9f806a3d767151bb895ec6c772488debdd0cafcf57",
        "internal_sha256": (
            "e4d96769dd3193b5e0de79e8d6b4d233ef388dd118be0b3e3f4680f9979e1d01"
        ),
    },
    "promotion_verification": {
        "path": (
            "evidence/lumi-4b-sft-massive-bn-bd-parser-source-promotion-"
            "verification-v0.6.json"
        ),
        "bytes": 1_974,
        "sha256": "a75985d10087db6d884ddbc935b07ca58d93e27b73fb557550790df640b7e8e7",
        "internal_sha256": (
            "e175c703e82dbff1d4914d8dcb091621fb8a9f2548137733d05e8ed5190e0bff"
        ),
    },
    "massive_candidate": {
        "path": (
            "../lumi-artifacts/4b-sft-massive-bn-bd-parser-canary-v0.3/"
            "38be2e88eabf/attempt-001/private/candidate-rows.jsonl"
        ),
        "bytes": 140_522,
        "sha256": "8e107adb12839baf9e0aca0d9eebe0b058a2e69d2e9e9b5f4985b69c143e243c",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.22.json",
        "bytes": 29_673,
        "sha256": "c472d2ca1fe16beff9700719d53a3f8b539073bf52a6a2fa297da192d944f6b8",
        "canonical_sha256": (
            "02a8de18c8a22dd62ca77e4cc5f094ab7e60116999dea0b02f7d7f5e38c1a240"
        ),
    },
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.16.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.16.json"
    ),
}

EXPECTED_INPUT_RECORDS = 314_780
EXPECTED_INPUT_TRAIN_TOKENS = 10_610_600
EXPECTED_INPUT_VALIDATION_TOKENS = 628_604
EXPECTED_INCLUDED_SOURCE_COUNT = 18
EXPECTED_REGISTRY_SOURCE_COUNT = 25
EXPECTED_EXCLUDED_SOURCE_IDS = frozenset(v15.EXPECTED_EXCLUDED_SOURCE_IDS)
EXPECTED_EVALUATION_PROHIBITION = dict(v15.EXPECTED_EVALUATION_PROHIBITION)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v15.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_TOKENIZER_SHA256 = v15.EXPECTED_TOKENIZER_SHA256
EXPECTED_CHAT_TEMPLATE_SHA256 = v15.EXPECTED_CHAT_TEMPLATE_SHA256
BILLSUM_SOURCE = dict(v15.BILLSUM_SOURCE)
BANGLABRIDGE_SOURCE = dict(v15.BANGLABRIDGE_SOURCE)
MASSIVE_SOURCE = {
    "candidate_sha256": EXPECTED_INPUTS["massive_candidate"]["sha256"],
    "family": "amazon_massive_family",
    "language_policy": "language_field",
    "revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
    "source_id": "amazon_massive_bn_bd_tool_canary",
}
EXPECTED_TASK_BEHAVIOR = {
    "coding": 2_151_155,
    "conversation": 494_066,
    "factual_qa": 1_385_694,
    "multilingual_bengali": 1_012_405,
    "reasoning_math_science": 2_269_713,
    "safety_helpfulness": 936_253,
    "tool_api": 806_449,
    "writing_summarization": 1_254_865,
}
EXPECTED_LANGUAGE_BEHAVIOR = {
    "banglish": 27_886,
    "bengali_script": 1_019_973,
    "english": 9_228_161,
    "natural_code_switching": 34_580,
}
EXPECTED_TASK_GAPS = {
    "coding": 0,
    "conversation": 5_934,
    "factual_qa": 114_306,
    "multilingual_bengali": 0,
    "reasoning_math_science": 0,
    "safety_helpfulness": 0,
    "tool_api": 0,
    "writing_summarization": 0,
}
EXPECTED_LANGUAGE_GAPS = {
    "banglish": 772_114,
    "bengali_script": 0,
    "english": 0,
    "natural_code_switching": 365_420,
}
EXPECTED_FAMILY_TOKENS = {
    "allenai_prosocial_family": 899_998,
    "amazon_massive_family": 6_451,
    "banglabridge_family": 250_441,
    "banglatlit_family": 300_000,
    "coherelabs_aya_family": 738_109,
    "deepmind_mathematics_family": 2_221_113,
    "dmath_family": 48_480,
    "fiscalnote_billsum_family": 849_991,
    "lumi_first_party_family": 114_853,
    "nvidia_family": 2_869_687,
    "openassistant_family": 2_311_477,
}
EXPECTED_FAMILY_SHARES = {
    "allenai_prosocial_family": 0.08482065,
    "amazon_massive_family": 0.00060798,
    "banglabridge_family": 0.02360291,
    "banglatlit_family": 0.02827361,
    "coherelabs_aya_family": 0.06956336,
    "deepmind_mathematics_family": 0.20932963,
    "dmath_family": 0.00456902,
    "fiscalnote_billsum_family": 0.08010772,
    "lumi_first_party_family": 0.01082436,
    "nvidia_family": 0.27045473,
    "openassistant_family": 0.21784602,
}
EXPECTED_INPUT_PROJECTION = {
    "final_records": 314_780,
    "final_train_supervised_tokens": 10_610_600,
    "final_validation_supervised_tokens": 628_604,
    "massive_records": 64,
    "massive_train_supervised_tokens": 6_451,
    "massive_validation_supervised_tokens": 0,
    "net_added_records": 64,
    "net_added_train_supervised_tokens": 6_451,
    "net_added_validation_supervised_tokens": 0,
    "predecessor_records": 314_716,
    "predecessor_train_supervised_tokens": 10_604_149,
    "predecessor_validation_supervised_tokens": 628_604,
}
EXPECTED_CLOSURE_BLOCKERS = [
    "task_token_floors_not_met",
    "language_token_floors_not_met",
    "active_production_data_contract_not_updated",
    "dataset_seal_cpu_canary_and_sft_not_authorized",
]
EXPECTED_RECEIPT_MARKER = b'{"started":"candidate_mix_reaudit_v0.16"}\n'
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"candidate_mix_reaudit_verification_v0.16"}\n'
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MASSIVE_RECORD_ID_RE = re.compile(r"^massive-bn-bd-parser-[0-9a-f]{24}$")
_FORBIDDEN_RAW_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "note",
    "record_id",
    "record_id_sha256",
    "reviewer_id",
    "source_text",
    "text",
    "user_prompt",
    "utterance",
}

json_bytes = promotion_v6.json_bytes
strict_json_bytes = promotion_v6.strict_json_bytes
sha256_bytes = promotion_v6.sha256_bytes
start_marker = promotion_v6.start_marker
finish_marker = promotion_v6.finish_marker


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    path = (root / str(binding.get("path"))).resolve()
    expected_parent = root.parent.resolve()
    if not path.is_relative_to(expected_parent) or not path.is_file():
        raise ValueError(f"invalid or missing v0.16 binding: {name}")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"v0.16 binding hash mismatch: {name}")
    return path


def _read_bound_bytes(root: Path, name: str) -> bytes:
    binding = EXPECTED_INPUTS[name]
    path = root / str(binding["path"])
    resolved = path.resolve()
    if not resolved.is_relative_to(root.parent.resolve()):
        raise ValueError("v0.16 bound path escaped")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != binding["bytes"]:
            raise ValueError("v0.16 bound input metadata changed")
        chunks: list[bytes] = []
        remaining = int(binding["bytes"]) + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != binding["bytes"] or sha256_bytes(payload) != binding["sha256"]:
        raise ValueError("v0.16 bound input content changed")
    return payload


def _read_json_input(root: Path, name: str) -> Any:
    return strict_json_bytes(_read_bound_bytes(root, name))


def _validate_internal_seal(
    value: Any,
    binding: Mapping[str, Any],
    seal_key: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("v0.16 sealed root changed")
    unsigned = {key: item for key, item in value.items() if key != seal_key}
    expected = binding["internal_sha256"]
    if value.get(seal_key) != expected or sha256_value(unsigned) != expected:
        raise ValueError("v0.16 bound internal seal changed")


def _assert_raw_text_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_RAW_KEYS:
                raise ValueError("candidate-mix v0.16 retained raw training text")
            _assert_raw_text_free(item)
    elif isinstance(value, list):
        for item in value:
            _assert_raw_text_free(item)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    if resolved != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("candidate-mix v0.16 plan path changed")
    plan = strict_json_bytes(resolved.read_bytes())
    expected_keys = {
        "authorization",
        "authorization_text",
        "authorization_text_sha256",
        "expected",
        "implementation",
        "inputs",
        "limits",
        "near_duplicate_policy",
        "observed_at",
        "output",
        "runner",
        "runtime",
        "schema_version",
        "status",
        "test",
        "verifier",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_text_sha256")
        != EXPECTED_AUTHORIZATION_TEXT_SHA256
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest()
        != EXPECTED_AUTHORIZATION_TEXT_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("expected")
        != {
            "included_candidate_sources": EXPECTED_INCLUDED_SOURCE_COUNT,
            "input_records": EXPECTED_INPUT_RECORDS,
            "input_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "input_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
            "registry_sources": EXPECTED_REGISTRY_SOURCE_COUNT,
        }
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("near_duplicate_policy") != EXPECTED_NEAR_DUPLICATE_POLICY
        or plan.get("observed_at") != "2026-09-01"
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("runtime")
        != {
            "datasketch_version": "2.0.0",
            "python_executable": "/Users/afnan/Documents/lumi-model/.venv/bin/python",
            "python_version": "3.12.13",
        }
    ):
        raise ValueError("candidate-mix v0.16 plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("candidate-mix v0.16 code binding changed")
        code_path = (root / expected_path).resolve()
        if not code_path.is_relative_to(root) or file_sha256(code_path) != binding["sha256"]:
            raise ValueError("candidate-mix v0.16 code hash changed")
    for name, binding in EXPECTED_INPUTS.items():
        unresolved = root / str(binding["path"])
        if not unresolved.resolve().is_relative_to(root.parent.resolve()):
            raise ValueError(f"candidate-mix v0.16 input escaped: {name}")
        path_info = unresolved.lstat()
        if (
            stat.S_ISLNK(path_info.st_mode)
            or not stat.S_ISREG(path_info.st_mode)
            or path_info.st_size != binding["bytes"]
        ):
            raise ValueError(f"candidate-mix v0.16 input metadata changed: {name}")
    return plan


def _validate_predecessor_and_promotion(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_plan_path = (root / EXPECTED_INPUTS["predecessor_plan"]["path"]).resolve()
    predecessor_plan = v15.load_plan(predecessor_plan_path, root)
    predecessor_report = _read_json_input(root, "predecessor_report")
    predecessor_verification = _read_json_input(root, "predecessor_verification")
    _validate_internal_seal(
        predecessor_report,
        EXPECTED_INPUTS["predecessor_report"],
        "report_sha256",
    )
    _validate_internal_seal(
        predecessor_verification,
        EXPECTED_INPUTS["predecessor_verification"],
        "verification_sha256",
    )
    v15.validate_report(predecessor_report)
    v15.validate_verification(predecessor_verification)
    if (
        predecessor_report.get("status")
        != "PASS_registry_v0_21_full_candidate_mix_reaudit_nonproduction"
        or predecessor_report.get("candidate_pool", {}).get("candidate_records_read")
        != 314_716
        or predecessor_report.get("candidate_pool", {}).get(
            "accepted_train_supervised_tokens"
        )
        != 10_604_149
        or predecessor_report.get("candidate_pool", {}).get(
            "accepted_validation_supervised_tokens"
        )
        != 628_604
        or predecessor_report.get("generic_sft_validation", {}).get("rejected") != 0
        or predecessor_report.get("exact_duplicate_audit", {}).get("duplicate_groups")
        != 0
        or predecessor_report.get("near_duplicate_audit", {}).get(
            "multi_record_components"
        )
        != 0
        or predecessor_verification.get("report_file_sha256")
        != EXPECTED_INPUTS["predecessor_report"]["sha256"]
    ):
        raise ValueError("candidate-mix v0.15 clean predecessor evidence changed")

    promotion_plan_path = (root / EXPECTED_INPUTS["promotion_plan"]["path"]).resolve()
    promotion_v6.load_plan(promotion_plan_path, root)
    promotion_receipt = _read_json_input(root, "promotion_receipt")
    promotion_verification = _read_json_input(root, "promotion_verification")
    _validate_internal_seal(
        promotion_receipt,
        EXPECTED_INPUTS["promotion_receipt"],
        "receipt_sha256",
    )
    _validate_internal_seal(
        promotion_verification,
        EXPECTED_INPUTS["promotion_verification"],
        "verification_sha256",
    )
    promotion_v6.validate_receipt(promotion_receipt)
    promotion_v6.validate_verification(promotion_verification)
    if (
        promotion_receipt.get("status") != promotion_v6.RECEIPT_STATUS
        or promotion_receipt.get("candidate_projection")
        != {
            "candidate_sources": 18,
            "records": EXPECTED_INPUT_RECORDS,
            "train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
        }
        or promotion_receipt.get("registry", {}).get("successor_sha256")
        != EXPECTED_INPUTS["source_registry"]["sha256"]
        or promotion_receipt.get("production_eligible") is not False
        or promotion_verification.get("status") != promotion_v6.VERIFICATION_STATUS
        or promotion_verification.get("receipt_file_sha256")
        != EXPECTED_INPUTS["promotion_receipt"]["sha256"]
        or promotion_verification.get("decision", {}).get(
            "registry_v0_22_materialized"
        )
        is not True
    ):
        raise ValueError("MASSIVE registry-v0.22 promotion evidence changed")
    return predecessor_plan, predecessor_report


def _source_inventory(
    *,
    root: Path,
    predecessor_plan: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, Path],
    Mapping[str, Path],
]:
    predecessor_bound = {
        name: v15._resolve_binding(root, binding, name)
        for name, binding in predecessor_plan["bindings"].items()
    }
    v15._validate_predecessor_and_promotion(predecessor_bound)
    source_plans, candidate_paths, _, old_resolved = v15._source_inventory(
        root=root,
        bound=predecessor_bound,
    )
    source_id = MASSIVE_SOURCE["source_id"]
    if source_id in source_plans or source_id in candidate_paths:
        raise ValueError("MASSIVE source already exists in predecessor candidate inventory")
    source_plans[source_id] = {
        "family": MASSIVE_SOURCE["family"],
        "language_policy": MASSIVE_SOURCE["language_policy"],
        "revision": MASSIVE_SOURCE["revision"],
        "source_id": source_id,
    }
    candidate_paths[source_id] = (
        root / EXPECTED_INPUTS["massive_candidate"]["path"]
    ).resolve()
    if (
        file_sha256(candidate_paths[source_id])
        != EXPECTED_INPUTS["massive_candidate"]["sha256"]
        or len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT
        or len(candidate_paths) != EXPECTED_INCLUDED_SOURCE_COUNT
    ):
        raise ValueError("MASSIVE v0.16 source inventory changed")
    return source_plans, candidate_paths, predecessor_bound, old_resolved


def _validate_massive_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    ids: set[str] = set()
    historical_owner = 0
    historical_registry = 0
    if len(rows) != 64:
        raise ValueError("MASSIVE candidate row count changed")
    for row in rows:
        source = row.get("source")
        provenance = row.get("provenance")
        messages = row.get("messages")
        record_id = row.get("record_id")
        if (
            not isinstance(record_id, str)
            or _MASSIVE_RECORD_ID_RE.fullmatch(record_id) is None
            or record_id in ids
            or not isinstance(source, Mapping)
            or source.get("source_id") != MASSIVE_SOURCE["source_id"]
            or source.get("dataset_id") != "AmazonScience/massive"
            or source.get("revision") != MASSIVE_SOURCE["revision"]
            or source.get("license_id") != "cc-by-4.0"
            or row.get("language") != "bn"
            or row.get("split") != "train"
            or row.get("task_category") != "tool_api"
            or not isinstance(messages, list)
            or [message.get("role") for message in messages if isinstance(message, Mapping)]
            != ["user", "tool", "tool_result", "assistant"]
            or any(
                not isinstance(message, Mapping)
                or not isinstance(message.get("content"), str)
                for message in messages
            )
            or not isinstance(provenance, Mapping)
        ):
            raise ValueError("MASSIVE candidate row contract changed")
        ids.add(record_id)
        historical_owner += int(
            provenance.get("owner_review_status")
            == "pending_deterministic_64_row_spot_check"
        )
        historical_registry += int(
            provenance.get("source_registry_status") == "pending_separate_promotion"
        )
    if historical_owner != 64 or historical_registry != 64:
        raise ValueError("MASSIVE historical external-gate provenance changed")
    return {
        "historical_owner_review_pending_rows": historical_owner,
        "historical_registry_pending_rows": historical_registry,
    }


def _normalize_train_only_source_metrics(metrics: dict[str, Any]) -> None:
    by_split = metrics.get("by_split")
    if not isinstance(by_split, dict) or not isinstance(by_split.get("train"), dict):
        raise ValueError("train-only source metrics changed")
    validation = by_split.get("validation")
    if validation not in ({}, None):
        raise ValueError("train-only source unexpectedly contains validation metrics")
    by_split["validation"] = {
        "records": 0,
        "serialized_tokens": 0,
        "supervised_tokens": 0,
    }


def rebuild_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    started: float | None = None,
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(plan_file_sha256) is None:
        raise ValueError("candidate-mix v0.16 plan file hash changed")
    root = Path(repository_root).resolve()
    started_at = time.monotonic() if started is None else started
    predecessor_plan, predecessor_report = _validate_predecessor_and_promotion(root)
    source_plans, candidate_paths, predecessor_bound, old_resolved = _source_inventory(
        root=root,
        predecessor_plan=predecessor_plan,
    )

    base_policy = load_sft_policy(predecessor_bound["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
    )
    benchmark_registry = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != EXPECTED_BENCHMARK_REGISTRY_SHA256:
        raise ValueError("candidate-mix v0.16 benchmark registry changed")
    registry = load_sft_source_registry(
        (root / EXPECTED_INPUTS["source_registry"]["path"]).resolve(),
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_INPUTS["source_registry"]["sha256"],
    )
    registry_raw = _read_json_input(root, "source_registry")
    if (
        len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT
        or not isinstance(registry_raw, Mapping)
        or registry_raw.get("sources", [])[-1] != promotion_v6.EXPECTED_SOURCE
    ):
        raise ValueError("candidate-mix v0.16 registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.16 excluded-source inventory changed")
    excluded = predecessor_report.get("excluded_source_gate", {}).get(
        "predecessor_evidence_replayed"
    )
    if not isinstance(excluded, list):
        raise ValueError("candidate-mix v0.16 predecessor exclusions changed")

    template = load_chat_template(predecessor_bound["chat_template"])
    if file_sha256(predecessor_bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.16 chat template changed")
    if file_sha256(predecessor_bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.16 tokenizer changed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(predecessor_bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    massive_history: dict[str, int] | None = None
    for source_id in sorted(source_plans):
        if time.monotonic() - started_at > EXPECTED_LIMITS[
            "maximum_runtime_seconds_per_process"
        ]:
            raise TimeoutError("candidate-mix v0.16 runtime cap reached")
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"candidate-mix v0.16 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        if source_id == BILLSUM_SOURCE["source_id"]:
            v15._validate_billsum_rows(rows)
        elif source_id == BANGLABRIDGE_SOURCE["source_id"]:
            v15._validate_banglabridge_rows(rows)
        elif source_id == MASSIVE_SOURCE["source_id"]:
            massive_history = _validate_massive_rows(rows)
        metrics = v13.v18._source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        if source_id == MASSIVE_SOURCE["source_id"]:
            if massive_history is None:
                raise ValueError("MASSIVE historical counters missing")
            _normalize_train_only_source_metrics(metrics)
            metrics.update(massive_history)
            metrics.update(
                {
                    "owner_review_external_receipt_resolves_historical_pending": True,
                    "registry_v0_22_receipt_resolves_historical_pending": True,
                }
            )
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    input_train, input_validation, *_ = v13.v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("candidate-mix v0.16 input totals changed")

    accepted, generic = filter_sft_records(
        all_rows,
        policy=audit_policy,
        benchmark_registry=benchmark_registry,
        source_registry=registry,
    )
    accepted_metrics = v13.v18._accepted_metrics(
        rows=accepted,
        source_plans=source_plans,
        tokenizer=tokenizer,
        template=template,
    )
    rows_by_id = {str(row["record_id"]): row for row in all_rows}
    quarantine: list[dict[str, Any]] = []
    for decision in generic["decisions"]:
        if decision.get("accepted") is True:
            continue
        row = rows_by_id[str(decision["record_id"])]
        encoded = encode_sft_messages(
            row["messages"],
            tokenizer=tokenizer,
            template=template,
        )
        quarantine.append(
            {
                "reasons": decision["reasons"],
                "record_id_sha256": sha256_bytes(str(row["record_id"]).encode()),
                "source_id": row["source"]["source_id"],
                "split": row["split"],
                "supervised_tokens": encoded.supervised_tokens,
            }
        )

    duplicate_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in accepted:
        duplicate_index[_canonical_message_digest(row["messages"])].append(
            (row["source"]["source_id"], row["split"])
        )
    exact = _duplicate_snapshot(duplicate_index)
    near = v2._near_duplicate_audit(accepted, policy=plan["near_duplicate_policy"])
    targets = _targets(_load_json(old_resolved["missing_mix_proposal"]))

    train_tokens = int(accepted_metrics["train_supervised_tokens"])
    task_behavior = Counter(accepted_metrics["task_train_supervised_tokens_behavior_safe"])
    language_behavior = Counter(
        accepted_metrics["language_train_supervised_tokens_behavior_safe"]
    )
    task_gaps = {
        task: max(0, int(targets["task_minimums"][task]) - task_behavior[task])
        for task in EXPECTED_TASKS
    }
    language_gaps = {
        mode: max(0, int(targets["language_minimums"][mode]) - language_behavior[mode])
        for mode in EXPECTED_LANGUAGE_MODES
    }
    family_tokens = accepted_metrics["family_train_supervised_tokens"]
    family_shares = {
        family: round(int(tokens) / train_tokens, 8)
        for family, tokens in sorted(family_tokens.items())
    }
    family_cap_violations = [
        family
        for family, share in family_shares.items()
        if share > float(targets["source_cap"])
    ]
    family_dilution_gap = max(
        (
            max(0, math.ceil(int(tokens) / float(targets["source_cap"]) - train_tokens))
            for tokens in family_tokens.values()
        ),
        default=0,
    )
    numeric_gap = max(0, int(targets["total"]["minimum"]) - train_tokens)
    strict_gap = max(
        numeric_gap,
        sum(task_gaps.values()),
        sum(language_gaps.values()),
        family_dilution_gap,
    )
    hygiene_needed = bool(
        generic["rejected"] or exact["duplicate_groups"] or near["multi_record_components"]
    )
    blockers: list[str] = []
    if generic["rejected"]:
        blockers.append("generic_filter_quarantine_not_materialized")
    if exact["duplicate_groups"]:
        blockers.append("cross_source_exact_duplicates_not_materialized")
    if near["multi_record_components"]:
        blockers.append("cross_source_near_duplicates_not_materialized")
    if numeric_gap:
        blockers.append("accepted_train_supervised_tokens_below_10m_floor")
    if any(task_gaps.values()):
        blockers.append("task_token_floors_not_met")
    if any(language_gaps.values()):
        blockers.append("language_token_floors_not_met")
    if family_cap_violations:
        blockers.append("source_family_concentration_cap_not_met")
    blockers.extend(
        [
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ]
    )

    generic_summary = {key: value for key, value in generic.items() if key != "decisions"}
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_records_after_generic_quarantine": len(accepted),
            "accepted_train_supervised_tokens": train_tokens,
            "accepted_validation_supervised_tokens": accepted_metrics[
                "validation_supervised_tokens"
            ],
            "candidate_records_read": len(all_rows),
            "input_train_supervised_tokens": input_train,
            "input_validation_supervised_tokens": input_validation,
            "numeric_gap_to_10m": numeric_gap,
            "strict_targeted_gap_lower_bound": strict_gap,
            **accepted_metrics,
        },
        "closure": {
            "blockers": blockers,
            "family_cap_dilution_gap": family_dilution_gap,
            "language_floor_gaps_behavior_safe": language_gaps,
            "production_dataset_ready": False,
            "source_family_cap": targets["source_cap"],
            "source_family_cap_violations": family_cap_violations,
            "source_family_shares": family_shares,
            "task_floor_gaps_behavior_safe": task_gaps,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "exact_duplicate_audit": exact,
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": excluded,
            "registry_sources_not_in_candidate_inventory": sorted(excluded_source_ids),
        },
        "external_gate_resolution": {
            "massive_historical_owner_review_pending_rows": 64,
            "massive_historical_registry_pending_rows": 64,
            "owner_review_receipt_accepted_rows": 64,
            "owner_review_receipt_rejected_rows": 0,
            "row_mutation_used": False,
            "source_registry_v0_22_materialized": True,
        },
        "generic_sft_validation": generic_summary,
        "input_projection": EXPECTED_INPUT_PROJECTION,
        "near_duplicate_audit": near,
        "next_gate": {
            "action": (
                "materialize_registry_v0_22_hygiene_successor"
                if hygiene_needed
                else "close_residual_behavior_safe_task_and_language_gaps"
            ),
            "authorization_required": True,
            "priority": (
                "hygiene_before_gap_closure" if hygiene_needed else "targeted_gap_closure"
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "quarantine": {
            "materialized": False,
            "reason_counts": generic["rejection_counts"],
            "records": quarantine,
            "total_records": len(quarantine),
        },
        "raw_training_text_retained": False,
        "registry": {
            "file_sha256": EXPECTED_INPUTS["source_registry"]["sha256"],
            "source_count": len(registry.by_id),
        },
        "resource_accounting": {
            "candidate_row_reads": EXPECTED_INPUT_RECORDS,
            "full_audit_passes": 1,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_metrics": source_metrics,
        "status": REPORT_STATUS,
        "token_consistency": {
            "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
            "predecessor_report_file_sha256": EXPECTED_INPUTS["predecessor_report"][
                "sha256"
            ],
            "pure_byte_level_tokenizer_used": True,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.16 report")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.16 report internal hash mismatch")
    pool = report.get("candidate_pool", {})
    generic = report.get("generic_sft_validation", {})
    exact = report.get("exact_duplicate_audit", {})
    near = report.get("near_duplicate_audit", {})
    closure = report.get("closure", {})
    source_metrics = report.get("source_metrics", {})
    massive = source_metrics.get(MASSIVE_SOURCE["source_id"], {})
    excluded_gate = report.get("excluded_source_gate", {})
    predecessor_exclusions = excluded_gate.get("predecessor_evidence_replayed")
    expected_next = {
        "action": "close_residual_behavior_safe_task_and_language_gaps",
        "authorization_required": True,
        "priority": "targeted_gap_closure",
    }
    if (
        report.get("status") != REPORT_STATUS
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("observed_at") != "2026-09-01"
        or not isinstance(report.get("plan_file_sha256"), str)
        or _SHA256_RE.fullmatch(report["plan_file_sha256"]) is None
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("accepted_records_after_generic_quarantine")
        != EXPECTED_INPUT_RECORDS
        or pool.get("accepted_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("accepted_validation_supervised_tokens")
        != EXPECTED_INPUT_VALIDATION_TOKENS
        or pool.get("numeric_gap_to_10m") != 0
        or pool.get("strict_targeted_gap_lower_bound") != 1_137_534
        or pool.get("task_train_supervised_tokens_behavior_safe")
        != EXPECTED_TASK_BEHAVIOR
        or pool.get("language_train_supervised_tokens_behavior_safe")
        != EXPECTED_LANGUAGE_BEHAVIOR
        or pool.get("family_train_supervised_tokens") != EXPECTED_FAMILY_TOKENS
        or pool.get("source_train_supervised_tokens", {}).get(
            MASSIVE_SOURCE["source_id"]
        )
        != 6_451
        or report.get("registry")
        != {
            "file_sha256": EXPECTED_INPUTS["source_registry"]["sha256"],
            "source_count": EXPECTED_REGISTRY_SOURCE_COUNT,
        }
        or not isinstance(source_metrics, Mapping)
        or len(source_metrics) != EXPECTED_INCLUDED_SOURCE_COUNT
        or report.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_INPUT_RECORDS,
            "full_audit_passes": 1,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
        or excluded_gate.get("candidate_rows_from_excluded_sources") != 0
        or excluded_gate.get("registry_sources_not_in_candidate_inventory")
        != sorted(EXPECTED_EXCLUDED_SOURCE_IDS)
        or not isinstance(predecessor_exclusions, list)
        or len(predecessor_exclusions) != 1
        or predecessor_exclusions[0].get("source_id") != "openr1_math_220k"
        or predecessor_exclusions[0].get("counted_train_supervised_tokens") != 0
        or generic.get("accepted") != EXPECTED_INPUT_RECORDS
        or generic.get("rejected") != 0
        or generic.get("total") != EXPECTED_INPUT_RECORDS
        or generic.get("rejection_counts") != {}
        or exact.get("duplicate_groups") != 0
        or exact.get("cross_source_duplicate_groups") != 0
        or exact.get("train_validation_overlap_groups") != 0
        or exact.get("duplicate_extra_records") != 0
        or near.get("comparison_unit") != "role_tagged_full_conversation"
        or near.get("records") != EXPECTED_INPUT_RECORDS
        or near.get("components") != EXPECTED_INPUT_RECORDS
        or near.get("multi_record_components") != 0
        or near.get("cross_source_components") != 0
        or near.get("train_validation_components") != 0
        or near.get("train_validation_records") != 0
        or near.get("duplicate_extra_records") != 0
        or near.get("edge_count") != 0
        or near.get("largest_component_records") != 1
        or near.get("component_ledger_sha256")
        != "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        or closure.get("language_floor_gaps_behavior_safe")
        != EXPECTED_LANGUAGE_GAPS
        or closure.get("task_floor_gaps_behavior_safe") != EXPECTED_TASK_GAPS
        or closure.get("family_cap_dilution_gap") != 0
        or closure.get("production_dataset_ready") is not False
        or closure.get("source_family_cap") != 0.35
        or closure.get("source_family_cap_violations") != []
        or closure.get("source_family_shares") != EXPECTED_FAMILY_SHARES
        or closure.get("blockers") != EXPECTED_CLOSURE_BLOCKERS
        or massive.get("candidate_file_sha256")
        != EXPECTED_INPUTS["massive_candidate"]["sha256"]
        or massive.get("records") != 64
        or massive.get("language_mode_policy") != "language_field"
        or massive.get("maximum_record_tokens") != 309
        or massive.get("rows_with_pending_owner_review_provenance") != 0
        or massive.get("historical_owner_review_pending_rows") != 64
        or massive.get("historical_registry_pending_rows") != 64
        or massive.get("owner_review_external_receipt_resolves_historical_pending")
        is not True
        or massive.get("registry_v0_22_receipt_resolves_historical_pending")
        is not True
        or massive.get("by_split", {}).get("train")
        != {"records": 64, "serialized_tokens": 12_754, "supervised_tokens": 6_451}
        or massive.get("by_split", {}).get("validation")
        != {"records": 0, "serialized_tokens": 0, "supervised_tokens": 0}
        or massive.get("train_by_language_mode") != {"bengali_script": 6_451}
        or massive.get("train_by_task") != {"tool_api": 6_451}
        or report.get("external_gate_resolution")
        != {
            "massive_historical_owner_review_pending_rows": 64,
            "massive_historical_registry_pending_rows": 64,
            "owner_review_receipt_accepted_rows": 64,
            "owner_review_receipt_rejected_rows": 0,
            "row_mutation_used": False,
            "source_registry_v0_22_materialized": True,
        }
        or report.get("input_projection") != EXPECTED_INPUT_PROJECTION
        or report.get("next_gate") != expected_next
        or report.get("quarantine", {}).get("total_records") != 0
        or report.get("quarantine", {}).get("records") != []
    ):
        raise ValueError("candidate-mix v0.16 report boundary or totals changed")
    _assert_raw_text_free(report)


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    if _SHA256_RE.fullmatch(report_file_sha256) is None:
        raise ValueError("candidate-mix v0.16 report file hash changed")
    verification: dict[str, Any] = {
        "checks": {
            "all_314780_rows_replayed_with_pure_byte_level_tokenizer": True,
            "billsum_test_and_ca_test_absent_and_evaluation_prohibition_preserved": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "massive_64_rows_owner_and_registry_receipts_resolved_without_mutation": True,
            "no_network_gpu_inference_seal_sft_registry_or_production_action": True,
            "predecessor_v0_15_and_massive_v0_6_promotion_bound": True,
            "raw_training_text_absent_from_outputs": True,
            "registry_v0_22_25_source_inventory_reproduced": True,
            "source_family_task_and_language_ledgers_reproduced": True,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "resource_accounting": {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "verification_sha256": "",
        "zero_use": {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        },
    }
    verification["verification_sha256"] = sha256_value(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
    checks = verification.get("checks")
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("production_eligible") is not False
        or verification.get("evaluation_prohibition")
        != EXPECTED_EVALUATION_PROHIBITION
        or not isinstance(checks, Mapping)
        or len(checks) != 10
        or not all(value is True for value in checks.values())
        or verification.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
        or verification.get("zero_use")
        != {
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "training_actions": 0,
        }
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256_RE.fullmatch(verification[field]) is None
            for field in ("report_file_sha256", "report_internal_sha256")
        )
    ):
        raise ValueError("candidate-mix v0.16 verification boundary changed")
    _assert_raw_text_free(verification)


def preflight_runner(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / str(plan["output"]["report_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if (
        not report.is_relative_to(root / "evidence")
        or not verification.is_relative_to(root / "evidence")
        or any(os.path.lexists(path) for path in (report, verification))
    ):
        raise FileExistsError("candidate-mix v0.16 pass already started")
    return report, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / str(plan["output"]["report_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if (
        not report.is_file()
        or report.stat().st_size <= len(EXPECTED_RECEIPT_MARKER)
        or os.path.lexists(verification)
    ):
        raise FileExistsError("candidate-mix v0.16 verification preflight changed")
    return report, verification


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    report_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += (root / str(plan[name]["path"])).stat().st_size
    return (
        static
        + len(EXPECTED_RECEIPT_MARKER)
        + len(report_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_FAMILY_SHARES",
    "EXPECTED_INCLUDED_SOURCE_COUNT",
    "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS",
    "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_LANGUAGE_GAPS",
    "EXPECTED_LIMITS",
    "EXPECTED_NEAR_DUPLICATE_POLICY",
    "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_REGISTRY_SOURCE_COUNT",
    "EXPECTED_TASK_GAPS",
    "EXPECTED_VERIFICATION_MARKER",
    "MASSIVE_SOURCE",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_bytes",
    "load_plan",
    "preflight_runner",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "rebuild_report",
    "sha256_bytes",
    "start_marker",
    "strict_json_bytes",
    "validate_report",
    "validate_verification",
]
