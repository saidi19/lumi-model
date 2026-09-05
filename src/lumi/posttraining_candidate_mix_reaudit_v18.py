"""Registry-v0.24-bound full local candidate-mix re-audit."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_ai_assisted_gap_closure_scaled_v2 as scaled
from . import posttraining_ai_gap_scaled_source_registration_v2 as registration
from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v17 as v17
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
from .posttraining_format import encode_sft_messages, load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_sources import load_sft_source_registry


PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.18"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.18"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.18"
PLAN_STATUS = "authorized_registry_v0_24_full_local_reaudit_no_production_action"
REPORT_STATUS = "PASS_registry_v0_24_full_candidate_mix_reaudit_nonproduction"
VERIFICATION_STATUS = "PASS_verified_registry_v0_24_full_candidate_mix_reaudit_nonproduction"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-candidate-mix-reaudit-v0.18.json")
EXPECTED_AUTHORIZATION_TEXT = "ok next step shuru de"
EXPECTED_AUTHORIZATION_BYTES = 21
EXPECTED_AUTHORIZATION_SHA256 = "7a2317268ba301ceb4a30bb9aa558fad1796139ccc3f5e87722f1e944c4ddcd3"
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
    "maximum_candidate_records_per_pass": 329_200,
    "maximum_full_audit_passes": 2,
    "maximum_total_candidate_row_reads": 658_400,
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
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v17.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_candidate_mix_reaudit_v18.py",
    "runner": "scripts/reaudit_sft_candidate_mix_v18.py",
    "verifier": "scripts/verify_sft_candidate_mix_reaudit_v18.py",
    "test": "tests/test_candidate_mix_reaudit_v18.py",
}
EXPECTED_INPUTS = {
    "predecessor_plan": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.17.json",
        "bytes": 5_361,
        "sha256": "79462cc7f77c8909411449b508a1a0cee7fbddd72c85eabb9d0eb9c5b6fd6137",
    },
    "predecessor_report": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.17.json",
        "bytes": 27_375,
        "sha256": "135d8c76aab14752bd2d9c14ca3519bd6ccdab5c27023d6a7b2a645ce516d692",
        "internal_sha256": "29ce079ec97786b7c5aafe2e39c790efd46bc47a198e145a36c919eb261e053a",
    },
    "predecessor_verification": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.17.json",
        "bytes": 1_394,
        "sha256": "96f926768e7a796ab4f05fe684c03441a86ed41edf1c041840088afa70a16d4b",
        "internal_sha256": "49f147e4c9051c2ef214a9c7b87f507ab66be02154a8db36f8a2713f1ab2a2c9",
    },
    "registration_plan": {
        "path": "configs/lumi-4b-sft-ai-gap-scaled-source-registration-v0.2.json",
        "bytes": 5_301,
        "sha256": "260e361d28acdc9a49e0c00064c1922ed010a9591f3a8a180d5fe92e0e464932",
    },
    "registration_receipt": {
        "path": "evidence/lumi-4b-sft-ai-gap-scaled-source-registration-v0.2.json",
        "bytes": 3_069,
        "sha256": "ed702db59b8bd3961efcca13cb50234cabd62e6bc429b0f5c99391feefb2bae9",
        "internal_sha256": "2cab16a06ebee4c546bd29dd3b708d1e834d75ea91d257e0c979ef688987d426",
    },
    "registration_verification": {
        "path": "evidence/lumi-4b-sft-ai-gap-scaled-source-registration-verification-v0.2.json",
        "bytes": 1_324,
        "sha256": "7665b493813057b7ca8fa32c81097ff78108286c19a9139d0d87c3f013585ca7",
        "internal_sha256": "d626b46af4a8075aae35919dd7bf8818066136952866067978c1a7be31502b6a",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.24.json",
        "bytes": 32_000,
        "sha256": "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    },
    "scaled_candidate": {
        "path": "artifacts/lumi-sft-ai-assisted-gap-closure-scaled-v0.2/attempt-001/private/candidate-rows.jsonl",
        "bytes": 28_324_248,
        "sha256": "9642885685993343148d531b00cf8f5e65042ae7b4eed72bae967a41d53b27c2",
    },
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
    "verification_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.18.json",
}
EXPECTED_RUNTIME = dict(v17.EXPECTED_RUNTIME)

EXPECTED_INPUT_RECORDS = 329_200
EXPECTED_INPUT_TRAIN_TOKENS = 11_855_217
EXPECTED_INPUT_VALIDATION_TOKENS = 939_293
EXPECTED_INCLUDED_SOURCE_COUNT = 20
EXPECTED_REGISTRY_SOURCE_COUNT = 27
EXPECTED_EVALUATION_PROHIBITION = dict(v17.EXPECTED_EVALUATION_PROHIBITION)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v17.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_TOKENIZER_SHA256 = v17.EXPECTED_TOKENIZER_SHA256
EXPECTED_CHAT_TEMPLATE_SHA256 = v17.EXPECTED_CHAT_TEMPLATE_SHA256
EXPECTED_EXCLUDED_SOURCE_IDS = frozenset(v17.EXPECTED_EXCLUDED_SOURCE_IDS)

SCALED_SOURCE = {
    "candidate_sha256": EXPECTED_INPUTS["scaled_candidate"]["sha256"],
    "family": "lumi_first_party_family",
    "language_policy": v17.AI_GAP_SOURCE["language_policy"],
    "language_contract": "exact_row_language_mode_field",
    "revision": registration.SOURCE_REVISION,
    "source_id": registration.SOURCE_ID,
}
EXPECTED_INPUT_PROJECTION = {
    "predecessor_records": 315_780,
    "predecessor_train_supervised_tokens": 10_697_312,
    "predecessor_validation_supervised_tokens": 650_196,
    "scaled_records": 13_420,
    "scaled_train_supervised_tokens": 1_157_905,
    "scaled_validation_supervised_tokens": 289_097,
    "final_records": EXPECTED_INPUT_RECORDS,
    "final_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
    "final_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
}
EXPECTED_TASK_ACCOUNTING = {
    "coding": 2_151_155,
    "conversation": 661_952,
    "factual_qa": 1_554_191,
    "multilingual_bengali": 1_474_264,
    "reasoning_math_science": 2_269_713,
    "safety_helpfulness": 1_242_795,
    "tool_api": 978_633,
    "writing_summarization": 1_522_514,
}
EXPECTED_TASK_BEHAVIOR = {
    **EXPECTED_TASK_ACCOUNTING,
    "multilingual_bengali": 1_174_264,
}
EXPECTED_LANGUAGE_ACCOUNTING = {
    "banglish": 1_172_471,
    "bengali_script": 1_019_973,
    "english": 9_228_161,
    "natural_code_switching": 434_612,
}
EXPECTED_LANGUAGE_BEHAVIOR = {
    **EXPECTED_LANGUAGE_ACCOUNTING,
    "banglish": 872_471,
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
    "lumi_first_party_family": 1_359_470,
    "nvidia_family": 2_869_687,
    "openassistant_family": 2_311_477,
}

EXPECTED_RECEIPT_MARKER = b'{"started":"candidate_mix_reaudit_v0.18"}\n'
EXPECTED_VERIFICATION_MARKER = b'{"started":"candidate_mix_reaudit_verification_v0.18"}\n'

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCALED_RECORD_ID = re.compile(
    r"^lumi-ai-gap-v02-(train|validation)-([a-z_]+)-(banglish|natural_code_switching)-([0-9]{5})$"
)
_FORBIDDEN_RAW_KEYS = set(v17._FORBIDDEN_RAW_KEYS)

json_bytes = v17.json_bytes
strict_json_bytes = v17.strict_json_bytes
sha256_bytes = v17.sha256_bytes
start_marker = v17.start_marker
finish_marker = v17.finish_marker


def _resolve_input(root: Path, name: str) -> Path:
    specification = EXPECTED_INPUTS[name]
    path = (root / specification["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing candidate-mix v0.18 input: {name}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"candidate-mix v0.18 input must be a regular file: {name}")
    if info.st_size != specification["bytes"] or file_sha256(path) != specification["sha256"]:
        raise ValueError(f"candidate-mix v0.18 input binding changed: {name}")
    return path


def _read_input_object(root: Path, name: str) -> dict[str, Any]:
    value = strict_json_bytes(_resolve_input(root, name).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"candidate-mix v0.18 input is not an object: {name}")
    return value


def _verify_internal(value: Mapping[str, Any], name: str, field: str) -> None:
    expected = EXPECTED_INPUTS[name].get("internal_sha256")
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != expected or sha256_value(unsigned) != expected:
        raise ValueError(f"candidate-mix v0.18 internal seal changed: {name}")


def _assert_raw_text_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_RAW_KEYS:
                raise ValueError(f"raw candidate key in evidence: {'.'.join(path + (str(key),))}")
            _assert_raw_text_free(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path + (str(index),))


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    if resolved != (root / EXPECTED_PLAN_PATH).resolve():
        raise ValueError("candidate-mix v0.18 plan path changed")
    info = resolved.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("candidate-mix v0.18 plan must be a regular file")
    plan = strict_json_bytes(resolved.read_bytes())
    expected_keys = {
        "authorization", "authorization_bytes", "authorization_sha256",
        "authorization_text", "expected", "implementation", "inputs",
        "limits", "near_duplicate_policy", "observed_at", "output", "runner",
        "runtime", "schema_version", "status", "test", "verifier",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest() != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("near_duplicate_policy") != EXPECTED_NEAR_DUPLICATE_POLICY
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("runtime") != EXPECTED_RUNTIME
        or plan.get("expected")
        != {
            "included_candidate_sources": EXPECTED_INCLUDED_SOURCE_COUNT,
            "input_records": EXPECTED_INPUT_RECORDS,
            "input_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
            "input_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
            "registry_sources": EXPECTED_REGISTRY_SOURCE_COUNT,
        }
        or plan.get("observed_at") != "2026-09-02"
    ):
        raise ValueError("candidate-mix v0.18 plan boundary changed")
    for label, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(label)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or _SHA256.fullmatch(str(binding.get("sha256"))) is None
        ):
            raise ValueError(f"candidate-mix v0.18 code binding changed: {label}")
        code_path = (root / expected_path).resolve()
        code_info = code_path.lstat()
        if (
            not code_path.is_relative_to(root)
            or stat.S_ISLNK(code_info.st_mode)
            or not stat.S_ISREG(code_info.st_mode)
            or file_sha256(code_path) != binding["sha256"]
        ):
            raise ValueError(f"candidate-mix v0.18 code file changed: {label}")
    for name, specification in EXPECTED_INPUTS.items():
        input_path = (root / specification["path"]).resolve()
        input_info = input_path.lstat()
        if stat.S_ISLNK(input_info.st_mode) or not stat.S_ISREG(input_info.st_mode) or input_info.st_size != specification["bytes"]:
            raise ValueError(f"candidate-mix v0.18 input metadata changed: {name}")
    return plan


def _validate_predecessor(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_plan = v17.load_plan(_resolve_input(root, "predecessor_plan"), root)
    report = _read_input_object(root, "predecessor_report")
    verification = _read_input_object(root, "predecessor_verification")
    _verify_internal(report, "predecessor_report", "report_sha256")
    _verify_internal(verification, "predecessor_verification", "verification_sha256")
    v17.validate_report(report)
    v17.validate_verification(verification)
    if (
        report.get("status") != v17.REPORT_STATUS
        or report.get("candidate_pool", {}).get("candidate_records_read") != 315_780
        or report.get("candidate_pool", {}).get("accepted_train_supervised_tokens") != 10_697_312
        or report.get("candidate_pool", {}).get("accepted_validation_supervised_tokens") != 650_196
        or report.get("generic_sft_validation", {}).get("rejected") != 0
        or report.get("exact_duplicate_audit", {}).get("duplicate_groups") != 0
        or report.get("near_duplicate_audit", {}).get("multi_record_components") != 0
        or verification.get("report_file_sha256") != EXPECTED_INPUTS["predecessor_report"]["sha256"]
    ):
        raise ValueError("candidate-mix v0.17 clean predecessor changed")
    return predecessor_plan, report


def _validate_registration(root: Path) -> None:
    plan_path = _resolve_input(root, "registration_plan")
    registration.load_plan(plan_path, repository_root=root)
    receipt = _read_input_object(root, "registration_receipt")
    verification = _read_input_object(root, "registration_verification")
    _verify_internal(receipt, "registration_receipt", "receipt_sha256")
    _verify_internal(verification, "registration_verification", "verification_sha256")
    replay = registration.verify_registration(
        plan_path=plan_path,
        repository_root=root,
        successor_path=_resolve_input(root, "source_registry"),
        receipt_path=_resolve_input(root, "registration_receipt"),
    )
    if replay != verification:
        raise ValueError("scaled source-registration verification changed")
    if (
        receipt.get("production_eligible") is not False
        or receipt.get("candidate", {}).get("records") != 13_420
        or receipt.get("candidate", {}).get("source_revision") != registration.SOURCE_REVISION
        or receipt.get("candidate", {}).get("candidate_rows_read") != 0
        or verification.get("successor_registry_sha256") != EXPECTED_INPUTS["source_registry"]["sha256"]
    ):
        raise ValueError("scaled source-registration boundary changed")


def _source_inventory(
    *, root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], Mapping[str, Path], list[dict[str, Any]]]:
    source_plans, candidate_paths, bound, excluded = v17._source_inventory(root=root)
    if registration.SOURCE_ID in source_plans or registration.SOURCE_ID in candidate_paths:
        raise ValueError("scaled gap source already exists in v0.17 inventory")
    source_plans[registration.SOURCE_ID] = {
        "family": SCALED_SOURCE["family"],
        "language_policy": SCALED_SOURCE["language_policy"],
        "revision": SCALED_SOURCE["revision"],
        "source_id": registration.SOURCE_ID,
    }
    candidate_paths[registration.SOURCE_ID] = _resolve_input(root, "scaled_candidate")
    if len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT or len(candidate_paths) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.18 source inventory changed")
    return source_plans, candidate_paths, bound, excluded


def _validate_scaled_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != scaled.EXPECTED_RECORDS:
        raise ValueError("scaled candidate row count changed")
    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    ids: set[str] = set()
    allowed_tasks = {item[0] for item in scaled.ALLOCATIONS}
    for row in rows:
        record_id = row.get("record_id")
        match = _SCALED_RECORD_ID.fullmatch(record_id) if isinstance(record_id, str) else None
        source = row.get("source")
        provenance = row.get("provenance")
        owner = row.get("owner_review")
        split = row.get("split")
        task = row.get("task_category")
        mode = row.get("language_mode")
        if (
            match is None
            or record_id in ids
            or split != match.group(1)
            or task != match.group(2)
            or mode != match.group(3)
            or task not in allowed_tasks
            or row.get("language") != "multilingual"
            or not isinstance(source, Mapping)
            or source.get("source_id") != registration.SOURCE_ID
            or source.get("dataset_id") != registration.DATASET_ID
            or source.get("revision") != registration.SOURCE_REVISION
            or source.get("license_id") != "lumi-owned"
            or not isinstance(provenance, Mapping)
            or provenance.get("model_assistance_used") is not True
            or provenance.get("human_written_from_scratch") is not False
            or provenance.get("runtime_external_model_api_calls") != 0
            or provenance.get("runtime_model_calls") != 0
            or owner
            != {
                "status": "not_required_for_nonproduction_scaled_candidate",
                "human_review_completed": False,
            }
        ):
            raise ValueError("scaled candidate row contract changed")
        ids.add(record_id)
        split_counts[str(split)] += 1
        mode_counts[str(mode)] += 1
        task_counts[str(task)] += 1
        cell_counts[(str(task), str(mode), str(split))] += 1
    expected_cells = {
        (task, mode, "train"): train
        for task, mode, train, _validation in scaled.ALLOCATIONS
    } | {
        (task, mode, "validation"): validation
        for task, mode, _train, validation in scaled.ALLOCATIONS
    }
    if (
        split_counts != Counter({"train": 10_736, "validation": 2_684})
        or mode_counts != Counter({"banglish": 9_750, "natural_code_switching": 3_670})
        or cell_counts != Counter(expected_cells)
    ):
        raise ValueError("scaled candidate allocation changed")
    return {
        "human_authorship_claimed_rows": 0,
        "human_review_completed_rows": 0,
        "language_mode_field_validated_rows": 13_420,
        "model_assisted_rows": 13_420,
        "runtime_inference_call_rows": 0,
        "split_records": dict(sorted(split_counts.items())),
        "language_mode_records": dict(sorted(mode_counts.items())),
        "task_records": dict(sorted(task_counts.items())),
    }


def rebuild_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str,
    repository_root: str | Path, started: float | None = None,
) -> dict[str, Any]:
    if not isinstance(plan_file_sha256, str) or _SHA256.fullmatch(plan_file_sha256) is None:
        raise ValueError("candidate-mix v0.18 plan file hash changed")
    root = Path(repository_root).resolve()
    started_at = time.monotonic() if started is None else started
    _validate_predecessor(root)
    _validate_registration(root)
    source_plans, candidate_paths, bound, excluded_evidence = _source_inventory(root=root)

    base_policy = load_sft_policy(bound["data_contract"])
    audit_policy = replace(
        base_policy,
        benchmark_decontamination={
            "anchor_ngram_tokens": 8,
            "minimum_anchor_matches": 2,
            "registry": "data/benchmark-signatures-v0.3.jsonl.gz",
            "registry_sha256": EXPECTED_BENCHMARK_REGISTRY_SHA256,
        },
    )
    benchmark = v2.BenchmarkRegistry(audit_policy.benchmark_decontamination, root)
    registry = load_sft_source_registry(
        _resolve_input(root, "source_registry"),
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_INPUTS["source_registry"]["sha256"],
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT or registry.sources[-1].source_id != registration.SOURCE_ID:
        raise ValueError("candidate-mix v0.18 registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.18 excluded-source inventory changed")

    template = load_chat_template(bound["chat_template"])
    if file_sha256(bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.18 chat template changed")
    if file_sha256(bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.18 tokenizer changed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    massive_history: dict[str, Any] | None = None
    canary_history: dict[str, Any] | None = None
    scaled_history: dict[str, Any] | None = None
    for source_id in sorted(source_plans):
        if time.monotonic() - started_at > EXPECTED_LIMITS["maximum_runtime_seconds_per_process"]:
            raise TimeoutError("candidate-mix v0.18 runtime cap reached")
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"candidate-mix v0.18 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        if source_id == v17.v16.BILLSUM_SOURCE["source_id"]:
            v17.v15._validate_billsum_rows(rows)
        elif source_id == v17.v16.BANGLABRIDGE_SOURCE["source_id"]:
            v17.v15._validate_banglabridge_rows(rows)
        elif source_id == v17.v16.MASSIVE_SOURCE["source_id"]:
            massive_history = v17.v16._validate_massive_rows(rows)
        elif source_id == v17.registration.SOURCE_ID:
            canary_history = v17._validate_ai_gap_rows(rows)
        elif source_id == registration.SOURCE_ID:
            scaled_history = _validate_scaled_rows(rows)
        metrics = v17._source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        if source_id == v17.v16.MASSIVE_SOURCE["source_id"]:
            if massive_history is None:
                raise ValueError("MASSIVE validation counters missing")
            v17.v16._normalize_train_only_source_metrics(metrics)
            metrics.update(massive_history)
        if source_id == v17.registration.SOURCE_ID:
            if canary_history is None:
                raise ValueError("AI gap canary validation counters missing")
            metrics["language_mode_policy"] = "exact_row_language_mode_field"
            metrics.update(canary_history)
        if source_id == registration.SOURCE_ID:
            if scaled_history is None:
                raise ValueError("scaled gap validation counters missing")
            metrics["language_mode_policy"] = "exact_row_language_mode_field"
            metrics.update(scaled_history)
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    input_train, input_validation, *_ = v17.v13.v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("candidate-mix v0.18 input totals changed")

    accepted, generic = filter_sft_records(
        all_rows, policy=audit_policy, benchmark_registry=benchmark, source_registry=registry
    )
    accepted_metrics = v17._accepted_metrics(
        rows=accepted, source_plans=source_plans, tokenizer=tokenizer, template=template
    )
    rows_by_id = {str(row["record_id"]): row for row in all_rows}
    quarantine: list[dict[str, Any]] = []
    for decision in generic["decisions"]:
        if decision.get("accepted") is True:
            continue
        row = rows_by_id[str(decision["record_id"])]
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
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
    targets = _targets(_load_json(bound["missing_mix_proposal"]))

    train_tokens = int(accepted_metrics["train_supervised_tokens"])
    task_behavior = Counter(accepted_metrics["task_train_supervised_tokens_behavior_safe"])
    language_behavior = Counter(accepted_metrics["language_train_supervised_tokens_behavior_safe"])
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
        family for family, share in family_shares.items() if share > float(targets["source_cap"])
    ]
    family_dilution_gap = max(
        (
            max(0, math.ceil(int(tokens) / float(targets["source_cap"]) - train_tokens))
            for tokens in family_tokens.values()
        ),
        default=0,
    )
    numeric_gap = max(0, int(targets["total"]["minimum"]) - train_tokens)
    strict_gap = max(numeric_gap, sum(task_gaps.values()), sum(language_gaps.values()), family_dilution_gap)
    hygiene_needed = bool(generic["rejected"] or exact["duplicate_groups"] or near["multi_record_components"])
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
    blockers.extend(["active_production_data_contract_not_updated", "dataset_seal_cpu_canary_and_sft_not_authorized"])
    if hygiene_needed:
        next_action, priority = "materialize_registry_v0_24_hygiene_successor", "hygiene_before_policy"
    elif strict_gap:
        next_action, priority = "materialize_additional_targeted_gap_closure", "floor_closure"
    else:
        next_action, priority = "production_source_policy_and_active_data_contract_decision_before_dataset_seal", "production_data_contract"

    generic_summary = {key: value for key, value in generic.items() if key != "decisions"}
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "authorization": dict(plan["authorization"]),
        "candidate_pool": {
            "accepted_records_after_generic_quarantine": len(accepted),
            "accepted_train_supervised_tokens": train_tokens,
            "accepted_validation_supervised_tokens": accepted_metrics["validation_supervised_tokens"],
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
        "near_duplicate_audit": near,
        "generic_sft_validation": generic_summary,
        "excluded_source_gate": {
            "candidate_rows_from_excluded_sources": 0,
            "predecessor_evidence_replayed": excluded_evidence,
            "registry_sources_not_in_candidate_inventory": sorted(excluded_source_ids),
        },
        "external_gate_resolution": {
            "scaled_candidate_rows": 13_420,
            "scaled_human_authorship_claimed_rows": 0,
            "scaled_human_review_completed_rows": 0,
            "scaled_model_assisted_rows": 13_420,
            "scaled_runtime_inference_call_rows": 0,
            "candidate_rows_mutated": False,
            "source_registry_v0_24_materialized": True,
        },
        "input_projection": EXPECTED_INPUT_PROJECTION,
        "next_gate": {
            "action": next_action,
            "authorization_required": True,
            "priority": priority,
        },
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
        "source_metrics": source_metrics,
        "token_consistency": {
            "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
            "predecessor_report_file_sha256": EXPECTED_INPUTS["predecessor_report"]["sha256"],
            "pure_byte_level_tokenizer_used": True,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-mix v0.18 report")
    unsigned = {key: item for key, item in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.18 report internal seal changed")
    pool = report.get("candidate_pool", {})
    registry = report.get("registry", {})
    resources = report.get("resource_accounting", {})
    generic = report.get("generic_sft_validation", {})
    near = report.get("near_duplicate_audit", {})
    if (
        report.get("status") != REPORT_STATUS
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or pool.get("candidate_records_read") != EXPECTED_INPUT_RECORDS
        or pool.get("input_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
        or pool.get("input_validation_supervised_tokens") != EXPECTED_INPUT_VALIDATION_TOKENS
        or generic.get("total") != EXPECTED_INPUT_RECORDS
        or not isinstance(generic.get("accepted"), int)
        or not isinstance(generic.get("rejected"), int)
        or int(generic["accepted"]) + int(generic["rejected"]) != EXPECTED_INPUT_RECORDS
        or near.get("records") != generic.get("accepted")
        or registry != {"file_sha256": EXPECTED_INPUTS["source_registry"]["sha256"], "source_count": EXPECTED_REGISTRY_SOURCE_COUNT}
        or resources
        != {
            "candidate_row_reads": EXPECTED_INPUT_RECORDS,
            "full_audit_passes": 1,
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
        or len(report.get("source_metrics", {})) != EXPECTED_INCLUDED_SOURCE_COUNT
        or report.get("input_projection") != EXPECTED_INPUT_PROJECTION
    ):
        raise ValueError("candidate-mix v0.18 report boundary or totals changed")
    scaled_metrics = report.get("source_metrics", {}).get(registration.SOURCE_ID, {})
    if (
        scaled_metrics.get("records") != 13_420
        or scaled_metrics.get("candidate_file_sha256") != EXPECTED_INPUTS["scaled_candidate"]["sha256"]
        or scaled_metrics.get("language_mode_policy") != "exact_row_language_mode_field"
        or scaled_metrics.get("by_split", {}).get("train", {}).get("supervised_tokens") != 1_157_905
        or scaled_metrics.get("by_split", {}).get("validation", {}).get("supervised_tokens") != 289_097
        or scaled_metrics.get("train_by_language_mode") != scaled.EXPECTED_TRAIN_TOKENS_BY_MODE
        or scaled_metrics.get("train_by_task") != scaled.EXPECTED_TRAIN_TOKENS_BY_TASK
        or scaled_metrics.get("human_authorship_claimed_rows") != 0
        or scaled_metrics.get("human_review_completed_rows") != 0
        or scaled_metrics.get("model_assisted_rows") != 13_420
        or scaled_metrics.get("runtime_inference_call_rows") != 0
    ):
        raise ValueError("candidate-mix v0.18 scaled source metrics changed")
    if generic.get("rejected") == 0:
        if (
            pool.get("accepted_train_supervised_tokens") != EXPECTED_INPUT_TRAIN_TOKENS
            or pool.get("accepted_validation_supervised_tokens") != EXPECTED_INPUT_VALIDATION_TOKENS
            or pool.get("task_train_supervised_tokens_accounting") != EXPECTED_TASK_ACCOUNTING
            or pool.get("task_train_supervised_tokens_behavior_safe") != EXPECTED_TASK_BEHAVIOR
            or pool.get("language_train_supervised_tokens_accounting") != EXPECTED_LANGUAGE_ACCOUNTING
            or pool.get("language_train_supervised_tokens_behavior_safe") != EXPECTED_LANGUAGE_BEHAVIOR
            or pool.get("family_train_supervised_tokens") != EXPECTED_FAMILY_TOKENS
        ):
            raise ValueError("candidate-mix v0.18 clean accepted ledger changed")
    _assert_raw_text_free(report)


def build_verification(
    *, plan: Mapping[str, Any], report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "all_329200_rows_replayed_with_frozen_tokenizer": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_or_production_promotion": True,
            "raw_training_text_absent_from_outputs": True,
            "registry_v0_24_source_family_task_and_language_ledgers_reproduced": True,
            "scaled_exact_language_mode_field_replayed": True,
            "scaled_machine_only_provenance_replayed_without_human_claim": True,
        },
        "resource_accounting": {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "production_eligible": False,
        "next_gate": report["next_gate"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {key: item for key, item in verification.items() if key != "verification_sha256"}
    checks = verification.get("checks")
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("production_eligible") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or verification.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
    ):
        raise ValueError("candidate-mix v0.18 verification boundary changed")
    _assert_raw_text_free(verification)


def preflight_runner(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    if report.exists() or verification.exists():
        raise FileExistsError("candidate-mix v0.18 pass already started")
    return report, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    if not report.is_file() or report.stat().st_size <= len(EXPECTED_RECEIPT_MARKER):
        raise ValueError("candidate-mix v0.18 report is missing or incomplete")
    if verification.exists():
        raise FileExistsError("candidate-mix v0.18 verification already started")
    return report, verification


def prospective_private_write_bytes(
    *, plan: Mapping[str, Any], root: Path,
    report_bytes: bytes, verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += (root / EXPECTED_CODE_PATHS[name]).stat().st_size
    return static + len(EXPECTED_RECEIPT_MARKER) + len(EXPECTED_VERIFICATION_MARKER) + len(report_bytes) + len(verification_bytes)


__all__ = [
    "EXPECTED_AUTHORIZATION", "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS", "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_LIMITS", "EXPECTED_NEAR_DUPLICATE_POLICY",
    "EXPECTED_RECEIPT_MARKER", "EXPECTED_REGISTRY_SOURCE_COUNT",
    "EXPECTED_VERIFICATION_MARKER", "REPORT_STATUS", "SCALED_SOURCE",
    "VERIFICATION_STATUS", "_assert_raw_text_free", "_validate_scaled_rows",
    "build_verification", "file_sha256", "finish_marker", "json_bytes",
    "load_plan", "preflight_runner", "preflight_verifier",
    "prospective_private_write_bytes", "rebuild_report", "sha256_bytes",
    "start_marker", "strict_json_bytes", "validate_report",
    "validate_verification", "v2",
]
