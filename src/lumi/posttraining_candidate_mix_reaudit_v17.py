"""Registry-v0.23-bound full local candidate-mix re-audit.

This successor replays the complete clean v0.16 inventory plus the 1,000-row
AI-assisted gap-closure canary.  It performs no mutation, network access,
inference, GPU work, dataset seal, or training action.
"""

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

from . import posttraining_ai_gap_source_registration_v1 as registration
from . import posttraining_candidate_mix_reaudit as v2
from . import posttraining_candidate_mix_reaudit_v13 as v13
from . import posttraining_candidate_mix_reaudit_v15 as v15
from . import posttraining_candidate_mix_reaudit_v16 as v16
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


PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.17"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-report.v0.17"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-verification.v0.17"
PLAN_STATUS = "authorized_registry_v0_23_full_local_reaudit_no_production_action"
REPORT_STATUS = "PASS_registry_v0_23_full_candidate_mix_reaudit_nonproduction"
VERIFICATION_STATUS = "PASS_verified_registry_v0_23_full_candidate_mix_reaudit_nonproduction"

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-candidate-mix-reaudit-v0.17.json")
EXPECTED_AUTHORIZATION_TEXT = "ok shuru kor ar koto gate baki ache"
EXPECTED_AUTHORIZATION_BYTES = 35
EXPECTED_AUTHORIZATION_SHA256 = "462a596e73ce6bf743d1c058b70530914d3700f3c0fa8e24d8a7a509c4575140"
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
    "maximum_candidate_records_per_pass": 315_780,
    "maximum_full_audit_passes": 2,
    "maximum_total_candidate_row_reads": 631_560,
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
EXPECTED_NEAR_DUPLICATE_POLICY = dict(v16.EXPECTED_NEAR_DUPLICATE_POLICY)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_candidate_mix_reaudit_v17.py",
    "runner": "scripts/reaudit_sft_candidate_mix_v17.py",
    "verifier": "scripts/verify_sft_candidate_mix_reaudit_v17.py",
    "test": "tests/test_candidate_mix_reaudit_v17.py",
}
EXPECTED_INPUTS = {
    "predecessor_plan": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.16.json",
        "bytes": 5_458,
        "sha256": "ebfda05b8b7df0325eb5abd4f4b19cca10d8b4d7d6342dd48eb0a4b69d6d6897",
    },
    "predecessor_report": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.16.json",
        "bytes": 25_871,
        "sha256": "e304401f3093ecae5dc5cbc81f443e8795069500ae0803856c488fae39d47777",
        "internal_sha256": "5247a81d4f64edfe33e53801f41599fd6765c4552244c5f20dd35dc2060820c7",
    },
    "predecessor_verification": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.16.json",
        "bytes": 1_746,
        "sha256": "7bd00df1d3fc49341a964c146dca735504eaf177de03eb3c931a3c11185692e5",
        "internal_sha256": "3fe0ebda0bac3d3160705ecdce5af0322c86114b03ac9b965e264d363f8f40ad",
    },
    "registration_plan": {
        "path": "configs/lumi-4b-sft-ai-gap-source-registration-v0.1.json",
        "bytes": 5_017,
        "sha256": "7e0d859d88a82d8349aa2358c4a39ac1131e76093970bda2353394d3f239ce20",
    },
    "registration_receipt": {
        "path": "evidence/lumi-4b-sft-ai-gap-source-registration-v0.1.json",
        "bytes": 2_962,
        "sha256": "b9f42e9be0f6dcfcf32393bda2093bf363f5565a78763d3aa387d531f5b56420",
        "internal_sha256": "51964e6645349b279e4cad8962f1ab9c3f67b48305b1c97c4dd940146b16ee8b",
    },
    "registration_verification": {
        "path": "evidence/lumi-4b-sft-ai-gap-source-registration-verification-v0.1.json",
        "bytes": 1_310,
        "sha256": "a70ea354d1b4312f36a950a866dcc9392e1aa6d60e8af040bec9e0cfdaa7ec41",
        "internal_sha256": "b7ba8b79732db63df864edc3707b9c1ee7921f76fd6ec783164e0d2fe92b0127",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.23.json",
        "bytes": 30_822,
        "sha256": "81009e0c47fda396384264dd285370530cf4b25ef0e55c0aad69a5525b484e94",
    },
    "ai_gap_candidate": {
        "path": "artifacts/lumi-sft-ai-assisted-gap-closure-canary-v0.1/attempt-001/private/candidate-rows.jsonl",
        "bytes": 2_101_054,
        "sha256": "4fae909021ac8900d8f8dca45d9d8e25c446b07597550d4737a7ec60c30bda0a",
    },
}
EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.17.json",
    "verification_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.17.json",
}
EXPECTED_RUNTIME = {
    "datasketch_version": "2.0.0",
    "python_executable": "/Users/afnan/Documents/lumi-model/.venv/bin/python",
    "python_version": "3.12.13",
}
EXPECTED_PLAN_KEYS = frozenset(
    {
        "authorization",
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
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
)

EXPECTED_INPUT_RECORDS = 315_780
EXPECTED_INPUT_TRAIN_TOKENS = 10_697_312
EXPECTED_INPUT_VALIDATION_TOKENS = 650_196
EXPECTED_INCLUDED_SOURCE_COUNT = 19
EXPECTED_REGISTRY_SOURCE_COUNT = 26
EXPECTED_EVALUATION_PROHIBITION = dict(v16.EXPECTED_EVALUATION_PROHIBITION)
EXPECTED_BENCHMARK_REGISTRY_SHA256 = v16.EXPECTED_BENCHMARK_REGISTRY_SHA256
EXPECTED_TOKENIZER_SHA256 = v16.EXPECTED_TOKENIZER_SHA256
EXPECTED_CHAT_TEMPLATE_SHA256 = v16.EXPECTED_CHAT_TEMPLATE_SHA256
EXPECTED_EXCLUDED_SOURCE_IDS = frozenset(v16.EXPECTED_EXCLUDED_SOURCE_IDS)

AI_GAP_SOURCE = {
    "candidate_sha256": EXPECTED_INPUTS["ai_gap_candidate"]["sha256"],
    "family": "lumi_first_party_family",
    "language_policy": "ai_gap_top_level_language_mode_v0.1",
    "language_contract": "exact_row_language_mode_field",
    "revision": registration.SOURCE_REVISION,
    "source_id": registration.SOURCE_ID,
}

EXPECTED_INPUT_PROJECTION = {
    "predecessor_records": 314_780,
    "predecessor_train_supervised_tokens": 10_610_600,
    "predecessor_validation_supervised_tokens": 628_604,
    "ai_gap_records": 1_000,
    "ai_gap_train_supervised_tokens": 86_712,
    "ai_gap_validation_supervised_tokens": 21_592,
    "final_records": EXPECTED_INPUT_RECORDS,
    "final_train_supervised_tokens": EXPECTED_INPUT_TRAIN_TOKENS,
    "final_validation_supervised_tokens": EXPECTED_INPUT_VALIDATION_TOKENS,
}

EXPECTED_RECEIPT_MARKER = b'{"started":"candidate_mix_reaudit_v0.17"}\n'
EXPECTED_VERIFICATION_MARKER = b'{"started":"candidate_mix_reaudit_verification_v0.17"}\n'

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AI_GAP_RECORD_ID = re.compile(
    r"-(banglish|natural_code_switching)-([0-9]{4})$"
)
_FORBIDDEN_RAW_KEYS = {
    "assistant_response", "content", "messages", "note", "record_id",
    "reviewer_id", "source_text", "text", "user_prompt", "utterance",
}

json_bytes = v16.json_bytes
strict_json_bytes = v16.strict_json_bytes
sha256_bytes = v16.sha256_bytes
start_marker = v16.start_marker
finish_marker = v16.finish_marker


def _resolve_input(root: Path, name: str) -> Path:
    specification = EXPECTED_INPUTS[name]
    path = (root / specification["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid or missing candidate-mix v0.17 input: {name}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"candidate-mix v0.17 input must be a regular file: {name}")
    if info.st_size != specification["bytes"] or file_sha256(path) != specification["sha256"]:
        raise ValueError(f"candidate-mix v0.17 input binding changed: {name}")
    return path


def _read_input_object(root: Path, name: str) -> dict[str, Any]:
    value = strict_json_bytes(_resolve_input(root, name).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"candidate-mix v0.17 input is not an object: {name}")
    return value


def _verify_internal(value: Mapping[str, Any], name: str, field: str) -> None:
    expected = EXPECTED_INPUTS[name].get("internal_sha256")
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != expected or sha256_value(unsigned) != expected:
        raise ValueError(f"candidate-mix v0.17 internal seal changed: {name}")


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
        raise ValueError("candidate-mix v0.17 plan path changed")
    plan_info = resolved.lstat()
    if stat.S_ISLNK(plan_info.st_mode) or not stat.S_ISREG(plan_info.st_mode):
        raise ValueError("candidate-mix v0.17 plan must be a regular file")
    plan = strict_json_bytes(resolved.read_bytes())
    if not isinstance(plan, dict):
        raise ValueError("candidate-mix v0.17 plan must be an object")
    if (
        set(plan) != EXPECTED_PLAN_KEYS
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
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
        raise ValueError("candidate-mix v0.17 plan boundary changed")
    for label, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(label)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256.fullmatch(str(binding["sha256"])) is None
        ):
            raise ValueError(f"candidate-mix v0.17 code binding changed: {label}")
        code_path = (root / expected_path).resolve()
        info = code_path.lstat()
        if (
            not code_path.is_relative_to(root)
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or file_sha256(code_path) != binding["sha256"]
        ):
            raise ValueError(f"candidate-mix v0.17 code file changed: {label}")
    for name in EXPECTED_INPUTS:
        unresolved = root / EXPECTED_INPUTS[name]["path"]
        info = unresolved.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size != EXPECTED_INPUTS[name]["bytes"]:
            raise ValueError(f"candidate-mix v0.17 input metadata changed: {name}")
    return plan


def _validate_predecessor(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_plan_path = _resolve_input(root, "predecessor_plan")
    predecessor_plan = v16.load_plan(predecessor_plan_path, root)
    report = _read_input_object(root, "predecessor_report")
    verification = _read_input_object(root, "predecessor_verification")
    _verify_internal(report, "predecessor_report", "report_sha256")
    _verify_internal(verification, "predecessor_verification", "verification_sha256")
    v16.validate_report(report)
    v16.validate_verification(verification)
    if (
        report.get("status") != v16.REPORT_STATUS
        or report.get("candidate_pool", {}).get("candidate_records_read") != 314_780
        or report.get("candidate_pool", {}).get("accepted_train_supervised_tokens") != 10_610_600
        or report.get("candidate_pool", {}).get("accepted_validation_supervised_tokens") != 628_604
        or report.get("generic_sft_validation", {}).get("rejected") != 0
        or report.get("exact_duplicate_audit", {}).get("duplicate_groups") != 0
        or report.get("near_duplicate_audit", {}).get("multi_record_components") != 0
        or verification.get("report_file_sha256") != EXPECTED_INPUTS["predecessor_report"]["sha256"]
    ):
        raise ValueError("candidate-mix v0.16 clean predecessor changed")
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
        raise ValueError("AI gap source-registration verification changed")
    if (
        receipt.get("production_eligible") is not False
        or receipt.get("candidate", {}).get("records") != 1_000
        or receipt.get("candidate", {}).get("source_revision") != registration.SOURCE_REVISION
        or receipt.get("candidate", {}).get("candidate_rows_read") != 0
        or verification.get("successor_registry_sha256") != EXPECTED_INPUTS["source_registry"]["sha256"]
    ):
        raise ValueError("AI gap source-registration boundary changed")


def _source_inventory(
    *, root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], Mapping[str, Path], list[dict[str, Any]]]:
    predecessor_v15_plan, predecessor_v15_report = v16._validate_predecessor_and_promotion(root)
    source_plans, candidate_paths, predecessor_bound, old_resolved = v16._source_inventory(
        root=root, predecessor_plan=predecessor_v15_plan
    )
    if registration.SOURCE_ID in source_plans or registration.SOURCE_ID in candidate_paths:
        raise ValueError("AI gap source already exists in v0.16 inventory")
    source_plans[registration.SOURCE_ID] = {
        "family": AI_GAP_SOURCE["family"],
        "language_policy": AI_GAP_SOURCE["language_policy"],
        "revision": AI_GAP_SOURCE["revision"],
        "source_id": registration.SOURCE_ID,
    }
    candidate_paths[registration.SOURCE_ID] = _resolve_input(root, "ai_gap_candidate")
    if len(source_plans) != EXPECTED_INCLUDED_SOURCE_COUNT or len(candidate_paths) != EXPECTED_INCLUDED_SOURCE_COUNT:
        raise ValueError("candidate-mix v0.17 source inventory changed")
    excluded = predecessor_v15_report.get("excluded_source_gate", {}).get("predecessor_evidence_replayed")
    if not isinstance(excluded, list):
        raise ValueError("candidate-mix v0.17 predecessor exclusion evidence changed")
    return source_plans, candidate_paths, {**predecessor_bound, **old_resolved}, excluded


def _validate_ai_gap_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1_000:
        raise ValueError("AI gap candidate row count changed")
    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        source = row.get("source")
        provenance = row.get("provenance")
        owner = row.get("owner_review")
        mode = row.get("language_mode")
        record_id_match = _AI_GAP_RECORD_ID.search(record_id) if isinstance(record_id, str) else None
        if (
            not isinstance(record_id, str)
            or record_id in ids
            or not isinstance(source, Mapping)
            or source.get("source_id") != registration.SOURCE_ID
            or source.get("dataset_id") != registration.DATASET_ID
            or source.get("revision") != registration.SOURCE_REVISION
            or source.get("license_id") != "lumi-owned"
            or row.get("language") != "multilingual"
            or row.get("split") not in {"train", "validation"}
            or row.get("task_category") not in set(EXPECTED_TASKS)
            or mode not in {"banglish", "natural_code_switching"}
            or record_id_match is None
            or record_id_match.group(1) != mode
            or not isinstance(provenance, Mapping)
            or provenance.get("model_assistance_used") is not True
            or provenance.get("human_written_from_scratch") is not False
            or provenance.get("runtime_external_model_api_calls") != 0
            or provenance.get("runtime_model_calls") != 0
            or owner
            != {
                "status": "not_required_for_nonproduction_canary",
                "human_review_completed": False,
            }
        ):
            raise ValueError("AI gap candidate row contract changed")
        ids.add(record_id)
        split_counts[str(row["split"])] += 1
        mode_counts[str(mode)] += 1
        task_counts[str(row["task_category"])] += 1
    if split_counts != Counter({"train": 800, "validation": 200}):
        raise ValueError("AI gap split counts changed")
    if mode_counts != Counter({"banglish": 666, "natural_code_switching": 334}):
        raise ValueError("AI gap language-mode counts changed")
    return {
        "human_authorship_claimed_rows": 0,
        "human_review_completed_rows": 0,
        "language_mode_field_validated_rows": 1_000,
        "model_assisted_rows": 1_000,
        "runtime_inference_call_rows": 0,
        "split_records": dict(sorted(split_counts.items())),
        "language_mode_records": dict(sorted(mode_counts.items())),
        "task_records": dict(sorted(task_counts.items())),
    }


def _mode_from_row(row: Mapping[str, Any], policy: str) -> str:
    if policy == AI_GAP_SOURCE["language_policy"]:
        mode = row.get("language_mode")
        if row.get("language") == "multilingual" and mode in {
            "banglish",
            "natural_code_switching",
        }:
            return str(mode)
        raise ValueError("cannot resolve exact AI gap language_mode field")
    return v13.v18._mode_from_row(row, policy)


def _source_metrics(
    *,
    rows: Sequence[dict[str, Any]],
    source_id: str,
    family: str,
    language_policy: str,
    candidate_sha256: str,
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    split_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    train_tasks: Counter[str] = Counter()
    train_languages: Counter[str] = Counter()
    maximum_record_tokens = 0
    pending = 0
    for row in rows:
        if row.get("source", {}).get("source_id") != source_id:
            raise ValueError(f"row source mismatch: {source_id}")
        split = row.get("split")
        task = row.get("task_category")
        if split not in {"train", "validation"} or not isinstance(task, str):
            raise ValueError(f"row split/task mismatch: {source_id}")
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        tokens = encoded.supervised_tokens
        maximum_record_tokens = max(maximum_record_tokens, len(encoded.input_ids))
        bucket = split_metrics[str(split)]
        bucket["records"] += 1
        bucket["serialized_tokens"] += len(encoded.input_ids)
        bucket["supervised_tokens"] += tokens
        if split == "train":
            train_tasks[str(task)] += tokens
            train_languages[_mode_from_row(row, language_policy)] += tokens
        if row.get("provenance", {}).get("owner_review_status") == "pending_deterministic_spot_check":
            pending += 1
    return {
        "auxiliary_only_behavior_credit": source_id != "banglatlit_auxiliary_closure_canary",
        "by_split": {split: dict(split_metrics[split]) for split in ("train", "validation")},
        "candidate_file_sha256": candidate_sha256,
        "language_mode_policy": language_policy,
        "maximum_record_tokens": maximum_record_tokens,
        "records": len(rows),
        "rows_with_pending_owner_review_provenance": pending,
        "source_family": family,
        "train_by_language_mode": dict(sorted(train_languages.items())),
        "train_by_task": dict(sorted(train_tasks.items())),
    }


def _accepted_metrics(
    *, rows: Sequence[dict[str, Any]], source_plans: Mapping[str, Mapping[str, Any]], tokenizer: Any, template: Any
) -> dict[str, Any]:
    train = 0
    validation = 0
    task_accounting: Counter[str] = Counter()
    task_behavior: Counter[str] = Counter()
    language_accounting: Counter[str] = Counter()
    language_behavior: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    family_tokens: Counter[str] = Counter()
    for row in rows:
        source_id = str(row["source"]["source_id"])
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        tokens = encoded.supervised_tokens
        if row["split"] == "validation":
            validation += tokens
            continue
        train += tokens
        task = str(row["task_category"])
        mode = _mode_from_row(row, str(source_plans[source_id]["language_policy"]))
        task_accounting[task] += tokens
        language_accounting[mode] += tokens
        source_tokens[source_id] += tokens
        family_tokens[str(source_plans[source_id]["family"])] += tokens
        if source_id != "banglatlit_auxiliary_closure_canary":
            task_behavior[task] += tokens
            language_behavior[mode] += tokens
    return {
        "family_train_supervised_tokens": dict(sorted(family_tokens.items())),
        "language_train_supervised_tokens_accounting": dict(sorted(language_accounting.items())),
        "language_train_supervised_tokens_behavior_safe": dict(sorted(language_behavior.items())),
        "source_train_supervised_tokens": dict(sorted(source_tokens.items())),
        "task_train_supervised_tokens_accounting": dict(sorted(task_accounting.items())),
        "task_train_supervised_tokens_behavior_safe": dict(sorted(task_behavior.items())),
        "train_supervised_tokens": train,
        "validation_supervised_tokens": validation,
    }


def rebuild_report(
    *, plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path, started: float | None = None
) -> dict[str, Any]:
    if not isinstance(plan_file_sha256, str) or _SHA256.fullmatch(plan_file_sha256) is None:
        raise ValueError("candidate-mix v0.17 plan file hash changed")
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
    registry_path = _resolve_input(root, "source_registry")
    registry = load_sft_source_registry(
        registry_path,
        allowed_licenses=audit_policy.allowed_licenses,
        allowed_languages=audit_policy.allowed_languages,
        allowed_splits=audit_policy.allowed_splits,
        allowed_tasks=frozenset(audit_policy.task_mix),
        expected_sha256=EXPECTED_INPUTS["source_registry"]["sha256"],
    )
    if len(registry.by_id) != EXPECTED_REGISTRY_SOURCE_COUNT or registry.sources[-1].source_id != registration.SOURCE_ID:
        raise ValueError("candidate-mix v0.17 registry inventory changed")
    excluded_source_ids = frozenset(registry.by_id) - frozenset(source_plans)
    if excluded_source_ids != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("candidate-mix v0.17 excluded-source inventory changed")

    template = load_chat_template(bound["chat_template"])
    if file_sha256(bound["chat_template"]) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("candidate-mix v0.17 chat template changed")
    if file_sha256(bound["tokenizer_json"]) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("candidate-mix v0.17 tokenizer changed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = PureByteLevelBPETokenizer(bound["tokenizer_json"])
    verify_chat_tokenizer(tokenizer, template)

    all_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    ai_history: dict[str, Any] | None = None
    for source_id in sorted(source_plans):
        if time.monotonic() - started_at > EXPECTED_LIMITS["maximum_runtime_seconds_per_process"]:
            raise TimeoutError("candidate-mix v0.17 runtime cap reached")
        source_plan = source_plans[source_id]
        registry_source = registry.by_id.get(source_id)
        if (
            registry_source is None
            or registry_source.revision != source_plan["revision"]
            or registry_source.status not in {"canary_eligible", "candidate_eligible"}
        ):
            raise ValueError(f"candidate-mix v0.17 registry source mismatch: {source_id}")
        rows = _load_jsonl(candidate_paths[source_id])
        if source_id == v16.BILLSUM_SOURCE["source_id"]:
            v15._validate_billsum_rows(rows)
        elif source_id == v16.BANGLABRIDGE_SOURCE["source_id"]:
            v15._validate_banglabridge_rows(rows)
        elif source_id == v16.MASSIVE_SOURCE["source_id"]:
            massive_history = v16._validate_massive_rows(rows)
        elif source_id == registration.SOURCE_ID:
            ai_history = _validate_ai_gap_rows(rows)
        metrics = _source_metrics(
            rows=rows,
            source_id=source_id,
            family=str(source_plan["family"]),
            language_policy=str(source_plan["language_policy"]),
            candidate_sha256=file_sha256(candidate_paths[source_id]),
            tokenizer=tokenizer,
            template=template,
        )
        if source_id == v16.MASSIVE_SOURCE["source_id"]:
            v16._normalize_train_only_source_metrics(metrics)
            metrics.update(massive_history)
        if source_id == registration.SOURCE_ID:
            if ai_history is None:
                raise ValueError("AI gap validation counters missing")
            metrics["language_mode_policy"] = "exact_row_language_mode_field"
            metrics.update(ai_history)
        source_metrics[source_id] = metrics
        all_rows.extend(rows)

    input_train, input_validation, *_ = v13.v18._aggregate_metrics(source_metrics)
    if (
        len(all_rows) != EXPECTED_INPUT_RECORDS
        or input_train != EXPECTED_INPUT_TRAIN_TOKENS
        or input_validation != EXPECTED_INPUT_VALIDATION_TOKENS
    ):
        raise ValueError("candidate-mix v0.17 input totals changed")

    accepted, generic = filter_sft_records(
        all_rows, policy=audit_policy, benchmark_registry=benchmark, source_registry=registry
    )
    accepted_metrics = _accepted_metrics(
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
            "ai_gap_candidate_rows": 1_000,
            "ai_gap_human_authorship_claimed_rows": 0,
            "ai_gap_human_review_completed_rows": 0,
            "ai_gap_model_assisted_rows": 1_000,
            "ai_gap_runtime_inference_call_rows": 0,
            "candidate_rows_mutated": False,
            "source_registry_v0_23_materialized": True,
        },
        "input_projection": EXPECTED_INPUT_PROJECTION,
        "next_gate": {
            "action": (
                "materialize_registry_v0_23_hygiene_successor"
                if hygiene_needed
                else "scale_ai_assisted_gap_closure_to_remaining_behavior_safe_gap"
            ),
            "authorization_required": True,
            "priority": "hygiene_before_scale" if hygiene_needed else "scaled_gap_closure",
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
        raise ValueError("unsupported candidate-mix v0.17 report")
    unsigned = {key: item for key, item in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("candidate-mix v0.17 report internal seal changed")
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
        or registry
        != {"file_sha256": EXPECTED_INPUTS["source_registry"]["sha256"], "source_count": EXPECTED_REGISTRY_SOURCE_COUNT}
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
        raise ValueError("candidate-mix v0.17 report boundary or totals changed")
    ai_metrics = report.get("source_metrics", {}).get(registration.SOURCE_ID, {})
    if (
        ai_metrics.get("records") != 1_000
        or ai_metrics.get("candidate_file_sha256") != EXPECTED_INPUTS["ai_gap_candidate"]["sha256"]
        or ai_metrics.get("language_mode_policy") != "exact_row_language_mode_field"
        or ai_metrics.get("by_split", {}).get("train", {}).get("supervised_tokens") != 86_712
        or ai_metrics.get("by_split", {}).get("validation", {}).get("supervised_tokens") != 21_592
        or ai_metrics.get("human_authorship_claimed_rows") != 0
        or ai_metrics.get("human_review_completed_rows") != 0
        or ai_metrics.get("model_assisted_rows") != 1_000
        or ai_metrics.get("runtime_inference_call_rows") != 0
    ):
        raise ValueError("candidate-mix v0.17 AI gap source metrics changed")
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
            "all_315780_rows_replayed_with_frozen_tokenizer": True,
            "ai_gap_exact_language_mode_field_replayed": True,
            "ai_gap_machine_only_provenance_replayed_without_human_claim": True,
            "exact_and_near_duplicate_results_reproduced": True,
            "generic_schema_rights_provenance_secret_pii_and_benchmark_gates_replayed": True,
            "no_network_gpu_seal_sft_or_production_promotion": True,
            "raw_training_text_absent_from_outputs": True,
            "registry_v0_23_source_family_task_and_language_ledgers_reproduced": True,
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
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("production_eligible") is not False
        or not all(value is True for value in verification.get("checks", {}).values())
        or verification.get("resource_accounting")
        != {
            "candidate_row_reads": EXPECTED_LIMITS["maximum_total_candidate_row_reads"],
            "full_audit_passes": EXPECTED_LIMITS["maximum_full_audit_passes"],
            "registry_writes": 0,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
    ):
        raise ValueError("candidate-mix v0.17 verification boundary changed")
    _assert_raw_text_free(verification)


def preflight_runner(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    if report.exists() or verification.exists():
        raise FileExistsError("candidate-mix v0.17 pass already started")
    return report, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    if not report.is_file() or report.stat().st_size <= len(EXPECTED_RECEIPT_MARKER):
        raise ValueError("candidate-mix v0.17 report is missing or incomplete")
    if verification.exists():
        raise FileExistsError("candidate-mix v0.17 verification already started")
    return report, verification


def prospective_private_write_bytes(
    *, plan: Mapping[str, Any], root: Path, report_bytes: bytes, verification_bytes: bytes
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += (root / EXPECTED_CODE_PATHS[name]).stat().st_size
    return static + len(EXPECTED_RECEIPT_MARKER) + len(EXPECTED_VERIFICATION_MARKER) + len(report_bytes) + len(verification_bytes)


__all__ = [
    "AI_GAP_SOURCE", "EXPECTED_AUTHORIZATION", "EXPECTED_INPUT_RECORDS",
    "EXPECTED_INPUT_TRAIN_TOKENS", "EXPECTED_INPUT_VALIDATION_TOKENS",
    "EXPECTED_LIMITS", "EXPECTED_NEAR_DUPLICATE_POLICY", "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_REGISTRY_SOURCE_COUNT", "EXPECTED_VERIFICATION_MARKER", "REPORT_STATUS",
    "VERIFICATION_STATUS", "_assert_raw_text_free", "_validate_ai_gap_rows",
    "build_verification", "file_sha256", "finish_marker", "json_bytes", "load_plan",
    "preflight_runner", "preflight_verifier", "prospective_private_write_bytes",
    "rebuild_report", "sha256_bytes", "start_marker", "strict_json_bytes",
    "validate_report", "validate_verification", "v2",
]
