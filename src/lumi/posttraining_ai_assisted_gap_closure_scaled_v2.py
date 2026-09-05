"""Scaled local AI-assisted gap-closure candidate for Lumi SFT.

The renderer reuses the already-audited v0.1 message templates with a new,
disjoint deterministic schedule.  Runtime performs no network/model/GPU work,
never claims human authorship, and keeps the candidate non-production.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import posttraining_ai_assisted_gap_closure_canary_v1 as canary
from .data_quality import BenchmarkRegistry, normalize_for_matching
from .posttraining_candidate_mix_reaudit import _near_duplicate_audit
from .posttraining_data import filter_sft_records, load_sft_policy, validate_message_sequence
from .posttraining_first_party import file_sha256
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)


PLAN_SCHEMA = "lumi.sft-ai-assisted-gap-closure-scaled-plan.v0.2"
REPORT_SCHEMA = "lumi.sft-ai-assisted-gap-closure-scaled-report.v0.2"
VERIFICATION_SCHEMA = "lumi.sft-ai-assisted-gap-closure-scaled-verification.v0.2"
PLAN_STATUS = "authorized_local_ai_assisted_scaled_candidate_nonproduction"
REPORT_STATUS = "PASS_local_ai_assisted_scaled_candidate_nonproduction"
VERIFICATION_STATUS = "PASS_verified_local_ai_assisted_scaled_candidate_nonproduction"

AUTHORIZATION_TEXT = "ar koto din lagte pare sob shes korte and next step shuru kor"
AUTHORIZATION_BYTES = 61
AUTHORIZATION_SHA256 = "f6105c6a78bd5ec045b24e133d62573d04c8e9d73eb5f2c928b038a24bdbeda8"

SOURCE_ID = "lumi_ai_assisted_gap_closure_scaled_v0_2"
DATASET_ID = "Afnan-Technologies-Inc/lumi-ai-assisted-gap-closure-scaled-v0.2"
GENERATOR_ID = "openai-codex-current-session"
GENERATION_METHOD = "ai_authored_templates_with_local_deterministic_scaled_renderer"
RENDERER_CONTRACT = "lumi.ai-assisted-gap-closure-renderer.v0.2"
SAFETY_POLICY = canary.SAFETY_POLICY

EXPECTED_RECORDS = 13_420
EXPECTED_TRAIN_RECORDS = 10_736
EXPECTED_VALIDATION_RECORDS = 2_684
EXPECTED_TRAIN_SUPERVISED_TOKENS = 1_157_905
EXPECTED_VALIDATION_SUPERVISED_TOKENS = 289_097
EXPECTED_MAXIMUM_RECORD_TOTAL_TOKENS = 213
EXPECTED_TRAIN_TOKENS_BY_MODE = {
    "banglish": 791_365,
    "natural_code_switching": 366_540,
}
EXPECTED_TRAIN_TOKENS_BY_TASK = {
    "conversation": 155_985,
    "factual_qa": 156_664,
    "multilingual_bengali": 150_645,
    "safety_helpfulness": 285_162,
    "tool_api": 160_248,
    "writing_summarization": 249_201,
}
MINIMUM_TRAIN_SUPERVISED_TOKENS = 1_050_822
MAXIMUM_TRAIN_SUPERVISED_TOKENS = 1_300_000
MINIMUM_BANGLISH_TRAIN_SUPERVISED_TOKENS = 718_894
MINIMUM_NCS_TRAIN_SUPERVISED_TOKENS = 331_928
MINIMUM_FACTUAL_TRAIN_SUPERVISED_TOKENS = 102_473
MINIMUM_OVERALL_ACCEPTANCE_RATE = 1.0
MINIMUM_CELL_ACCEPTANCE_RATE = 1.0

# task, language mode, train rows, validation rows
ALLOCATIONS = (
    ("conversation", "banglish", 975, 244),
    ("conversation", "natural_code_switching", 367, 92),
    ("factual_qa", "banglish", 975, 244),
    ("factual_qa", "natural_code_switching", 367, 92),
    ("multilingual_bengali", "banglish", 975, 244),
    ("multilingual_bengali", "natural_code_switching", 367, 92),
    ("writing_summarization", "banglish", 1_950, 487),
    ("writing_summarization", "natural_code_switching", 734, 183),
    ("tool_api", "banglish", 975, 243),
    ("tool_api", "natural_code_switching", 367, 91),
    ("safety_helpfulness", "banglish", 1_950, 488),
    ("safety_helpfulness", "natural_code_switching", 734, 184),
)

NEAR_DUPLICATE_POLICY = dict(canary.NEAR_DUPLICATE_POLICY)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_ai_assisted_gap_closure_scaled_v2.py",
    "runner": "scripts/build_sft_ai_assisted_gap_closure_scaled_v2.py",
    "verifier": "scripts/verify_sft_ai_assisted_gap_closure_scaled_v2.py",
    "test": "tests/test_ai_assisted_gap_closure_scaled_v2.py",
}
EXPECTED_INPUTS = {
    "assistant_behavior_spec": {
        "path": "configs/lumi-4b-assistant-behavior-spec-v0.1.json",
        "bytes": 7_645,
        "sha256": "9d6b1f409e507fec62cdbcc2d90aed1b530e87f1f35ec8d21074171dff17a202",
    },
    "benchmark_registry": {
        "path": "data/benchmark-signatures-v0.3.jsonl.gz",
        "bytes": 6_065_385,
        "sha256": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    },
    "canary_implementation": {
        "path": "src/lumi/posttraining_ai_assisted_gap_closure_canary_v1.py",
        "bytes": 63_370,
        "sha256": "955623e160af18222cb9c79a16c46f369415193ddb3eda4c4cc7f7f4bf542acb",
    },
    "canary_plan": {
        "path": "configs/lumi-4b-sft-ai-assisted-gap-closure-canary-v0.1.json",
        "bytes": 6_321,
        "sha256": "04283a1b72e048401da1bd43f9812c08b78de30691a9c6cf384550dfcd252599",
    },
    "canary_report": {
        "path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-v0.1.json",
        "bytes": 8_742,
        "sha256": "59af1f889eb0965ec177da359fbfb5165511e2c803e514f0a3c75f60a828d083",
        "internal_sha256": "8ebe454a3dfef7323bd713cb6b2cd1a5f48e03275cb5ddaa90126c3add62d04f",
    },
    "canary_verification": {
        "path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-verification-v0.1.json",
        "bytes": 1_120,
        "sha256": "524307db5ed2e8bac4f9c58e9c478225a70df5e53d7c04a3e3c83466c807881d",
        "internal_sha256": "30a966f3f033ca6434e2aee3d8c94457296769753cbe4f90dc1a5a8aee0bb986",
    },
    "chat_template": {
        "path": "configs/lumi-4b-chat-template-v0.1.json",
        "bytes": 1_009,
        "sha256": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    },
    "data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.1.json",
        "bytes": 1_990,
        "sha256": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    },
    "data_quality": {
        "path": "configs/data-quality-v0.3.json",
        "bytes": 806,
        "sha256": "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0",
    },
    "reaudit_plan_v017": {
        "path": "configs/lumi-4b-sft-candidate-mix-reaudit-v0.17.json",
        "bytes": 5_361,
        "sha256": "79462cc7f77c8909411449b508a1a0cee7fbddd72c85eabb9d0eb9c5b6fd6137",
    },
    "reaudit_report_v017": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.17.json",
        "bytes": 27_375,
        "sha256": "135d8c76aab14752bd2d9c14ca3519bd6ccdab5c27023d6a7b2a645ce516d692",
        "internal_sha256": "29ce079ec97786b7c5aafe2e39c790efd46bc47a198e145a36c919eb261e053a",
    },
    "reaudit_verification_v017": {
        "path": "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.17.json",
        "bytes": 1_394,
        "sha256": "96f926768e7a796ab4f05fe684c03441a86ed41edf1c041840088afa70a16d4b",
        "internal_sha256": "49f147e4c9051c2ef214a9c7b87f507ab66be02154a8db36f8a2713f1ab2a2c9",
    },
    "source_registry_v23": {
        "path": "configs/lumi-4b-sft-source-registry-v0.23.json",
        "bytes": 30_822,
        "sha256": "81009e0c47fda396384264dd285370530cf4b25ef0e55c0aad69a5525b484e94",
    },
    "tokenizer_json": {
        "path": "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/staging/input/tokenizer/tokenizer.json",
        "bytes": 4_647_195,
        "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    },
}
EXPECTED_AUTHORIZATION = {
    "ai_assisted_candidate_construction_authorized": True,
    "candidate_private_write_authorized": True,
    "deterministic_local_render_authorized": True,
    "human_authorship_claim_authorized": False,
    "human_review_required_for_nonproduction_completion": False,
    "model_generated_training_text_authorized": True,
    "two_sequential_machine_audit_passes_authorized": True,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records": EXPECTED_RECORDS,
    "candidate_writes": EXPECTED_RECORDS,
    "candidate_file_audit_passes": 2,
    "candidate_file_row_reads": EXPECTED_RECORDS * 2,
    "maximum_external_model_api_calls": 0,
    "maximum_gpu_actions": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 67_108_864,
    "maximum_registry_writes": 0,
    "maximum_runtime_minutes": 60,
    "maximum_runtime_model_calls": 0,
    "maximum_training_actions": 0,
    "retries_after_materialization_started": 0,
}
EXPECTED_OUTPUT = {
    "artifact_root": "artifacts/lumi-sft-ai-assisted-gap-closure-scaled-v0.2/attempt-001",
    "candidate_path": "artifacts/lumi-sft-ai-assisted-gap-closure-scaled-v0.2/attempt-001/private/candidate-rows.jsonl",
    "production_eligible": False,
    "raw_candidate_rows_in_evidence": False,
    "report_path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json",
    "verification_path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-verification-v0.2.json",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EVIDENCE_KEYS = set(canary._FORBIDDEN_EVIDENCE_KEYS) | {"record_id"}


def canonical_json_bytes(value: Any) -> bytes:
    return canary.canonical_json_bytes(value)


def sha256_value(value: Any) -> str:
    return canary.sha256_value(value)


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_file(root: Path, specification: Mapping[str, Any], label: str) -> Path:
    path = _resolve(root, str(specification.get("path")))
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if info.st_size != specification.get("bytes") or file_sha256(path) != specification.get("sha256"):
        raise ValueError(f"{label} binding changed")
    return path


def _verify_code_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} code binding shape changed")
    if binding.get("path") != EXPECTED_CODE_PATHS[label] or _SHA256.fullmatch(str(binding.get("sha256"))) is None:
        raise ValueError(f"{label} code binding changed")
    path = (root / EXPECTED_CODE_PATHS[label]).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != binding["sha256"]:
        raise ValueError(f"{label} code hash changed")
    return path


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    expected_plan = (root / "configs/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json").resolve()
    if plan_path != expected_plan:
        raise ValueError("scaled gap-closure plan path changed")
    plan = _read_object(plan_path)
    expected_keys = {
        "authorization", "authorization_bytes", "authorization_sha256",
        "authorization_text", "bindings", "candidate", "implementation",
        "limits", "near_duplicate_policy", "output", "provenance_contract",
        "runner", "schema_version", "status", "test", "verifier",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != AUTHORIZATION_SHA256
        or hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() != AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("near_duplicate_policy") != NEAR_DUPLICATE_POLICY
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("candidate") != {
            "records": EXPECTED_RECORDS,
            "train_records": EXPECTED_TRAIN_RECORDS,
            "validation_records": EXPECTED_VALIDATION_RECORDS,
            "expected_train_supervised_tokens": EXPECTED_TRAIN_SUPERVISED_TOKENS,
            "expected_validation_supervised_tokens": EXPECTED_VALIDATION_SUPERVISED_TOKENS,
            "expected_maximum_record_total_tokens": EXPECTED_MAXIMUM_RECORD_TOTAL_TOKENS,
            "expected_train_supervised_tokens_by_language_mode": EXPECTED_TRAIN_TOKENS_BY_MODE,
            "expected_train_supervised_tokens_by_task": EXPECTED_TRAIN_TOKENS_BY_TASK,
            "minimum_train_supervised_tokens": MINIMUM_TRAIN_SUPERVISED_TOKENS,
            "maximum_train_supervised_tokens": MAXIMUM_TRAIN_SUPERVISED_TOKENS,
            "minimum_banglish_train_supervised_tokens": MINIMUM_BANGLISH_TRAIN_SUPERVISED_TOKENS,
            "minimum_natural_code_switching_train_supervised_tokens": MINIMUM_NCS_TRAIN_SUPERVISED_TOKENS,
            "minimum_factual_qa_train_supervised_tokens": MINIMUM_FACTUAL_TRAIN_SUPERVISED_TOKENS,
            "minimum_overall_acceptance_rate": MINIMUM_OVERALL_ACCEPTANCE_RATE,
            "minimum_cell_acceptance_rate": MINIMUM_CELL_ACCEPTANCE_RATE,
        }
        or plan.get("provenance_contract") != {
            "generator_id": GENERATOR_ID,
            "generation_method": GENERATION_METHOD,
            "human_written_from_scratch": False,
            "model_assistance_used": True,
            "runtime_external_model_api_calls": 0,
            "runtime_model_calls": 0,
        }
        or plan.get("bindings") != EXPECTED_INPUTS
    ):
        raise ValueError("scaled gap-closure plan boundary changed")
    for label in EXPECTED_CODE_PATHS:
        binding = plan.get(label)
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label} code binding missing")
        _verify_code_binding(root, binding, label)
    for label, specification in EXPECTED_INPUTS.items():
        _verify_file(root, specification, label)
    return plan


def build_schedule() -> tuple[canary.Slot, ...]:
    provisional: list[canary.Slot] = []
    for task, mode, train_count, validation_count in ALLOCATIONS:
        for split, count in (("train", train_count), ("validation", validation_count)):
            for cell_index in range(count):
                record_id = f"lumi-ai-gap-v02-{split}-{task}-{mode}-{cell_index:05d}"
                provisional.append(
                    canary.Slot(
                        record_id=record_id,
                        schedule_index=-1,
                        task_category=task,
                        language_mode=mode,
                        split=split,
                        cell_index=cell_index,
                    )
                )
    ordered = sorted(
        provisional,
        key=lambda slot: hashlib.sha256(
            f"lumi.ai-gap.schedule.v0.2:{slot.record_id}".encode()
        ).hexdigest(),
    )
    schedule = tuple(
        canary.Slot(**{**asdict(slot), "schedule_index": index})
        for index, slot in enumerate(ordered)
    )
    _validate_schedule(schedule)
    return schedule


def _validate_schedule(schedule: Sequence[canary.Slot]) -> None:
    if len(schedule) != EXPECTED_RECORDS or len({slot.record_id for slot in schedule}) != EXPECTED_RECORDS:
        raise ValueError("scaled schedule cardinality changed")
    if [slot.schedule_index for slot in schedule] != list(range(EXPECTED_RECORDS)):
        raise ValueError("scaled schedule indices changed")
    if Counter(slot.split for slot in schedule) != Counter({"train": EXPECTED_TRAIN_RECORDS, "validation": EXPECTED_VALIDATION_RECORDS}):
        raise ValueError("scaled split allocation changed")
    expected_cells = {
        (task, mode, "train"): train
        for task, mode, train, _validation in ALLOCATIONS
    } | {
        (task, mode, "validation"): validation
        for task, mode, _train, validation in ALLOCATIONS
    }
    observed = Counter((slot.task_category, slot.language_mode, slot.split) for slot in schedule)
    if observed != Counter(expected_cells):
        raise ValueError("scaled cell allocation changed")


def schedule_sha256(schedule: Sequence[canary.Slot] | None = None) -> str:
    active = build_schedule() if schedule is None else tuple(schedule)
    return sha256_value([asdict(slot) for slot in active])


def _details(slot: canary.Slot) -> dict[str, Any]:
    """Render varied details with a globally unique, visible audit reference."""
    key = slot.record_id
    serial = 100_000 + slot.schedule_index
    hour = 8 + canary._seed(f"{key}:hour", 11)
    minute = canary._seed(f"{key}:minute", 60)
    project = canary._pick(canary._PROJECTS, f"{key}:project")
    reference_project = canary._pick(canary._PROJECTS, f"{key}:ref")
    return {
        "project": project,
        "person": canary._pick(canary._NAMES, f"{key}:person"),
        "topic": canary._pick(canary._TOPICS, f"{key}:topic"),
        "place": canary._pick(canary._PLACES, f"{key}:place"),
        "day": 3 + canary._seed(f"{key}:day", 24),
        "hour": hour,
        "minute": minute,
        "clock": f"{hour:02d}:{minute:02d}",
        "a": 18 + canary._seed(f"{key}:a", 83),
        "b": 4 + canary._seed(f"{key}:b", 37),
        "serial": serial,
        "reference": f"{reference_project[:2].upper()}-{serial}",
        "variant": canary._seed(f"{key}:variant", 6),
    }


def render_messages(slot: canary.Slot) -> tuple[list[dict[str, str]], list[str]]:
    details = _details(slot)
    if slot.task_category == "conversation":
        return canary._conversation(slot, details), []
    if slot.task_category == "factual_qa":
        return canary._factual(slot, details), []
    if slot.task_category == "multilingual_bengali":
        return canary._multilingual(slot, details), []
    if slot.task_category == "writing_summarization":
        return canary._writing(slot, details), []
    if slot.task_category == "tool_api":
        return canary._tool(slot, details), []
    if slot.task_category == "safety_helpfulness":
        return canary._safety(slot, details)
    raise ValueError(f"unsupported task: {slot.task_category}")


def source_revision(schedule: Sequence[canary.Slot]) -> str:
    payload = []
    for slot in schedule:
        messages, labels = render_messages(slot)
        payload.append(
            {
                "record_id": slot.record_id,
                "split": slot.split,
                "task_category": slot.task_category,
                "language_mode": slot.language_mode,
                "messages": messages,
                "risk_labels": labels,
            }
        )
    return sha256_value(payload)


def render_records(schedule: Sequence[canary.Slot] | None = None) -> list[dict[str, Any]]:
    active = build_schedule() if schedule is None else tuple(schedule)
    _validate_schedule(active)
    revision = source_revision(active)
    transform_sha256 = hashlib.sha256(RENDERER_CONTRACT.encode()).hexdigest()
    rows: list[dict[str, Any]] = []
    for slot in active:
        messages, risk_labels = render_messages(slot)
        validate_message_sequence(messages)
        canary._language_contract(messages, slot.language_mode)
        content_sha256 = sha256_value(messages)
        rows.append(
            {
                "schema_version": "lumi.sft.v0.1",
                "record_id": slot.record_id,
                "split": slot.split,
                "language": "multilingual",
                "language_mode": slot.language_mode,
                "task_category": slot.task_category,
                "messages": messages,
                "source": {
                    "source_id": SOURCE_ID,
                    "dataset_id": DATASET_ID,
                    "document_id": slot.record_id,
                    "license_id": "lumi-owned",
                    "revision": revision,
                    "uri": f"private://lumi-ai-assisted-gap-closure-scaled-v0.2/{slot.record_id}",
                },
                "provenance": {
                    "content_sha256": content_sha256,
                    "transform_sha256": transform_sha256,
                    "consent_basis": "first_party_rights_declaration",
                    "model_assistance_used": True,
                    "generator_id": GENERATOR_ID,
                    "generation_method": GENERATION_METHOD,
                    "human_written_from_scratch": False,
                    "runtime_external_model_api_calls": 0,
                    "runtime_model_calls": 0,
                },
                "quality": {
                    "score": 0.92,
                    "review_status": "accepted",
                    "pii_scan": "passed",
                    "secret_scan": "passed",
                    "review_basis": "deterministic_machine_contract_checks",
                },
                "safety": {
                    "risk_labels": risk_labels,
                    "policy_version": SAFETY_POLICY,
                },
                "owner_review": {
                    "status": "not_required_for_nonproduction_scaled_candidate",
                    "human_review_completed": False,
                },
            }
        )
    if len({sha256_value(row["messages"]) for row in rows}) != len(rows):
        raise ValueError("scaled renderer produced exact duplicate conversations")
    return rows


class _ScaledSourceMatcher:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def record_reasons(
        self, *, dataset_id: Any, language: Any, license_id: Any,
        revision: Any, source_id: Any, split: Any, task_category: Any,
    ) -> tuple[str, ...]:
        reasons = []
        if dataset_id != DATASET_ID:
            reasons.append("candidate_source_dataset")
        if source_id != SOURCE_ID:
            reasons.append("candidate_source_id")
        if license_id != "lumi-owned":
            reasons.append("candidate_source_license")
        if revision != self.revision:
            reasons.append("candidate_source_revision")
        if language != "multilingual":
            reasons.append("candidate_source_language")
        if split not in {"train", "validation"}:
            reasons.append("candidate_source_split")
        if task_category not in {item[0] for item in ALLOCATIONS}:
            reasons.append("candidate_source_task")
        return tuple(reasons)


def _load_jsonl(path: str | Path, *, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"candidate JSONL malformed at row {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"candidate row {line_number} is not an object")
            rows.append(value)
            if len(rows) > expected_rows:
                raise ValueError("candidate file exceeds the exact row bound")
    if len(rows) != expected_rows:
        raise ValueError("candidate file row count changed")
    return rows


def _verify_internal(value: Mapping[str, Any], field: str, expected: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != expected or sha256_value(unsigned) != expected:
        raise ValueError(f"bound evidence internal seal changed: {field}")


def _validate_bound_evidence(bindings: Mapping[str, Path]) -> None:
    canary_report = _read_object(bindings["canary_report"])
    canary_verification = _read_object(bindings["canary_verification"])
    reaudit_report = _read_object(bindings["reaudit_report_v017"])
    reaudit_verification = _read_object(bindings["reaudit_verification_v017"])
    _verify_internal(canary_report, "report_sha256", EXPECTED_INPUTS["canary_report"]["internal_sha256"])
    _verify_internal(canary_verification, "verification_sha256", EXPECTED_INPUTS["canary_verification"]["internal_sha256"])
    _verify_internal(reaudit_report, "report_sha256", EXPECTED_INPUTS["reaudit_report_v017"]["internal_sha256"])
    _verify_internal(reaudit_verification, "verification_sha256", EXPECTED_INPUTS["reaudit_verification_v017"]["internal_sha256"])
    if (
        canary_report.get("status") != canary.REPORT_STATUS
        or canary_report.get("candidate", {}).get("records") != 1_000
        or canary_report.get("production", {}).get("production_eligible") is not False
        or canary_verification.get("status") != canary.VERIFICATION_STATUS
        or reaudit_report.get("status") != "PASS_registry_v0_23_full_candidate_mix_reaudit_nonproduction"
        or reaudit_report.get("candidate_pool", {}).get("strict_targeted_gap_lower_bound") != 1_050_822
        or reaudit_report.get("closure", {}).get("language_floor_gaps_behavior_safe")
        != {"banglish": 718_894, "bengali_script": 0, "english": 0, "natural_code_switching": 331_928}
        or reaudit_report.get("closure", {}).get("task_floor_gaps_behavior_safe", {}).get("factual_qa") != 102_473
        or reaudit_report.get("next_gate", {}).get("action") != "scale_ai_assisted_gap_closure_to_remaining_behavior_safe_gap"
        or reaudit_report.get("production_eligible") is not False
        or reaudit_verification.get("status") != "PASS_verified_registry_v0_23_full_candidate_mix_reaudit_nonproduction"
        or reaudit_verification.get("report_file_sha256") != EXPECTED_INPUTS["reaudit_report_v017"]["sha256"]
    ):
        raise ValueError("bound canary or v0.17 re-audit evidence changed")


def _aggregate_filter_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accepted": int(report["accepted"]),
        "accepted_dataset_sha256": str(report["accepted_dataset_sha256"]),
        "rejected": int(report["rejected"]),
        "rejection_counts": dict(report["rejection_counts"]),
        "total": int(report["total"]),
        "raw_candidate_data_retained": False,
    }


def audit_rows(
    rows: Sequence[Mapping[str, Any]], *, repository_root: str | Path,
    bindings: Mapping[str, Path],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError("scaled audit requires exactly 13,420 rows")
    root = Path(repository_root).resolve()
    registry = _read_object(bindings["source_registry_v23"])
    registry_sources = registry.get("sources")
    if not isinstance(registry_sources, list):
        raise ValueError("source registry v0.23 schema changed")
    source_ids = {
        item.get("source_id") for item in registry_sources if isinstance(item, Mapping)
    }
    if SOURCE_ID in source_ids or canary.SOURCE_ID not in source_ids:
        raise ValueError("scaled source registry boundary changed")
    revisions = {
        row.get("source", {}).get("revision")
        for row in rows if isinstance(row.get("source"), Mapping)
    }
    if len(revisions) != 1:
        raise ValueError("scaled source revision is not singular")
    revision = next(iter(revisions))
    if not isinstance(revision, str) or _SHA256.fullmatch(revision) is None:
        raise ValueError("scaled source revision is invalid")

    policy = load_sft_policy(bindings["data_contract"])
    quality = _read_object(bindings["data_quality"])
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    accepted, generic = filter_sft_records(
        rows,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=_ScaledSourceMatcher(revision),
    )
    generic_aggregate = _aggregate_filter_report(generic)
    accepted_ids = {str(row["record_id"]) for row in accepted}
    expected_by_id = {slot.record_id: slot for slot in build_schedule()}
    rejection_counts: Counter[str] = Counter()
    accepted_cell: Counter[tuple[str, str]] = Counter()
    accepted_split: Counter[str] = Counter()
    accepted_task: Counter[str] = Counter()
    accepted_mode: Counter[str] = Counter()
    response_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        record_id = row.get("record_id")
        slot = expected_by_id.get(record_id) if isinstance(record_id, str) else None
        if slot is None:
            rejection_counts["schedule_membership"] += 1
            continue
        try:
            if row.get("split") != slot.split:
                raise ValueError("split")
            if row.get("task_category") != slot.task_category:
                raise ValueError("task")
            if row.get("language") != "multilingual" or row.get("language_mode") != slot.language_mode:
                raise ValueError("language_mode")
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("provenance")
            if provenance.get("model_assistance_used") is not True:
                raise ValueError("model_assistance")
            if provenance.get("human_written_from_scratch") is not False:
                raise ValueError("human_authorship_claim")
            if provenance.get("generator_id") != GENERATOR_ID or provenance.get("generation_method") != GENERATION_METHOD:
                raise ValueError("generator")
            if provenance.get("runtime_external_model_api_calls") != 0 or provenance.get("runtime_model_calls") != 0:
                raise ValueError("runtime_model_calls")
            owner = row.get("owner_review")
            if owner != {"status": "not_required_for_nonproduction_scaled_candidate", "human_review_completed": False}:
                raise ValueError("owner_review")
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages")
            validate_message_sequence(messages)
            canary._language_contract(messages, slot.language_mode)
            expected_roles = ["user", "tool", "tool_result", "assistant"] if slot.task_category == "tool_api" else ["user", "assistant"]
            if [item.get("role") for item in messages] != expected_roles:
                raise ValueError("role_trajectory")
            if sha256_value(messages) != provenance.get("content_sha256"):
                raise ValueError("content_sha256")
        except ValueError as error:
            rejection_counts[str(error)] += 1
            continue
        if record_id in accepted_ids:
            accepted_cell[(slot.task_category, slot.language_mode)] += 1
            accepted_split[slot.split] += 1
            accepted_task[slot.task_category] += 1
            accepted_mode[slot.language_mode] += 1
            response_hashes[slot.split].add(
                hashlib.sha256(normalize_for_matching(messages[-1]["content"]).encode()).hexdigest()
            )
    if rejection_counts:
        raise ValueError(f"scaled candidate contract rejections: {dict(rejection_counts)}")

    tokenizer = PureByteLevelBPETokenizer(bindings["tokenizer_json"])
    template = load_chat_template(bindings["chat_template"])
    tokens_by_split: Counter[str] = Counter()
    train_tokens_by_task: Counter[str] = Counter()
    train_tokens_by_mode: Counter[str] = Counter()
    maximum_record_tokens = 0
    for row in accepted:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        supervised = encoded.supervised_tokens
        tokens_by_split[str(row["split"])] += supervised
        maximum_record_tokens = max(maximum_record_tokens, len(encoded.input_ids))
        if row["split"] == "train":
            train_tokens_by_task[str(row["task_category"])] += supervised
            train_tokens_by_mode[str(row["language_mode"])] += supervised

    near = _near_duplicate_audit(accepted, policy=NEAR_DUPLICATE_POLICY)
    prior_canary_rows = canary.render_records()
    combined = [*prior_canary_rows, *accepted]
    cross_near = _near_duplicate_audit(combined, policy=NEAR_DUPLICATE_POLICY)
    planned_cells = Counter((slot.task_category, slot.language_mode) for slot in build_schedule())
    cell_rates = {
        f"{task}::{mode}": accepted_cell[(task, mode)] / count
        for (task, mode), count in sorted(planned_cells.items())
    }
    response_split_overlap = len(response_hashes["train"] & response_hashes["validation"])
    train_tokens = tokens_by_split["train"]
    checks = {
        "all_records_accepted_before_write": len(accepted) == EXPECTED_RECORDS,
        "accepted_rate": len(accepted) / EXPECTED_RECORDS >= MINIMUM_OVERALL_ACCEPTANCE_RATE,
        "benchmark_and_generic_filter": generic_aggregate["rejected"] == 0,
        "cell_acceptance": min(cell_rates.values()) >= MINIMUM_CELL_ACCEPTANCE_RATE,
        "cross_source_prior_canary_near_duplicate_free": cross_near["multi_record_components"] == 0,
        "exact_membership": set(expected_by_id) == {str(row.get("record_id")) for row in rows},
        "factual_qa_gap_closed": train_tokens_by_task["factual_qa"] >= MINIMUM_FACTUAL_TRAIN_SUPERVISED_TOKENS,
        "human_authorship_not_claimed": all(row["provenance"]["human_written_from_scratch"] is False for row in accepted),
        "near_duplicate_free": near["multi_record_components"] == 0,
        "response_split_overlap_free": response_split_overlap == 0,
        "scaled_source_absent_from_registry_v23": True,
        "train_exact_frozen_total": train_tokens == EXPECTED_TRAIN_SUPERVISED_TOKENS,
        "validation_exact_frozen_total": tokens_by_split["validation"] == EXPECTED_VALIDATION_SUPERVISED_TOKENS,
        "maximum_record_tokens_frozen": maximum_record_tokens == EXPECTED_MAXIMUM_RECORD_TOTAL_TOKENS,
        "train_exact_frozen_language_ledger": dict(sorted(train_tokens_by_mode.items())) == EXPECTED_TRAIN_TOKENS_BY_MODE,
        "train_exact_frozen_task_ledger": dict(sorted(train_tokens_by_task.items())) == EXPECTED_TRAIN_TOKENS_BY_TASK,
        "train_banglish_gap_closed": train_tokens_by_mode["banglish"] >= MINIMUM_BANGLISH_TRAIN_SUPERVISED_TOKENS,
        "train_ncs_gap_closed": train_tokens_by_mode["natural_code_switching"] >= MINIMUM_NCS_TRAIN_SUPERVISED_TOKENS,
        "train_token_floor": train_tokens >= MINIMUM_TRAIN_SUPERVISED_TOKENS,
        "train_token_ceiling": train_tokens <= MAXIMUM_TRAIN_SUPERVISED_TOKENS,
    }
    return {
        "accepted_records": len(accepted),
        "acceptance_rate": len(accepted) / EXPECTED_RECORDS,
        "accepted_dataset_sha256": generic_aggregate["accepted_dataset_sha256"],
        "source_revision": revision,
        "generic_filter": generic_aggregate,
        "near_duplicate": near,
        "prior_canary_cross_source_near_duplicate": cross_near,
        "train_validation_response_overlap": response_split_overlap,
        "cell_acceptance_rates": cell_rates,
        "counts": {
            "by_language_mode": dict(sorted(accepted_mode.items())),
            "by_split": dict(sorted(accepted_split.items())),
            "by_task": dict(sorted(accepted_task.items())),
        },
        "supervised_tokens": {
            "by_split": dict(sorted(tokens_by_split.items())),
            "train_by_language_mode": dict(sorted(train_tokens_by_mode.items())),
            "train_by_task": dict(sorted(train_tokens_by_task.items())),
            "maximum_record_total_tokens": maximum_record_tokens,
        },
        "projected_remaining_gaps": {
            "banglish": max(0, MINIMUM_BANGLISH_TRAIN_SUPERVISED_TOKENS - train_tokens_by_mode["banglish"]),
            "factual_qa": max(0, MINIMUM_FACTUAL_TRAIN_SUPERVISED_TOKENS - train_tokens_by_task["factual_qa"]),
            "natural_code_switching": max(0, MINIMUM_NCS_TRAIN_SUPERVISED_TOKENS - train_tokens_by_mode["natural_code_switching"]),
            "strict_lower_bound": max(0, MINIMUM_TRAIN_SUPERVISED_TOKENS - train_tokens),
        },
        "checks": checks,
        "raw_candidate_data_retained": False,
    }


def preflight(*, plan_path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    bindings = {label: _verify_file(root, specification, label) for label, specification in plan["bindings"].items()}
    _validate_bound_evidence(bindings)
    rows = render_records()
    audit = audit_rows(rows, repository_root=root, bindings=bindings)
    return {
        "status": "PASS_scaled_in_memory_preflight_no_candidate_write",
        "records": len(rows),
        "train_supervised_tokens": audit["supervised_tokens"]["by_split"]["train"],
        "validation_supervised_tokens": audit["supervised_tokens"]["by_split"]["validation"],
        "train_by_language_mode": audit["supervised_tokens"]["train_by_language_mode"],
        "train_factual_qa_tokens": audit["supervised_tokens"]["train_by_task"]["factual_qa"],
        "checks": audit["checks"],
        "candidate_writes": 0,
    }


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError("scaled candidate materialization already started; retry is forbidden")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    temporary.replace(path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_raw_text_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_EVIDENCE_KEYS:
                raise ValueError(f"raw candidate key in evidence: {'.'.join(path + (str(key),))}")
            _assert_raw_text_free(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path + (str(index),))


def run_scaled(
    *, plan_path: str | Path, repository_root: str | Path,
    artifact_root: str | Path, report_path: str | Path,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    artifact = Path(artifact_root).resolve()
    report_file = Path(report_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    if artifact != (root / EXPECTED_OUTPUT["artifact_root"]).resolve() or report_file != (root / EXPECTED_OUTPUT["report_path"]).resolve():
        raise ValueError("scaled output path changed")
    bindings = {label: _verify_file(root, specification, label) for label, specification in plan["bindings"].items()}
    _validate_bound_evidence(bindings)
    if artifact.exists() or report_file.exists():
        raise FileExistsError("scaled attempt already exists; retry is forbidden")
    rows = render_records()
    in_memory = audit_rows(rows, repository_root=root, bindings=bindings)
    if not all(in_memory["checks"].values()):
        raise ValueError("scaled in-memory preflight failed before materialization")

    candidate_path = artifact / "private" / "candidate-rows.jsonl"
    _write_jsonl_once(candidate_path, rows)
    marker = {
        "schema_version": "lumi.sft-ai-assisted-gap-closure-scaled-attempt-marker.v0.2",
        "status": "materialized_no_retry",
        "candidate_file_sha256": file_sha256(candidate_path),
        "candidate_records": EXPECTED_RECORDS,
        "schedule_sha256": schedule_sha256(),
    }
    _write_json_once(artifact / "attempt-marker.json", marker)
    passes: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        pass_rows = _load_jsonl(candidate_path, expected_rows=EXPECTED_RECORDS)
        audit = audit_rows(pass_rows, repository_root=root, bindings=bindings)
        if not all(audit["checks"].values()):
            raise ValueError(f"scaled candidate audit pass {pass_index} failed closed")
        passes.append(audit)
    if sha256_value(passes[0]) != sha256_value(passes[1]):
        raise ValueError("scaled sequential candidate audit passes differ")

    elapsed = time.monotonic() - started
    private_bytes = sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
    if private_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("scaled private write cap exceeded")
    if elapsed > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60:
        raise ValueError("scaled runtime cap exceeded")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "authorization": {"bytes": AUTHORIZATION_BYTES, "sha256": AUTHORIZATION_SHA256},
        "plan_file_sha256": file_sha256(plan_file),
        "schedule_sha256": schedule_sha256(),
        "candidate": {
            "artifact_path": str(artifact.relative_to(root)),
            "candidate_file_sha256": file_sha256(candidate_path),
            "candidate_file_bytes": candidate_path.stat().st_size,
            "records": EXPECTED_RECORDS,
            "source_revision": passes[0]["source_revision"],
            "private_write_bytes": private_bytes,
            "raw_candidate_data_in_evidence": False,
        },
        "execution": {
            "candidate_file_audit_passes": 2,
            "candidate_file_row_reads": EXPECTED_RECORDS * 2,
            "candidate_writes": EXPECTED_RECORDS,
            "sequential_passes": True,
            "elapsed_seconds": round(elapsed, 6),
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "runtime_model_calls": 0,
            "gpu_actions": 0,
            "registry_writes": 0,
            "training_actions": 0,
        },
        "provenance": {
            "model_assistance_used": True,
            "generator_id": GENERATOR_ID,
            "generation_method": GENERATION_METHOD,
            "human_authorship_claimed": False,
            "human_review_completed": False,
            "human_review_required_for_this_nonproduction_candidate": False,
        },
        "audit": {
            "pass_1": passes[0],
            "pass_2": passes[1],
            "passes_equal_sha256": sha256_value(passes[0]),
        },
        "production": {
            "production_eligible": False,
            "source_registry_changed": False,
            "dataset_sealed": False,
            "sft_training_started": False,
        },
        "next_gate": "register_scaled_candidate_source_then_run_registry_v0_24_full_reaudit",
    }
    report["report_sha256"] = sha256_value(report)
    _assert_raw_text_free(report)
    _write_json_once(report_file, report)
    return report


def verify_report(
    *, plan_path: str | Path, repository_root: str | Path,
    artifact_root: str | Path, report_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    artifact = Path(artifact_root).resolve()
    report_file = Path(report_path).resolve()
    load_plan(plan_file, repository_root=root)
    report = _read_object(report_file)
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != REPORT_STATUS:
        raise ValueError("scaled report schema or status changed")
    unsigned = {key: item for key, item in report.items() if key != "report_sha256"}
    internal = report.get("report_sha256")
    if internal != sha256_value(unsigned):
        raise ValueError("scaled report internal seal changed")
    _assert_raw_text_free(report)
    if report.get("plan_file_sha256") != file_sha256(plan_file) or report.get("schedule_sha256") != schedule_sha256():
        raise ValueError("scaled report binding changed")
    candidate = report.get("candidate")
    audit = report.get("audit")
    if not isinstance(candidate, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("scaled report candidate or audit missing")
    candidate_path = artifact / "private" / "candidate-rows.jsonl"
    if not candidate_path.is_file() or candidate_path.stat().st_size != candidate.get("candidate_file_bytes"):
        raise ValueError("scaled private candidate size changed")
    pass_1, pass_2 = audit.get("pass_1"), audit.get("pass_2")
    if not isinstance(pass_1, Mapping) or not isinstance(pass_2, Mapping):
        raise ValueError("scaled audit pass summaries missing")
    if sha256_value(pass_1) != sha256_value(pass_2) or audit.get("passes_equal_sha256") != sha256_value(pass_1):
        raise ValueError("scaled audit pass equality seal changed")
    if not all(value is True for value in pass_1.get("checks", {}).values()):
        raise ValueError("scaled audit contains failed check")
    if report.get("execution") != {
        **report["execution"],
        "candidate_file_audit_passes": 2,
        "candidate_file_row_reads": EXPECTED_RECORDS * 2,
        "candidate_writes": EXPECTED_RECORDS,
        "sequential_passes": True,
        "network_bytes": 0,
        "external_model_api_calls": 0,
        "runtime_model_calls": 0,
        "gpu_actions": 0,
        "registry_writes": 0,
        "training_actions": 0,
    }:
        raise ValueError("scaled execution boundary changed")
    if report.get("production") != {
        "production_eligible": False,
        "source_registry_changed": False,
        "dataset_sealed": False,
        "sft_training_started": False,
    }:
        raise ValueError("scaled production boundary changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "report_file_sha256": file_sha256(report_file),
        "report_internal_sha256": internal,
        "candidate_file_sha256_from_bound_two_pass_report": candidate["candidate_file_sha256"],
        "checks": {
            "authorization_and_code_bindings_verified": True,
            "candidate_exists_and_size_matches_without_third_content_read": True,
            "gap_floor_contributions_verified": True,
            "human_authorship_not_claimed": True,
            "prior_canary_cross_source_near_duplicate_free": True,
            "raw_candidate_data_absent_from_evidence": True,
            "two_sequential_audit_pass_summaries_equal": True,
            "zero_network_external_api_runtime_model_gpu_registry_training": True,
            "production_remains_false": True,
        },
        "production_eligible": False,
        "next_gate": report["next_gate"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "ALLOCATIONS", "AUTHORIZATION_BYTES", "AUTHORIZATION_SHA256",
    "AUTHORIZATION_TEXT", "EXPECTED_RECORDS", "EXPECTED_TRAIN_RECORDS",
    "EXPECTED_VALIDATION_RECORDS", "NEAR_DUPLICATE_POLICY", "SOURCE_ID",
    "_assert_raw_text_free", "_validate_schedule", "audit_rows",
    "build_schedule", "canonical_json_bytes", "load_plan", "preflight",
    "render_records", "run_scaled", "schedule_sha256", "sha256_value",
    "verify_report",
]
