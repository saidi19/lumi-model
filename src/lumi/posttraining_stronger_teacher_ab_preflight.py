"""Fail-closed preflight for a rights-safe local stronger-teacher paired A/B."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    file_sha256,
    sha256_value as canonical_sha256_value,
)
from .posttraining_first_party_scale_canary import (
    LANGUAGE_MODES,
    TASKS,
    Slot,
    build_schedule,
    sha256_value as schedule_sha256_value,
)

PLAN_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-stronger-teacher-ab-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-stronger-teacher-ab-preflight-verification.v0.1"
)

BASELINE_MODEL = {
    "file_bytes": 5_027_783_488,
    "file_name": "Qwen3-8B-Q4_K_M.gguf",
    "file_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    "model_id": "Qwen/Qwen3-8B-GGUF",
    "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
}
CANDIDATE_MODEL = {
    "file_bytes": 9_001_752_960,
    "file_name": "Qwen3-14B-Q4_K_M.gguf",
    "file_sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
    "license_bytes": 11_544,
    "license_sha256": "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd",
    "model_card_bytes": 7_493,
    "model_card_sha256": "45989a28a5ba6525cbbce906d8bc165ca612d8856a68ca27f568ce62056405a3",
    "model_id": "Qwen/Qwen3-14B-GGUF",
    "revision": "c75e7b2d0234068f674a1bacf548ea32e27ccd29",
    "weights_license": "apache-2.0",
}
REMOTE_OBJECT_BYTES = (
    CANDIDATE_MODEL["file_bytes"]
    + CANDIDATE_MODEL["license_bytes"]
    + CANDIDATE_MODEL["model_card_bytes"]
)
EXPECTED_REMOTE_OBJECTS = [
    {
        "bytes": CANDIDATE_MODEL["file_bytes"],
        "path": CANDIDATE_MODEL["file_name"],
        "sha256": CANDIDATE_MODEL["file_sha256"],
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            f"{CANDIDATE_MODEL['revision']}/{CANDIDATE_MODEL['file_name']}"
        ),
    },
    {
        "bytes": CANDIDATE_MODEL["license_bytes"],
        "path": "LICENSE",
        "sha256": CANDIDATE_MODEL["license_sha256"],
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            f"{CANDIDATE_MODEL['revision']}/LICENSE"
        ),
    },
    {
        "bytes": CANDIDATE_MODEL["model_card_bytes"],
        "path": "README.md",
        "sha256": CANDIDATE_MODEL["model_card_sha256"],
        "url": (
            "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
            f"{CANDIDATE_MODEL['revision']}/README.md"
        ),
    },
]

BASE_SCHEDULE_SHA256 = (
    "7c6251fc99159f698424a7d2162a8d0a56389935b0d7d6627186b7cd7fe3a66e"
)
SELECTED_SCHEDULE_SHA256 = (
    "326236a562bcf7a79ec3f76e544bb1ad35e34cd870b5b29c046a89a0dc21397a"
)
SELECTED_IDS_SHA256 = (
    "42d765fc44b8bcd5624002a3c833eb2b72a3d849416c7bb522dcdd0f5255355d"
)

EXPECTED_AUTHORIZATION = {
    "candidate_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "metadata_network_access_authorized": True,
    "model_download_authorized": False,
    "package_installation_authorized": False,
    "paid_or_cloud_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "rights_safe_stronger_teacher_paired_ab_preflight_only",
    "raw_text_free_evidence_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records_generated": 0,
    "external_api_calls": 0,
    "local_inference_calls": 0,
    "maximum_metadata_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 120,
    "metadata_objects_downloaded": 2,
    "metadata_payload_bytes_downloaded": 19_037,
    "model_download_bytes": 0,
}
EXPECTED_OUTPUT = {
    "production_eligible": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-stronger-teacher-ab-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-stronger-teacher-ab-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "base_schedule_source",
        "benchmark_signatures",
        "chat_template",
        "closure_plan",
        "closure_report",
        "closure_verification",
        "data_quality",
        "failed_scale_report",
        "failed_scale_verification",
        "source_registry",
    }
)

TOOLING_PATHS = {
    "implementation": "src/lumi/posttraining_stronger_teacher_ab_preflight.py",
    "runner": "scripts/check_stronger_teacher_ab_preflight.py",
    "test": "tests/test_stronger_teacher_ab_preflight.py",
    "verifier": "scripts/verify_stronger_teacher_ab_preflight.py",
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return canonical_sha256_value(unsigned)


def _resolve_binding(
    root: Path, binding: Mapping[str, Any], *, name: str
) -> Path:
    raw = binding.get("path")
    if not isinstance(raw, str):
        raise ValueError(f"stronger-teacher binding path missing: {name}")
    path = Path(raw).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"stronger-teacher binding missing: {name}")
    if file_sha256(resolved) != binding.get("sha256"):
        raise ValueError(f"stronger-teacher binding changed: {name}")
    return resolved


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("stronger-teacher binding set changed")
    root = Path(repository_root).resolve()
    return {
        name: _resolve_binding(root, binding, name=name)
        for name, binding in bindings.items()
    }


def selected_schedule() -> tuple[Slot, ...]:
    base = build_schedule()
    selected: list[Slot] = []
    for task in TASKS:
        for language in LANGUAGE_MODES:
            train = sorted(
                (
                    slot
                    for slot in base
                    if slot.task == task
                    and slot.language_mode == language
                    and slot.split == "train"
                ),
                key=lambda slot: slot.cell_index,
            )
            validation = sorted(
                (
                    slot
                    for slot in base
                    if slot.task == task
                    and slot.language_mode == language
                    and slot.split == "validation"
                ),
                key=lambda slot: slot.cell_index,
            )
            selected.extend(train[:4])
            selected.extend(validation[:1])
    result = tuple(sorted(selected, key=lambda slot: slot.record_id))
    _validate_selected_schedule(result)
    return result


def _validate_selected_schedule(schedule: Sequence[Slot]) -> None:
    if len(schedule) != 100 or len({slot.record_id for slot in schedule}) != 100:
        raise ValueError("paired A/B schedule is not exactly 100 unique prompts")
    cells = Counter((slot.task, slot.language_mode) for slot in schedule)
    split_cells = Counter(
        (slot.task, slot.language_mode, slot.split) for slot in schedule
    )
    for task in TASKS:
        for language in LANGUAGE_MODES:
            if cells[(task, language)] != 5:
                raise ValueError("paired A/B task-language cell is not exactly five")
            if split_cells[(task, language, "train")] != 4:
                raise ValueError("paired A/B cell train count is not four")
            if split_cells[(task, language, "validation")] != 1:
                raise ValueError("paired A/B cell validation count is not one")
    if Counter(slot.split for slot in schedule) != Counter(
        {"train": 80, "validation": 20}
    ):
        raise ValueError("paired A/B split is not 80/20")
    schedule_sha = schedule_sha256_value([asdict(slot) for slot in schedule])
    ids_sha = schedule_sha256_value([slot.record_id for slot in schedule])
    if schedule_sha != SELECTED_SCHEDULE_SHA256 or ids_sha != SELECTED_IDS_SHA256:
        raise ValueError("paired A/B selected schedule identity changed")


def _verify_predecessors(resolved: Mapping[str, Path]) -> dict[str, Any]:
    failed = _load_json(resolved["failed_scale_report"])
    acceptance = failed.get("acceptance", {})
    if (
        failed.get("status") != "stopped_fail_closed_quality_screen_failed"
        or acceptance.get("cumulative_calls") != 50
        or acceptance.get("cumulative_accepted") != 30
        or acceptance.get("amended_screen_calls") != 20
        or acceptance.get("amended_screen_accepted") != 9
        or failed.get("call_budget", {}).get("calls_unspent") != 950
    ):
        raise ValueError("failed Qwen3-8B scale-canary evidence changed")
    failed_verification = _load_json(resolved["failed_scale_verification"])
    if (
        failed_verification.get("status") != "PASS"
        or failed_verification.get("quality_stop_report_file_sha256")
        != file_sha256(resolved["failed_scale_report"])
        or not all(failed_verification.get("independent_checks", {}).values())
    ):
        raise ValueError("failed scale-canary verification is not clean")
    closure = _load_json(resolved["closure_report"])
    gate = closure.get("next_gate", {})
    if (
        gate.get("action") != "provider_bound_first_party_scale_canary"
        or gate.get("maximum_candidate_records") != 1_000
        or gate.get("target_source_id")
        != "lumi_first_party_multilingual_safety_tool_v0_3"
    ):
        raise ValueError("10M closure next gate changed")
    closure_verification = _load_json(resolved["closure_verification"])
    if not all(closure_verification.get("checks", {}).values()):
        raise ValueError("10M closure verification is not clean")
    return {
        "amended_screen_acceptance_rate": acceptance["amended_screen_rate"],
        "calls_used": failed["call_budget"]["calls_used"],
        "cumulative_acceptance_rate": acceptance["cumulative_rate"],
        "failed_model_file_sha256": failed["bindings"]["model_file_sha256"],
        "quality_stop_report_file_sha256": file_sha256(
            resolved["failed_scale_report"]
        ),
        "quality_stop_verification_file_sha256": file_sha256(
            resolved["failed_scale_verification"]
        ),
    }


def _verify_contract(contract: Mapping[str, Any]) -> None:
    expected = {
        "candidate_minimum_advantage_over_baseline": 0.15,
        "candidate_minimum_machine_acceptance_rate": 0.90,
        "candidate_minimum_owner_acceptance_rate": 0.90,
        "candidate_minimum_owner_preference_rate": 0.70,
        "candidate_minimum_per_cell_acceptance_rate": 0.80,
        "critical_factual_safety_or_tool_failures": 0,
        "expected_final_outputs": 200,
        "maximum_context_tokens": 4096,
        "maximum_generated_output_tokens": 56_320,
        "maximum_inference_attempts": 220,
        "maximum_response_tokens": 256,
        "maximum_retained_candidate_supervised_tokens": 30_000,
        "owner_review_paired_rows": 40,
        "paired_prompts": 100,
        "prompts_per_task_language_cell": 5,
        "train_prompts_per_cell": 4,
        "validation_prompts_per_cell": 1,
    }
    if dict(contract) != expected:
        raise ValueError("stronger-teacher paired A/B contract changed")


def load_stronger_teacher_ab_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-09",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_metadata_only_stronger_teacher_ab_preflight",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"stronger-teacher preflight plan changed: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    _verify_predecessors(resolved)
    _verify_contract(plan.get("ab_contract", {}))
    schedule = plan.get("paired_prompt_selection", {})
    if (
        schedule.get("base_schedule_internal_sha256") != BASE_SCHEDULE_SHA256
        or schedule.get("selected_schedule_internal_sha256")
        != SELECTED_SCHEDULE_SHA256
        or schedule.get("selected_record_ids_sha256") != SELECTED_IDS_SHA256
    ):
        raise ValueError("stronger-teacher schedule binding changed")
    selected_schedule()
    if plan.get("baseline_model") != BASELINE_MODEL:
        raise ValueError("stronger-teacher baseline model changed")
    if plan.get("candidate_model") != CANDIDATE_MODEL:
        raise ValueError("stronger-teacher candidate model changed")
    if plan.get("candidate_remote_objects") != EXPECTED_REMOTE_OBJECTS:
        raise ValueError("stronger-teacher candidate remote objects changed")
    if sum(row["bytes"] for row in EXPECTED_REMOTE_OBJECTS) != REMOTE_OBJECT_BYTES:
        raise ValueError("stronger-teacher remote-object byte arithmetic changed")
    if plan.get("next_gate", {}).get("maximum_network_bytes") != REMOTE_OBJECT_BYTES:
        raise ValueError("stronger-teacher next-gate network bound changed")
    return plan


def _tooling_hashes(repository_root: str | Path) -> dict[str, str]:
    root = Path(repository_root).resolve()
    return {name: file_sha256(root / path) for name, path in TOOLING_PATHS.items()}


def build_stronger_teacher_ab_preflight_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    predecessor = _verify_predecessors(resolved)
    selected = selected_schedule()
    report: dict[str, Any] = {
        "ab_contract": dict(plan["ab_contract"]),
        "authorization": dict(plan["authorization"]),
        "candidate_model": dict(CANDIDATE_MODEL),
        "candidate_remote_objects": list(EXPECTED_REMOTE_OBJECTS),
        "local_capacity": dict(plan["local_capacity"]),
        "network_and_compute_accounting": {
            "candidate_records_generated": 0,
            "cloud_or_paid_gpu_used": False,
            "external_api_calls": 0,
            "local_inference_calls": 0,
            "metadata_objects_downloaded": 2,
            "metadata_payload_bytes_downloaded": 19_037,
            "model_download_bytes": 0,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "paired_prompt_selection": {
            "base_schedule_internal_sha256": BASE_SCHEDULE_SHA256,
            "selection_algorithm": plan["paired_prompt_selection"][
                "selection_algorithm"
            ],
            "selected_prompt_count": len(selected),
            "selected_record_ids_sha256": SELECTED_IDS_SHA256,
            "selected_schedule_internal_sha256": SELECTED_SCHEDULE_SHA256,
            "split_counts": {"train": 80, "validation": 20},
            "task_language_cells": 20,
        },
        "plan_file_sha256": plan_file_sha256,
        "predecessor_quality_stop": predecessor,
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "provider_decisions": dict(plan["provider_decisions"]),
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "qwen3_14b_exact_objects_not_downloaded_or_hash_verified_locally",
            "paired_ab_execution_package_not_materialized",
            "paired_ab_model_inference_not_explicitly_authorized",
            "candidate_machine_quality_and_owner_review_not_measured",
            "one_thousand_row_candidate_only_scale_canary_not_run",
            "three_point_two_million_token_generation_not_authorized",
            "registry_dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "verified_rights_safe_qwen14b_paired_ab_preflight_no_model_download_"
            "or_inference"
        ),
        "tooling_file_sha256": _tooling_hashes(repository_root),
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_stronger_teacher_ab_preflight_report(report)
    return report


def validate_stronger_teacher_ab_preflight_report(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("stronger-teacher report schema changed")
    if report.get("report_sha256") != _internal_sha(report, "report_sha256"):
        raise ValueError("stronger-teacher report SHA-256 changed")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("stronger-teacher preflight retained raw prompt text")
    accounting = report.get("network_and_compute_accounting", {})
    if (
        accounting.get("candidate_records_generated") != 0
        or accounting.get("external_api_calls") != 0
        or accounting.get("local_inference_calls") != 0
        or accounting.get("model_download_bytes") != 0
        or accounting.get("cloud_or_paid_gpu_used") is not False
    ):
        raise ValueError("stronger-teacher preflight performed a prohibited action")
    if any(report.get("production", {}).values()):
        raise ValueError("stronger-teacher preflight changed production state")
    _verify_contract(report.get("ab_contract", {}))
    if report.get("candidate_model") != CANDIDATE_MODEL:
        raise ValueError("stronger-teacher report candidate model changed")
    if report.get("candidate_remote_objects") != EXPECTED_REMOTE_OBJECTS:
        raise ValueError("stronger-teacher report remote objects changed")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"user_prompt"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("stronger-teacher evidence contains raw training text")


def build_stronger_teacher_ab_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_stronger_teacher_ab_preflight_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "baseline_failure_and_independent_verification_bound": True,
            "candidate_apache_two_license_and_exact_remote_objects_bound": True,
            "candidate_model_download_and_inference_not_performed": True,
            "exact_one_hundred_prompt_paired_schedule_reproduced": True,
            "external_api_and_paid_gpu_use_remain_prohibited": True,
            "openai_and_azure_teacher_lanes_fail_closed_under_observed_terms": True,
            "quality_advantage_cell_safety_tool_and_owner_gates_bound": True,
            "registry_dataset_seal_bulk_generation_and_sft_not_authorized": True,
            "sequential_single_model_memory_safety_contract_bound": True,
            "tooling_and_predecessor_file_hashes_bound": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonexecuting_rights_safe_stronger_teacher_ab_preflight",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
