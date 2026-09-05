"""Rights-aware, no-weight-download preflight for a local first-party SFT teacher."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-first-party-teacher-canary-preflight-verification.v0.1"
)

MODEL_REPOSITORY = "https://huggingface.co/Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
MODEL_FILE = "Qwen3-8B-Q4_K_M.gguf"
MODEL_FILE_BYTES = 5_027_783_488
MODEL_FILE_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
MODEL_POINTER_BYTES = 135
MODEL_POINTER_SHA256 = "bf9310b066270482a5ca9f738f70099358046e86c76c1a827c87e0d787db45bb"
MODEL_README_BYTES = 7_577
MODEL_README_SHA256 = "913fd918b4afb4511accca0874f701248b024013e804fa7a463639ead74e4dcb"
MODEL_LICENSE_BYTES = 11_544
MODEL_LICENSE_SHA256 = "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd"

EXPECTED_CLOSURE_PLAN_FILE_SHA256 = (
    "672e1faf1e558af7fc6c39afccda3dbe0eafeca77e0b6e6324215a8c2f3bb0c7"
)
EXPECTED_CLOSURE_REPORT_FILE_SHA256 = (
    "3013e3d09ada6dec55911606f5586b59a972027906fb273457afd867e9fad071"
)
EXPECTED_CLOSURE_VERIFICATION_FILE_SHA256 = (
    "a190c0a7edbfae169b5d6245126919d423159beeba14e193f3c2429a35639148"
)
EXPECTED_DATA_SCALE_POLICY_FILE_SHA256 = (
    "f55a4c62d26b666141e32d4255759dd1752af16986fe3ebe44623cb898d91b0d"
)

TASKS = (
    "factual_qa",
    "multilingual_bengali",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
)
LANGUAGE_MODES = (
    "banglish",
    "bengali_script",
    "english",
    "natural_code_switching",
)

EXPECTED_AUTHORIZATION = {
    "candidate_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "local_model_execution_authorized": False,
    "model_weight_download_authorized": False,
    "official_metadata_lookup_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_teacher_selection_and_exact_download_canary_preflight_only",
    "raw_text_free_evidence_authorized": True,
    "unrelated_data_access_authorized": False,
}

EXPECTED_LIMITS = {
    "generated_candidate_records": 0,
    "maximum_metadata_network_bytes": 1_048_576,
    "maximum_runtime_minutes": 60,
    "model_weight_bytes_downloaded": 0,
}

EXPECTED_OUTPUT = {
    "production_eligible": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-first-party-teacher-canary-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-first-party-teacher-canary-preflight-verification-v0.1.json"
    ),
}

EXPECTED_BINDING_PATHS = {
    "ai_assisted_pilot": "configs/lumi-4b-ai-assisted-sft-pilot-v0.3.json",
    "benchmark_signatures": "data/benchmark-signatures-v0.3.jsonl.gz",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "closure_plan": "configs/lumi-4b-sft-10m-mix-closure-source-selection-v0.2.json",
    "closure_report": "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.2.json",
    "closure_verification": (
        "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.2.json"
    ),
    "data_quality": "configs/data-quality-v0.3.json",
    "data_scale_policy": "docs/sft-data-scale-gate-v0.1.md",
    "implementation": "src/lumi/posttraining_first_party_teacher_preflight.py",
    "runner": "scripts/check_first_party_teacher_preflight.py",
    "safety_tool_pilot": "configs/lumi-4b-safety-tool-sft-pilot-v0.1.json",
    "test": "tests/test_first_party_teacher_preflight.py",
    "verifier": "scripts/verify_first_party_teacher_preflight.py",
}

MODEL_METADATA = {
    "declared_license_id": "apache-2.0",
    "license": {
        "path": "LICENSE",
        "sha256": MODEL_LICENSE_SHA256,
        "size_bytes": MODEL_LICENSE_BYTES,
    },
    "model_card": {
        "path": "README.md",
        "sha256": MODEL_README_SHA256,
        "size_bytes": MODEL_README_BYTES,
    },
    "model_file": {
        "path": MODEL_FILE,
        "sha256": MODEL_FILE_SHA256,
        "size_bytes": MODEL_FILE_BYTES,
    },
    "model_id": "Qwen/Qwen3-8B-GGUF",
    "pointer": {
        "path": MODEL_FILE,
        "sha256": MODEL_POINTER_SHA256,
        "size_bytes": MODEL_POINTER_BYTES,
    },
    "repository_url": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
}

MODEL_CARD_CLAIMS = {
    "bengali_quality_guaranteed": False,
    "context_tokens_native": 32_768,
    "local_ollama_usage_documented": True,
    "multilingual_claim": "100_plus_languages_and_dialects",
    "parameter_count": 8_200_000_000,
    "q4_k_m_available": True,
    "tool_agent_capability_claimed": True,
}

LOCAL_SNAPSHOT = {
    "available_disk_gib_observed": 553,
    "cached_model_present": False,
    "chip": "Apple M5",
    "memory_bytes": 16 * 1024**3,
    "ollama_client_version": "0.32.5",
    "ollama_installed": True,
}

CANARY_CONTRACT = {
    "candidate_records": 80,
    "generation": {
        "max_new_tokens": 384,
        "parallelism": 1,
        "presence_penalty": 1.5,
        "seed_derivation": "sha256_first_u32_of_record_id",
        "temperature": 0.7,
        "thinking_mode": False,
        "top_k": 20,
        "top_p": 0.8,
    },
    "language_modes": list(LANGUAGE_MODES),
    "maximum_context_tokens": 4_096,
    "maximum_retained_supervised_tokens": 24_000,
    "owner_review_records": 32,
    "records_per_task_language_cell": 4,
    "required_checks": [
        "truthful_per_record_self_hosted_ai_assisted_provenance",
        "provider_model_revision_prompt_template_and_generation_parameter_binding",
        "no_human_authorship_claim",
        "no_hidden_chain_of_thought_or_think_tag_retention",
        "factual_grounding_and_deterministic_tool_result_verification",
        "safety_policy_and_overrefusal_checks",
        "native_bengali_banglish_code_switch_and_punctuation_review",
        "assistant_and_tool_only_loss_masking_and_chat_template_validation",
        "semantic_train_validation_isolation_exact_and_near_deduplication",
        "benchmark_decontamination_four_k_fit_and_frozen_token_measurement",
        "memory_pressure_thermal_stability_and_sequential_runtime_check",
    ],
    "split": {
        "train_records": 60,
        "validation_records": 20,
    },
    "task_categories": list(TASKS),
}

NEXT_AUTHORIZATION_STATEMENT = (
    "I approve one bounded Mac-local Qwen3-8B-GGUF first-party teacher canary at "
    f"revision {MODEL_REVISION}, using only {MODEL_FILE} "
    f"({MODEL_FILE_BYTES} bytes; SHA-256 {MODEL_FILE_SHA256}), the pinned README and "
    "Apache-2.0 LICENSE, maximum 5,100,000,000 network bytes / 12 GiB private disk / "
    "180 minutes / 80 generated candidate records / 24,000 retained supervised tokens. "
    "It may verify all hashes, use the already-installed Ollama 0.32.5 locally with "
    "sequential on-device CPU/Metal execution and no remote inference API, generate "
    "exactly 60 train and 20 validation candidates across the five planned tasks and "
    "four language modes, and write private non-production rows, a deterministic "
    "32-row owner-review packet, and raw-text-free evidence. No cloud or paid GPU, "
    "OpenAI/Azure/other external API, package installation, candidate-registry change, "
    "dataset seal, SFT training, unrelated data, external distribution, or human "
    "authorship claim. Stop fail-closed on checksum mismatch, memory pressure, thermal "
    "instability, hidden chain-of-thought, unsafe or ungrounded output, Bengali quality "
    "failure, benchmark contamination, or split leakage."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or relative != EXPECTED_BINDING_PATHS[name]:
        raise ValueError(f"first-party teacher binding path changed: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"first-party teacher binding missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"first-party teacher binding changed: {name}")
    return resolved


def load_first_party_teacher_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "canary_contract": CANARY_CONTRACT,
        "limits": EXPECTED_LIMITS,
        "local_snapshot": LOCAL_SNAPSHOT,
        "model_card_claims": MODEL_CARD_CLAIMS,
        "model_metadata": MODEL_METADATA,
        "next_gate": {
            "authorization_required": True,
            "exact_authorization_statement": NEXT_AUTHORIZATION_STATEMENT,
        },
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "verified_metadata_only_local_teacher_preflight_pending_download_canary",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"first-party teacher preflight plan changed: {field}")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("first-party teacher binding set changed")
    resolved = {
        name: _resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    fixed = {
        "closure_plan": EXPECTED_CLOSURE_PLAN_FILE_SHA256,
        "closure_report": EXPECTED_CLOSURE_REPORT_FILE_SHA256,
        "closure_verification": EXPECTED_CLOSURE_VERIFICATION_FILE_SHA256,
        "data_scale_policy": EXPECTED_DATA_SCALE_POLICY_FILE_SHA256,
    }
    for name, expected_sha in fixed.items():
        if bindings[name]["sha256"] != expected_sha:
            raise ValueError(f"fixed first-party teacher binding changed: {name}")

    closure = _load_json(resolved["closure_plan"])
    next_gate = closure.get("next_gate", {})
    if next_gate.get("target_source_id") != "lumi_first_party_multilingual_safety_tool_v0_3":
        raise ValueError("closure target source changed")
    if next_gate.get("maximum_candidate_records") != 1_000:
        raise ValueError("closure canary record ceiling changed")
    if next_gate.get("maximum_retained_train_supervised_tokens") != 120_000:
        raise ValueError("closure canary token ceiling changed")

    policy = resolved["data_scale_policy"].read_text(encoding="utf-8")
    for phrase in (
        "both direct OpenAI API and Azure OpenAI generation are fail-closed",
        "Synthetic output must never be relabeled as human",
    ):
        if phrase not in policy:
            raise ValueError("data-scale provider-rights policy changed")
    return plan


def _provider_decisions() -> list[dict[str, Any]]:
    return [
        {
            "decision": "ineligible_fail_closed_without_written_contract_specific_clearance",
            "provider": "openai_direct_api",
            "remote_generation_service": True,
            "training_rows_generated": 0,
        },
        {
            "decision": "ineligible_fail_closed_without_written_contract_specific_clearance",
            "provider": "azure_openai",
            "remote_generation_service": True,
            "training_rows_generated": 0,
        },
        {
            "decision": "provisional_private_local_canary_only",
            "output_rights_warranty_claimed": False,
            "provider": "self_hosted_qwen3_8b_gguf",
            "remote_generation_service": False,
            "third_party_reproduction_filter_required": True,
            "training_rows_generated": 0,
            "weights_license": "apache-2.0",
        },
    ]


def _coverage() -> dict[str, Any]:
    cells = len(TASKS) * len(LANGUAGE_MODES)
    records = cells * CANARY_CONTRACT["records_per_task_language_cell"]
    return {
        "cross_product_cells": cells,
        "language_modes": len(LANGUAGE_MODES),
        "records": records,
        "records_per_cell": CANARY_CONTRACT["records_per_task_language_cell"],
        "task_categories": len(TASKS),
        "train_records": CANARY_CONTRACT["split"]["train_records"],
        "validation_records": CANARY_CONTRACT["split"]["validation_records"],
    }


def build_first_party_teacher_preflight_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    working_set_ratio = round(MODEL_FILE_BYTES / LOCAL_SNAPSHOT["memory_bytes"], 6)
    report: dict[str, Any] = {
        "authorization": {
            "candidate_records_generated": 0,
            "external_api_calls": 0,
            "model_weight_bytes_downloaded": 0,
            "production_eligible": False,
        },
        "canary": {
            "contract": CANARY_CONTRACT,
            "coverage": _coverage(),
            "status": "not_started_separate_exact_authorization_required",
        },
        "fit_assessment": {
            "maximum_context_tokens": CANARY_CONTRACT["maximum_context_tokens"],
            "model_weight_to_system_memory_ratio": working_set_ratio,
            "parallelism": 1,
            "status": "provisional_fit_requires_empirical_memory_pressure_and_thermal_canary",
        },
        "model": {
            "card_claims": MODEL_CARD_CLAIMS,
            "metadata": MODEL_METADATA,
            "terms_receipt": {
                "apache_license_applies_to_model_work": True,
                "external_service_terms_apply": False,
                "output_noninfringement_warranty_claimed": False,
                "private_research_canary_only_until_quality_and_rights_review": True,
                "self_hosted_generation": True,
            },
        },
        "next_gate": plan["next_gate"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "provider_decisions": _provider_decisions(),
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass_metadata_only_teacher_selection_no_rows_generated",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_first_party_teacher_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("first-party teacher report schema changed")
    unsigned = dict(report)
    observed_sha = unsigned.pop("report_sha256", None)
    if observed_sha != sha256_value(unsigned):
        raise ValueError("first-party teacher report SHA-256 changed")
    authorization = report.get("authorization", {})
    if authorization != {
        "candidate_records_generated": 0,
        "external_api_calls": 0,
        "model_weight_bytes_downloaded": 0,
        "production_eligible": False,
    }:
        raise ValueError("first-party teacher preflight performed unauthorized work")
    if report.get("model", {}).get("metadata") != MODEL_METADATA:
        raise ValueError("first-party teacher model metadata changed")
    coverage = report.get("canary", {}).get("coverage", {})
    if coverage != _coverage() or coverage.get("records") != 80:
        raise ValueError("first-party teacher canary coverage changed")
    providers = {
        row.get("provider"): row
        for row in report.get("provider_decisions", [])
        if isinstance(row, Mapping)
    }
    if set(providers) != {
        "azure_openai",
        "openai_direct_api",
        "self_hosted_qwen3_8b_gguf",
    }:
        raise ValueError("first-party teacher provider decision set changed")
    if any(row.get("training_rows_generated") != 0 for row in providers.values()):
        raise ValueError("provider preflight generated training rows")


def build_first_party_teacher_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_first_party_teacher_preflight_report(report)
    _require_sha256(report_file_sha256, "report_file_sha256")
    verification: dict[str, Any] = {
        "checks": {
            "bengali_quality_not_prejudged": True,
            "canary_cross_product_records": 80,
            "candidate_records_generated": 0,
            "external_api_calls": 0,
            "model_file_exact_sha256_bound": True,
            "model_weight_bytes_downloaded": 0,
            "openai_and_azure_fail_closed": True,
            "owner_review_required": True,
            "provider_terms_receipt_bound": True,
            "separate_exact_download_canary_authorization_required": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_metadata_only_teacher_preflight_no_rows_generated",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification

