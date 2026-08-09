"""Fail-closed provider binding for the Lumi first-party 1,000-row scale canary."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    _load_json,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-first-party-scale-canary-preflight-verification.v0.1"
)

MODEL_ID = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
MODEL_FILE_SHA256 = (
    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
)
MODEL_FILE_BYTES = 5_027_783_488
OLLAMA_MODEL = "lumi-qwen3-8b-teacher-canary:7c41481-q4km"
OLLAMA_MANIFEST_PATH = Path(
    "/Users/afnan/.ollama/models/manifests/registry.ollama.ai/library/"
    "lumi-qwen3-8b-teacher-canary/7c41481-q4km"
)
OLLAMA_MANIFEST_SHA256 = (
    "25af7ed9af299f2e55dda9efc922ecb61186c1c7a88024b11e1207c49ff268ef"
)
OLLAMA_BINARY = Path("/usr/local/bin/ollama")
OLLAMA_VERSION = "0.32.5"

TASK_ALLOCATION = {
    "factual_qa": 47,
    "multilingual_bengali": 162,
    "safety_helpfulness": 266,
    "tool_api": 266,
    "writing_summarization": 259,
}
LANGUAGE_ALLOCATION = {
    "banglish": 266,
    "bengali_script": 172,
    "english": 422,
    "natural_code_switching": 140,
}
CELL_ALLOCATION = {
    "factual_qa": {
        "banglish": 12,
        "bengali_script": 8,
        "english": 20,
        "natural_code_switching": 7,
    },
    "multilingual_bengali": {
        "banglish": 43,
        "bengali_script": 28,
        "english": 68,
        "natural_code_switching": 23,
    },
    "safety_helpfulness": {
        "banglish": 71,
        "bengali_script": 46,
        "english": 112,
        "natural_code_switching": 37,
    },
    "tool_api": {
        "banglish": 71,
        "bengali_script": 46,
        "english": 112,
        "natural_code_switching": 37,
    },
    "writing_summarization": {
        "banglish": 69,
        "bengali_script": 44,
        "english": 110,
        "natural_code_switching": 36,
    },
}
TRAIN_CELL_ALLOCATION = {
    "factual_qa": {
        "banglish": 10,
        "bengali_script": 6,
        "english": 16,
        "natural_code_switching": 6,
    },
    "multilingual_bengali": {
        "banglish": 34,
        "bengali_script": 22,
        "english": 54,
        "natural_code_switching": 18,
    },
    "safety_helpfulness": {
        "banglish": 57,
        "bengali_script": 37,
        "english": 90,
        "natural_code_switching": 30,
    },
    "tool_api": {
        "banglish": 57,
        "bengali_script": 37,
        "english": 90,
        "natural_code_switching": 29,
    },
    "writing_summarization": {
        "banglish": 55,
        "bengali_script": 35,
        "english": 88,
        "natural_code_switching": 29,
    },
}

EXPECTED_AUTHORIZATION = {
    "candidate_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "local_ollama_generation_authorized": False,
    "model_download_authorized": False,
    "network_access_authorized": False,
    "package_installation_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_provider_binding_and_scale_canary_preflight_only",
    "raw_text_free_evidence_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records_generated": 0,
    "external_api_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "model_download_bytes": 0,
}
EXPECTED_OUTPUT = {
    "production_eligible": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-first-party-scale-canary-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-first-party-scale-canary-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_signatures",
        "chat_template",
        "closure_plan",
        "closure_report",
        "closure_verification",
        "data_quality",
        "data_scale_policy",
        "implementation",
        "model_card",
        "model_file",
        "model_license",
        "runner",
        "source_registry",
        "teacher_preflight_report",
        "teacher_preflight_verification",
        "teacher_promotion_receipt",
        "teacher_promotion_verification",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
    }
)
EXPECTED_FIXED_SHA256 = {
    "benchmark_signatures": (
        "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
    ),
    "chat_template": (
        "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12"
    ),
    "closure_plan": (
        "d95b61d68a2f3491deae537f8370c17665490ae71c50c893f4470d081cee0fad"
    ),
    "closure_report": (
        "629b1115feda72fe9c32a38eb18bb5b868b885c0e20ad801406a5e38c90f66f1"
    ),
    "closure_verification": (
        "9933a0e5900709ed81900de7b040a93685252dccc856d3c24ead0cbe372ed4b0"
    ),
    "data_quality": (
        "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0"
    ),
    "data_scale_policy": (
        "d87e59a8f38174c2b405372dafd9ec1e38a52af5a207e837f84e36996455cbc7"
    ),
    "model_card": (
        "913fd918b4afb4511accca0874f701248b024013e804fa7a463639ead74e4dcb"
    ),
    "model_file": MODEL_FILE_SHA256,
    "model_license": (
        "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd"
    ),
    "source_registry": (
        "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
    ),
    "teacher_preflight_report": (
        "caeb9e95ed1717ce6f753681c15012946314edb374ec045330dd8edc4fa59e8d"
    ),
    "teacher_preflight_verification": (
        "5b923a0689e53b436045a8c940548ff893146091d8d98e0f3b506a3cdbf5a164"
    ),
    "teacher_promotion_receipt": (
        "1683a001256346afa56501aba5d6d13af0cf84dfa5d1963a2d17c71e568edb5e"
    ),
    "teacher_promotion_verification": (
        "e23cdc9592cc208537a98f63895d3910af67abfa91160409fe7489b200490a2a"
    ),
    "tokenizer_config": (
        "d31c91cf95389f72e3b5edcb71b67129e259e4f36149fb435e0a3cc15437d0a5"
    ),
    "tokenizer_json": (
        "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
    ),
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_value(unsigned)


def _sum_nested(value: Mapping[str, Mapping[str, int]]) -> int:
    return sum(sum(row.values()) for row in value.values())


def _column_totals(value: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {
        language: sum(row[language] for row in value.values())
        for language in LANGUAGE_ALLOCATION
    }


def _row_totals(value: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {task: sum(row.values()) for task, row in value.items()}


def _validation_cells() -> dict[str, dict[str, int]]:
    return {
        task: {
            language: CELL_ALLOCATION[task][language]
            - TRAIN_CELL_ALLOCATION[task][language]
            for language in LANGUAGE_ALLOCATION
        }
        for task in TASK_ALLOCATION
    }


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("scale-canary preflight binding set changed")
    return {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _verify_runtime() -> dict[str, Any]:
    if not OLLAMA_BINARY.is_file():
        raise ValueError("Ollama client is missing")
    if not OLLAMA_MANIFEST_PATH.is_file():
        raise ValueError("bound Ollama model manifest is missing")
    if file_sha256(OLLAMA_MANIFEST_PATH) != OLLAMA_MANIFEST_SHA256:
        raise ValueError("bound Ollama model manifest changed")
    manifest = _load_json(OLLAMA_MANIFEST_PATH)
    model_layers = [
        row
        for row in manifest.get("layers", [])
        if row.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if model_layers != [
        {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{MODEL_FILE_SHA256}", "size": MODEL_FILE_BYTES}
    ]:
        raise ValueError("Ollama manifest model layer changed")
    observed = subprocess.run(
        [str(OLLAMA_BINARY), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_text = f"{observed.stdout}\n{observed.stderr}"
    if OLLAMA_VERSION not in version_text:
        raise ValueError("Ollama client version changed")
    return {
        "binary_path": str(OLLAMA_BINARY),
        "client_version": OLLAMA_VERSION,
        "daemon_started_by_preflight": False,
        "manifest_file_sha256": OLLAMA_MANIFEST_SHA256,
        "manifest_path": str(OLLAMA_MANIFEST_PATH),
        "model_alias": OLLAMA_MODEL,
    }


def _verify_provider_policy(resolved: Mapping[str, Path]) -> dict[str, Any]:
    policy = _load_json(resolved["data_scale_policy"])
    providers = policy.get("provider_output_policy", {})
    openai = providers.get("openai_api", {})
    azure = providers.get("azure_openai", {})
    if openai.get("status") != "blocked_for_lumi_training_under_standard_services_agreement":
        raise ValueError("direct OpenAI provider policy changed")
    if azure.get("status") != "blocked_pending_contract_specific_clearance":
        raise ValueError("Azure OpenAI provider policy changed")

    teacher_preflight = _load_json(resolved["teacher_preflight_report"])
    decisions = {
        row.get("provider"): row
        for row in teacher_preflight.get("provider_decisions", [])
        if isinstance(row, Mapping)
    }
    local = decisions.get("self_hosted_qwen3_8b_gguf", {})
    terms = teacher_preflight.get("model", {}).get("terms_receipt", {})
    if (
        local.get("decision") != "provisional_private_local_canary_only"
        or local.get("weights_license") != "apache-2.0"
        or terms.get("self_hosted_generation") is not True
        or terms.get("external_service_terms_apply") is not False
    ):
        raise ValueError("self-hosted Qwen provider terms receipt changed")
    return {
        "azure_openai": {
            "decision": "ineligible_fail_closed_pending_written_contract_specific_clearance",
            "external_api_calls": 0,
        },
        "openai_direct_api": {
            "decision": "ineligible_fail_closed_pending_written_contract_exception",
            "external_api_calls": 0,
        },
        "self_hosted_qwen3_8b_gguf": {
            "decision": "selected_for_bounded_private_scale_canary",
            "external_api_calls": 0,
            "external_service_terms_apply": False,
            "weights_license": "apache-2.0",
        },
    }


def _verify_predecessors(resolved: Mapping[str, Path]) -> None:
    closure = _load_json(resolved["closure_plan"])
    gate = closure.get("next_gate", {})
    if (
        gate.get("action") != "provider_bound_first_party_scale_canary"
        or gate.get("maximum_candidate_records") != 1_000
        or gate.get("maximum_retained_train_supervised_tokens") != 120_000
        or gate.get("target_source_id")
        != "lumi_first_party_multilingual_safety_tool_v0_3"
    ):
        raise ValueError("10M closure scale-canary gate changed")
    closure_verification = _load_json(resolved["closure_verification"])
    if not all(closure_verification.get("checks", {}).values()):
        raise ValueError("10M closure verification is not clean")
    promotion = _load_json(resolved["teacher_promotion_receipt"])
    candidate = promotion.get("candidate", {})
    if (
        candidate.get("records") != 80
        or candidate.get("train_records") != 60
        or candidate.get("validation_records") != 20
        or candidate.get("train_supervised_tokens") != 2_002
        or candidate.get("validation_supervised_tokens") != 799
    ):
        raise ValueError("registered teacher canary proof changed")
    promotion_verification = _load_json(resolved["teacher_promotion_verification"])
    if not all(promotion_verification.get("checks", {}).values()):
        raise ValueError("teacher canary promotion verification is not clean")


def _verify_allocations(contract: Mapping[str, Any]) -> None:
    if contract.get("task_allocation") != TASK_ALLOCATION:
        raise ValueError("scale-canary task allocation changed")
    if contract.get("language_allocation") != LANGUAGE_ALLOCATION:
        raise ValueError("scale-canary language allocation changed")
    if contract.get("cell_allocation") != CELL_ALLOCATION:
        raise ValueError("scale-canary cell allocation changed")
    if contract.get("train_cell_allocation") != TRAIN_CELL_ALLOCATION:
        raise ValueError("scale-canary train cell allocation changed")
    if contract.get("validation_cell_allocation") != _validation_cells():
        raise ValueError("scale-canary validation cell allocation changed")
    if (
        sum(TASK_ALLOCATION.values()) != 1_000
        or sum(LANGUAGE_ALLOCATION.values()) != 1_000
        or _sum_nested(CELL_ALLOCATION) != 1_000
        or _sum_nested(TRAIN_CELL_ALLOCATION) != 800
        or _sum_nested(_validation_cells()) != 200
        or _row_totals(CELL_ALLOCATION) != TASK_ALLOCATION
        or _column_totals(CELL_ALLOCATION) != LANGUAGE_ALLOCATION
    ):
        raise ValueError("scale-canary allocation arithmetic changed")


def load_first_party_scale_canary_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-09",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_local_provider_binding_preflight_only",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"scale-canary preflight plan changed: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    for name, expected_sha256 in EXPECTED_FIXED_SHA256.items():
        if plan["bindings"][name].get("sha256") != expected_sha256:
            raise ValueError(f"scale-canary fixed binding changed: {name}")
    if resolved["model_file"].stat().st_size != MODEL_FILE_BYTES:
        raise ValueError("scale-canary model size changed")
    contract = plan.get("canary_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("scale-canary contract is missing")
    _verify_allocations(contract)
    prompt_contract = plan.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise ValueError("scale-canary prompt contract is missing")
    if plan.get("prompt_contract_sha256") != sha256_value(prompt_contract):
        raise ValueError("scale-canary prompt contract hash changed")
    _verify_predecessors(resolved)
    _verify_provider_policy(resolved)
    return plan


def build_first_party_scale_canary_preflight_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve_bindings(plan, repository_root)
    _verify_predecessors(resolved)
    provider_decisions = _verify_provider_policy(resolved)
    runtime = _verify_runtime()
    contract = plan["canary_contract"]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(plan["bindings"]),
        "canary_contract": dict(contract),
        "network_and_compute_accounting": {
            "candidate_records_generated": 0,
            "cloud_or_paid_gpu_used": False,
            "external_api_calls": 0,
            "external_api_cost_usd": 0,
            "local_inference_calls": 0,
            "model_download_bytes": 0,
            "network_bytes": 0,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "prompt_contract_sha256": plan["prompt_contract_sha256"],
        "provider_decisions": provider_decisions,
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "exact_scale_canary_execution_package_not_materialized",
            "bounded_local_generation_not_explicitly_authorized",
            "one_thousand_row_acceptance_latency_throughput_and_quality_not_measured",
            "scale_canary_owner_review_not_completed",
            "three_point_two_million_token_generation_not_authorized",
            "candidate_registry_dataset_seal_and_sft_not_authorized",
        ],
        "runtime_readiness": runtime,
        "schema_version": REPORT_SCHEMA_VERSION,
        "selected_provider": {
            "model_file_bytes": MODEL_FILE_BYTES,
            "model_file_sha256": MODEL_FILE_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "provider": "self_hosted_qwen3_8b_gguf",
            "terms_receipt": {
                "external_service_terms_apply": False,
                "output_noninfringement_warranty_claimed": False,
                "private_research_canary_only": True,
                "self_hosted_generation": True,
                "weights_license": "apache-2.0",
            },
        },
        "status": "verified_local_qwen_provider_binding_no_generation",
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_first_party_scale_canary_preflight_report(report)
    return report


def validate_first_party_scale_canary_preflight_report(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("scale-canary preflight report schema changed")
    if report.get("report_sha256") != _internal_sha(report, "report_sha256"):
        raise ValueError("scale-canary preflight report SHA-256 changed")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("scale-canary preflight retained raw training text")
    accounting = report.get("network_and_compute_accounting", {})
    if accounting != {
        "candidate_records_generated": 0,
        "cloud_or_paid_gpu_used": False,
        "external_api_calls": 0,
        "external_api_cost_usd": 0,
        "local_inference_calls": 0,
        "model_download_bytes": 0,
        "network_bytes": 0,
    }:
        raise ValueError("scale-canary preflight performed an external action")
    if any(report.get("production", {}).values()):
        raise ValueError("scale-canary preflight authorized production")
    selected = report.get("selected_provider", {})
    if (
        selected.get("provider") != "self_hosted_qwen3_8b_gguf"
        or selected.get("model_file_sha256") != MODEL_FILE_SHA256
        or selected.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("scale-canary selected provider changed")
    _verify_allocations(report.get("canary_contract", {}))


def build_first_party_scale_canary_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_first_party_scale_canary_preflight_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "exact_local_model_blob_hash_and_size_verified": True,
            "exact_ollama_manifest_and_client_version_verified": True,
            "external_api_credentials_not_read_or_required": True,
            "no_api_network_download_generation_gpu_registry_seal_or_sft_action": True,
            "openai_and_azure_remain_fail_closed_without_written_clearance": True,
            "prior_80_row_teacher_canary_and_registry_proof_verified": True,
            "prompt_contract_hash_bound": True,
            "qwen_apache_two_terms_receipt_bound": True,
            "task_language_cell_and_split_arithmetic_verified": True,
            "one_thousand_row_and_120k_train_token_ceilings_preserved": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_local_qwen_provider_binding_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
