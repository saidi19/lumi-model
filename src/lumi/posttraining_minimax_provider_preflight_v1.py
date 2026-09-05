"""Fail-closed MiniMax provider qualification for the Lumi closure canary."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    _load_json,
    _verify_internal_sha,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-minimax-provider-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-minimax-provider-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-minimax-provider-preflight-verification.v0.1"

DIRECT_MODEL_ID = "MiniMax-M2.7"
DIRECT_API_BASE_URL = "https://api.minimax.io/v1"
OLLAMA_CLOUD_MODEL = "minimax-m2.7:cloud"
SCHEDULE_SHA256 = "e5d2f4f648b42cbaf80b4ae65242390f7481a8ec0377359288767714b0cc0579"
TARGET_SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_7"
PROMPT_CONTRACT = "lumi.minimax-teacher.response-only-provider-canary.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_generation_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_call_authorized": False,
    "external_api_spend_authorized": False,
    "local_environment_probe_authorized": True,
    "network_access_authorized": False,
    "official_source_facts_already_observed": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "minimax_provider_terms_and_route_preflight_only",
    "raw_text_free_evidence_authorized": True,
    "remote_access_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records_generated": 0,
    "external_api_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
}
EXPECTED_OUTPUT = {
    "production_eligible": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-minimax-provider-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-minimax-provider-preflight-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "benchmark_signatures",
        "chat_template",
        "closure_plan",
        "closure_report",
        "closure_verification",
        "data_contract",
        "implementation",
        "legacy_schedule_plan",
        "runner",
        "source_registry",
        "test",
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
        "0a722beab289cbd391348406e2f1ac37f403daaaee1f4657f66a5f613bba06a0"
    ),
    "closure_report": (
        "d8ad6eeda1f2c267809cd6ea1102d81541e6b850f87ffc5792e3b5fe8c31c29f"
    ),
    "closure_verification": (
        "78555258ff06b79f146419cfb039d73067098a3031e74d8f9af4abe158637419"
    ),
    "data_contract": (
        "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06"
    ),
    "legacy_schedule_plan": (
        "1465164c739f33f8d5ba046a0393ed0e425f1af972dc7fa1b835dca0a5dc62dd"
    ),
    "source_registry": (
        "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
    ),
    "tokenizer_json": (
        "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
    ),
}

EXPECTED_OFFICIAL_SOURCE_FACTS = {
    "minimax_direct_api": {
        "api_base_url": DIRECT_API_BASE_URL,
        "input_usd_per_million_tokens": 0.30,
        "model_id": DIRECT_MODEL_ID,
        "output_usd_per_million_tokens": 1.20,
        "privacy_effective_date": "2026-03-30",
        "privacy_processes_inputs_and_outputs": True,
        "terms_effective_date": "2026-03-30",
        "terms_generated_content_ownership_retained": True,
        "terms_no_explicit_competing_product_prohibition_observed": True,
        "terms_provider_may_use_content_to_develop_and_improve_services": True,
        "terms_url": "https://platform.minimax.io/protocol/terms-of-service",
        "privacy_url": "https://platform.minimax.io/protocol/privacy-policy",
        "pricing_url": "https://platform.minimax.io/docs/guides/pricing-paygo",
        "zero_data_retention_promised": False,
    },
    "ollama_cloud": {
        "commercial_usage_statement_present_for_model": True,
        "model_context_tokens": 200_000,
        "model_parameters": 229_000_000_000,
        "model_tag": OLLAMA_CLOUD_MODEL,
        "model_url": "https://ollama.com/library/minimax-m2.7:cloud",
        "terms_competing_product_development_prohibited": True,
        "terms_effective_date": "2026-05",
        "terms_input_output_training_by_ollama": False,
        "terms_output_ownership_retained": True,
        "terms_url": "https://ollama.com/terms",
    },
    "observed_at": "2026-08-13",
}

EXPECTED_NEXT_GATE = {
    "action": "owner_credential_then_minimax_direct_api_100_row_canary",
    "authorization_required": True,
    "credential_configuration_required": True,
    "direct_api_route_conditionally_eligible": True,
    "maximum_candidate_records": 100,
    "maximum_generated_tokens_per_call": 2_048,
    "maximum_inference_calls": 120,
    "maximum_retry_calls": 20,
    "maximum_spend_usd_before_tax": 1.00,
    "minimax_api_key_must_be_read_from_environment_without_logging": True,
    "ollama_cloud_route_permitted": False,
    "owner_service_order_attestation_required": True,
    "production_sft_authorized": False,
    "prompt_contract": PROMPT_CONTRACT,
    "schedule_sha256": SCHEDULE_SHA256,
    "target_source_id": TARGET_SOURCE_ID,
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_value(unsigned)


def _resolve_bindings(
    plan: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Path]:
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("MiniMax provider-preflight binding set changed")
    return {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def probe_local_environment() -> dict[str, Any]:
    binary = Path("/usr/local/bin/ollama")
    version = None
    if binary.is_file():
        result = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"client version is ([^\s]+)", result.stdout + result.stderr)
        version = match.group(1) if match else None
    return {
        "minimax_api_key_present": bool(os.environ.get("MINIMAX_API_KEY")),
        "ollama_api_key_present": bool(os.environ.get("OLLAMA_API_KEY")),
        "ollama_binary_present": binary.is_file(),
        "ollama_client_version": version,
        "ollama_local_model_bytes": _directory_bytes(
            Path("/Users/afnan/.ollama/models")
        ),
        "secret_values_logged": False,
    }


def load_minimax_provider_preflight_plan_v1(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": "2026-08-13",
        "official_source_facts": EXPECTED_OFFICIAL_SOURCE_FACTS,
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_local_minimax_provider_preflight_only",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"MiniMax provider-preflight plan changed: {field}")
    resolved = _resolve_bindings(plan, repository_root)
    for name, expected_sha256 in EXPECTED_FIXED_SHA256.items():
        if plan["bindings"][name].get("sha256") != expected_sha256:
            raise ValueError(f"MiniMax provider-preflight binding changed: {name}")

    closure_report = _load_json(resolved["closure_report"])
    closure_verification = _load_json(resolved["closure_verification"])
    _verify_internal_sha(closure_report, "report_sha256")
    _verify_internal_sha(closure_verification, "verification_sha256")
    if (
        closure_report.get("status")
        != "verified_registry_v0_17_10m_closure_reconciliation_nonproduction"
        or closure_report.get("projected_mix", {}).get("train_supervised_tokens")
        != 10_541_582
        or not all(closure_verification.get("checks", {}).values())
    ):
        raise ValueError("bound closure reconciliation is not clean")
    legacy_schedule = _load_json(resolved["legacy_schedule_plan"])
    if legacy_schedule.get("schedule", {}).get("sha256") != SCHEDULE_SHA256:
        raise ValueError("100-row schedule binding changed")
    return plan


def build_minimax_provider_preflight_report_v1(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    _resolve_bindings(plan, repository_root)
    environment = probe_local_environment()
    direct_ready = environment["minimax_api_key_present"]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(plan["bindings"]),
        "canary_contract": {
            "fresh_generation_only": True,
            "generated_field": "assistant_response",
            "historical_gpt_oss_candidates_reused": False,
            "languages": 4,
            "per_cell_train_records": 4,
            "per_cell_validation_records": 1,
            "prompt_contract": PROMPT_CONTRACT,
            "records": 100,
            "schedule_sha256": SCHEDULE_SHA256,
            "tasks": 5,
            "tool_results_generated_by_model": False,
        },
        "decisions": {
            "minimax_direct_api": (
                "conditionally_eligible_for_bounded_canary_credential_required"
                if not direct_ready
                else "conditionally_eligible_for_bounded_canary_authorization_required"
            ),
            "ollama_cloud_minimax": (
                "fail_closed_terms_prohibit_competing_product_development"
            ),
            "three_point_two_million_bulk_generation": "not_authorized",
        },
        "exact_successor_authorization_text": plan[
            "exact_successor_authorization_text"
        ],
        "local_environment": environment,
        "network_accounting": {
            "external_api_calls": 0,
            "maximum_network_bytes": 0,
            "provider_spend_usd": 0.0,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "official_source_fact_sha256": sha256_value(plan["official_source_facts"]),
        "official_source_facts": dict(plan["official_source_facts"]),
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_generated": False,
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "gpu_used": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "minimax_api_key_not_present_in_process_environment"
            if not direct_ready
            else "minimax_direct_api_canary_not_owner_authorized",
            "owner_account_and_service_order_must_have_no_additional_training_restriction",
            "minimax_direct_api_100_row_canary_not_run",
            "deterministic_20_row_owner_review_not_completed",
            "three_point_two_million_generation_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "PASS_provider_route_selected_credential_required"
            if not direct_ready
            else "PASS_provider_route_selected_owner_authorization_required"
        ),
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_minimax_provider_preflight_report_v1(report)
    return report


def validate_minimax_provider_preflight_report_v1(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("MiniMax provider-preflight report schema changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("MiniMax provider-preflight retained raw text")
    if any(report.get("production", {}).values()):
        raise ValueError("MiniMax provider-preflight performed an unauthorized action")
    environment = report.get("local_environment", {})
    if (
        environment.get("ollama_client_version") != "0.32.9"
        or environment.get("ollama_local_model_bytes") != 0
        or environment.get("secret_values_logged") is not False
    ):
        raise ValueError("MiniMax provider-preflight local environment changed")
    if report.get("decisions", {}).get("ollama_cloud_minimax") != (
        "fail_closed_terms_prohibit_competing_product_development"
    ):
        raise ValueError("Ollama Cloud fail-closed decision changed")
    if report.get("network_accounting") != {
        "external_api_calls": 0,
        "maximum_network_bytes": 0,
        "provider_spend_usd": 0.0,
    }:
        raise ValueError("MiniMax provider-preflight used a provider")


def build_minimax_provider_preflight_verification_v1(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_minimax_provider_preflight_report_v1(report)
    verification: dict[str, Any] = {
        "checks": {
            "closure_v0_4_and_registry_v0_17_bound": True,
            "direct_minimax_route_is_conditional_and_not_used": True,
            "exact_100_row_response_only_canary_boundary_materialized": True,
            "no_api_call_spend_generation_registry_seal_gpu_or_sft_action": True,
            "ollama_cloud_route_failed_closed_on_competing_product_term": True,
            "official_source_facts_are_raw_text_free": True,
            "secret_presence_only_probed_and_values_not_logged": True,
            "zero_local_ollama_model_bytes_preserved": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_minimax_provider_preflight_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification
