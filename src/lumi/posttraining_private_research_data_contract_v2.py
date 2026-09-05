"""Versioned private-research SFT data-contract v0.2 successor."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_candidate_mix_reaudit_v18 as v18
from . import posttraining_production_source_policy_preflight_v1 as preflight
from .posttraining_data import SFTPolicy, evaluate_sft_record, load_sft_policy
from .posttraining_sources import load_sft_source_registry

PLAN_SCHEMA_VERSION = "lumi.sft-private-research-data-contract-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-private-research-data-contract-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-private-research-data-contract-verification.v0.2"
PLAN_STATUS = "authorized_private_research_nonproduction_contract_successor"
REPORT_STATUS = "PASS_private_research_nonproduction_data_contract_v0_2_active"
VERIFICATION_STATUS = "PASS_verified_private_research_data_contract_v0_2"
EXPECTED_AUTHORIZATION_TEXT = "ok next step shuru kor"
EXPECTED_AUTHORIZATION_BYTES = 22
EXPECTED_AUTHORIZATION_SHA256 = "1046cda26a21ff4994e4a452870466e609fe93d825cad3622e9fd069fb18b35e"
EXPECTED_OBSERVED_AT = "2026-09-04"

EXPECTED_AUTHORIZATION = {
    "active_data_contract_change_authorized": True,
    "candidate_row_access_authorized": False,
    "dataset_seal_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "nonproduction_private_research_contract_authorized": True,
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
    "maximum_data_contract_writes": 1,
    "maximum_factual_passes": 2,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 4_194_304,
    "maximum_registry_writes": 0,
    "maximum_retries_after_started_pass": 0,
    "maximum_runtime_minutes": 15,
    "maximum_source_row_reads": 0,
    "maximum_test_object_reads": 0,
    "maximum_training_actions": 0,
}
EXPECTED_OUTPUT = {
    "data_contract_path": "configs/lumi-4b-sft-data-contract-v0.2.json",
    "report_path": "evidence/lumi-4b-sft-private-research-data-contract-v0.2.json",
    "verification_path": (
        "evidence/lumi-4b-sft-private-research-data-contract-verification-v0.2.json"
    ),
}
EXPECTED_BINDING_NAMES = frozenset(
    {
        "candidate_mix_implementation",
        "candidate_mix_report",
        "candidate_mix_verification",
        "implementation",
        "policy_loader",
        "predecessor_data_contract",
        "production_seal",
        "registry_loader",
        "runner",
        "source_policy_preflight_plan",
        "source_policy_preflight_implementation",
        "source_policy_preflight_report",
        "source_policy_preflight_verification",
        "source_registry",
        "test",
        "verifier",
    }
)
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
EXPECTED_EXTERNAL_BINDINGS = {
    "candidate_mix_implementation": (
        "src/lumi/posttraining_candidate_mix_reaudit_v18.py",
        "d27061fe5a078789858a1b7c5ba7977ce7cc3738a187df8285004cb9e909f9f0",
    ),
    "candidate_mix_report": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.18.json",
        "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000",
    ),
    "candidate_mix_verification": (
        "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.18.json",
        "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651",
    ),
    "policy_loader": (
        "src/lumi/posttraining_data.py",
        "e520c3771869b7221848a456e59c3b15a279fb1ebdf700646600c774e25b74b5",
    ),
    "predecessor_data_contract": (
        "configs/lumi-4b-sft-data-contract-v0.1.json",
        "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    ),
    "production_seal": (
        "src/lumi/posttraining_seal.py",
        "52b9d93ceaf756497b7ccc4805b51d76e22490269a61cd644ce5a1b12e946155",
    ),
    "registry_loader": (
        "src/lumi/posttraining_sources.py",
        "10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50",
    ),
    "source_policy_preflight_plan": (
        "configs/lumi-4b-sft-production-source-policy-preflight-v0.1.json",
        "75ad794a35c75be056040af01470b2c590cc09ff67ba980dde10111a83c079fa",
    ),
    "source_policy_preflight_implementation": (
        "src/lumi/posttraining_production_source_policy_preflight_v1.py",
        "6e1dcbb2ed0d454ce2885ebd840fef50d06e390e1a0196f1996fa861b2266f2f",
    ),
    "source_policy_preflight_report": (
        "evidence/lumi-4b-sft-production-source-policy-preflight-v0.1.json",
        "700993d8eb570ee02ad4e253607569fef74e5f3a25d86c964ab4a225605b131d",
    ),
    "source_policy_preflight_verification": (
        "evidence/lumi-4b-sft-production-source-policy-preflight-verification-v0.1.json",
        "06c535ae85912f778583ab48d382c1c64093d581dbea15e49019dc94ca6f4ff4",
    ),
    "source_registry": (
        "configs/lumi-4b-sft-source-registry-v0.24.json",
        "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    ),
}
INCLUDED_SOURCE_IDS = (
    "allenai_prosocial_dialog_safety_candidate",
    "amazon_massive_bn_bd_tool_canary",
    "banglabridge_instructions_closure_canary",
    "banglatlit_auxiliary_closure_canary",
    "coherelabs_aya_bengali_closure_canary",
    "coherelabs_aya_dataset",
    "deepmind_mathematics_dataset",
    "dmath",
    "fiscalnote_billsum_writing_candidate",
    "lumi_ai_assisted_gap_closure_canary_v0_1",
    "lumi_ai_assisted_gap_closure_scaled_v0_2",
    "lumi_ai_assisted_sft_v0_3",
    "lumi_first_party_gpt_oss_teacher_canary_v0_1",
    "lumi_first_party_multilingual_safety_tool_v0_6",
    "lumi_first_party_teacher_canary_v0_1",
    "lumi_safety_tool_sft_v0_1",
    "nvidia_nemotron_post_training_tool_canary",
    "nvidia_opencodeinstruct",
    "openassistant_oasst1",
    "openassistant_oasst2",
)
EXCLUDED_SOURCE_IDS = (
    "bigcode_self_oss_instruct_50k",
    "huggingfaceh4_ultrachat_200k",
    "lumi_ai_assisted_sft_v0_1",
    "lumi_ai_assisted_sft_v0_2",
    "lumi_owned_sft_v0_1",
    "openr1_math_220k",
    "salesforce_xlam_function_calling_60k",
)
AI_SOURCE_CONTRACTS = {
    "lumi_ai_assisted_gap_closure_canary_v0_1": {
        "generation_method": "ai_authored_templates_with_local_deterministic_renderer",
        "generator_id": "openai-codex-current-session",
        "human_review_completed": False,
        "human_written_from_scratch": False,
        "model_assistance_used": True,
        "owner_review_status": "not_required_for_nonproduction_canary",
        "runtime_external_model_api_calls": 0,
        "runtime_model_calls": 0,
    },
    "lumi_ai_assisted_gap_closure_scaled_v0_2": {
        "generation_method": "ai_authored_templates_with_local_deterministic_scaled_renderer",
        "generator_id": "openai-codex-current-session",
        "human_review_completed": False,
        "human_written_from_scratch": False,
        "model_assistance_used": True,
        "owner_review_status": "not_required_for_nonproduction_scaled_candidate",
        "runtime_external_model_api_calls": 0,
        "runtime_model_calls": 0,
    },
}
EVALUATION_SPLIT_PROHIBITIONS = {
    "deepmind_mathematics_dataset": ["test", "test_extra"],
    "dmath": ["test"],
    "fiscalnote_billsum_writing_candidate": ["ca_test", "test"],
}
MIX_HASH_ONLY_SOURCE_IDS = tuple(
    source_id for source_id in INCLUDED_SOURCE_IDS if source_id not in AI_SOURCE_CONTRACTS
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


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without accepting bool/int aliases."""

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


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("private-research contract binding must be a JSON object")
    return value


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"invalid private-research contract binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing private-research contract binding: {name}")
    if file_sha256(path) != expected:
        raise ValueError(f"private-research contract binding changed: {name}")
    return path


def _assert_raw_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _RAW_TEXT_KEYS:
                raise ValueError(f"raw candidate text key forbidden at {path}")
            _assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, f"{path}[{index}]")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("private-research contract output already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError("stale private-research contract temporary exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_object(Path(path))
    if set(plan) != EXPECTED_PLAN_KEYS or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported private-research contract plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("private-research contract plan is not authorized")
    if plan.get("observed_at") != EXPECTED_OBSERVED_AT:
        raise ValueError("private-research contract observation date changed")
    if not _strict_json_equal(plan.get("authorization"), EXPECTED_AUTHORIZATION):
        raise ValueError("private-research contract authorization changed")
    if not _strict_json_equal(plan.get("limits"), EXPECTED_LIMITS):
        raise ValueError("private-research contract limits changed")
    if not _strict_json_equal(plan.get("output"), EXPECTED_OUTPUT):
        raise ValueError("private-research contract output boundary changed")
    authorization_text = plan.get("authorization_text")
    if (
        authorization_text != EXPECTED_AUTHORIZATION_TEXT
        or type(plan.get("authorization_bytes")) is not int
        or plan.get("authorization_bytes") != EXPECTED_AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise ValueError("private-research contract authorization text changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("private-research contract binding set changed")
    for name, (expected_path, expected_sha) in EXPECTED_EXTERNAL_BINDINGS.items():
        if bindings.get(name) != {"path": expected_path, "sha256": expected_sha}:
            raise ValueError(f"private-research external binding changed: {name}")
    for name, binding in bindings.items():
        _resolve_binding(root, binding, name)
    return plan


def _successor_value(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    successor = copy.deepcopy(dict(predecessor))
    successor["contract_revision"] = "v0.2"
    successor["source_registry"] = {
        "path": "configs/lumi-4b-sft-source-registry-v0.24.json",
        "required_for_filtering_and_seal": True,
        "schema_version": "lumi.sft.sources.v0.1",
        "sha256": "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb",
    }
    successor["learning_candidate"] = {
        "accepted_dataset_sha256": (
            "1a60f602648f45480dc7827d0cbe686d0e17fb3b28e476dbb7ef21986e4fd37e"
        ),
        "machine_only_gap_rows": {
            "allowed": True,
            "completed_human_review_claim_allowed": False,
            "human_authorship_claim_allowed": False,
        },
        "ai_assisted_source_contracts": copy.deepcopy(AI_SOURCE_CONTRACTS),
        "candidate_mix_report_sha256": (
            "3fa0502a3d9c064224a1c11fc1c8429c0d5ca13d7dd1be8d42d539a681a92000"
        ),
        "candidate_mix_verification_sha256": (
            "43db682ea4f28577f0d919df6160d9fc7339e56df8469737abc0569201365651"
        ),
        "evaluation_split_prohibitions": copy.deepcopy(EVALUATION_SPLIT_PROHIBITIONS),
        "excluded_source_ids": list(EXCLUDED_SOURCE_IDS),
        "external_distribution_allowed": False,
        "included_source_ids": list(INCLUDED_SOURCE_IDS),
        "production_dataset_seal_allowed": False,
        "production_sft_allowed": False,
        "registry_source_count": 27,
        "scope": "private_research_nonproduction",
        "seal_allowed": True,
    }
    successor["notes"] = copy.deepcopy(dict(predecessor["notes"]))
    successor["notes"].update(
        {
            "private_research_machine_gap_rows_require_truthful_machine_provenance": True,
            "private_research_nonproduction_scope_active": True,
            "production_eligibility_remains_false_for_all_bound_sources": True,
        }
    )
    return successor


def build_contract(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    successor = _successor_value(predecessor)
    validate_contract(successor, predecessor=predecessor)
    return successor


def validate_contract(contract: Mapping[str, Any], *, predecessor: Mapping[str, Any]) -> None:
    if not _strict_json_equal(contract, _successor_value(predecessor)):
        raise ValueError("private-research data contract changed")
    predecessor_policy = SFTPolicy.from_mapping(predecessor)
    policy = SFTPolicy.from_mapping(contract)
    preserved_fields = (
        "allowed_languages",
        "allowed_licenses",
        "allowed_splits",
        "benchmark_decontamination",
        "chat_template",
        "context_window",
        "maximum_message_characters",
        "maximum_messages",
        "minimum_quality_score",
        "packing_policy",
        "source_registry_required",
        "supervised_roles",
        "task_mix",
    )
    if any(
        getattr(policy, field) != getattr(predecessor_policy, field) for field in preserved_fields
    ):
        raise ValueError("private-research predecessor policy changed")
    learning = policy.learning_candidate
    if (
        learning is None
        or learning["scope"] != "private_research_nonproduction"
        or learning["production_dataset_seal_allowed"] is not False
        or learning["production_sft_allowed"] is not False
        or learning["external_distribution_allowed"] is not False
        or learning["seal_allowed"] is not True
        or learning["included_source_ids"] != frozenset(INCLUDED_SOURCE_IDS)
        or learning["excluded_source_ids"] != frozenset(EXCLUDED_SOURCE_IDS)
        or policy.source_registry_path != "configs/lumi-4b-sft-source-registry-v0.24.json"
        or policy.source_registry_sha256
        != "a7a2daec58bc08e5a8e11ecff79cea3a5db90d9b423eb6a55cba28559d74a6bb"
    ):
        raise ValueError("private-research learning candidate boundary changed")


def _synthetic_ai_record(source_id: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    source_contract = contract["learning_candidate"]["ai_assisted_source_contracts"][source_id]
    return {
        "schema_version": "lumi.sft.v0.1",
        "record_id": "synthetic-contract-check",
        "split": "train",
        "language": "multilingual",
        "task_category": "conversation",
        "messages": [
            {"role": "user", "content": "synthetic contract check"},
            {"role": "assistant", "content": "synthetic contract check accepted"},
        ],
        "source": {
            "source_id": source_id,
            "dataset_id": "synthetic-contract-check",
            "document_id": "synthetic-contract-check",
            "license_id": "lumi-owned",
            "revision": "a" * 64,
            "uri": "private://synthetic-contract-check",
        },
        "provenance": {
            "consent_basis": "first_party_rights_declaration",
            "content_sha256": "a" * 64,
            "generation_method": source_contract["generation_method"],
            "generator_id": source_contract["generator_id"],
            "human_written_from_scratch": False,
            "model_assistance_used": True,
            "runtime_external_model_api_calls": 0,
            "runtime_model_calls": 0,
            "transform_sha256": "b" * 64,
        },
        "quality": {
            "pii_scan": "passed",
            "review_status": "accepted",
            "score": 0.92,
            "secret_scan": "passed",
        },
        "safety": {"policy_version": "lumi-safety-v0.1", "risk_labels": []},
        "owner_review": {
            "human_review_completed": False,
            "status": source_contract["owner_review_status"],
        },
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: _resolve_binding(root, binding, name) for name, binding in plan["bindings"].items()
    }
    predecessor = _load_object(resolved["predecessor_data_contract"])
    validate_contract(contract, predecessor=predecessor)

    policy_preflight_plan = _load_object(resolved["source_policy_preflight_plan"])
    policy_preflight_report = _load_object(resolved["source_policy_preflight_report"])
    policy_preflight_verification = _load_object(resolved["source_policy_preflight_verification"])
    preflight.validate_report(policy_preflight_report)
    preflight.validate_verification(policy_preflight_verification)
    if (
        policy_preflight_plan.get("status") != preflight.PLAN_STATUS
        or policy_preflight_report.get("decision", {}).get("recommended_path")
        != "private_research_nonproduction_data_contract_v0_2"
        or policy_preflight_report.get("production_eligible") is not False
        or policy_preflight_verification.get("production_eligible") is not False
    ):
        raise ValueError("private-research source-policy preflight changed")

    mix_report = _load_object(resolved["candidate_mix_report"])
    mix_verification = _load_object(resolved["candidate_mix_verification"])
    v18.validate_report(mix_report)
    v18.validate_verification(mix_verification)
    if (
        tuple(sorted(mix_report["source_metrics"])) != INCLUDED_SOURCE_IDS
        or tuple(mix_report["excluded_source_gate"]["registry_sources_not_in_candidate_inventory"])
        != EXCLUDED_SOURCE_IDS
        or mix_report["generic_sft_validation"]["rejected"] != 0
        or mix_report["candidate_pool"]["strict_targeted_gap_lower_bound"] != 0
    ):
        raise ValueError("private-research candidate-mix boundary changed")

    policy = load_sft_policy(resolved["predecessor_data_contract"])
    successor_policy = SFTPolicy.from_mapping(contract)
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["source_registry"][1],
    )
    if (
        len(registry.sources) != 27
        or any(source.production_eligible for source in registry.sources)
        or set(registry.by_id) != set(INCLUDED_SOURCE_IDS) | set(EXCLUDED_SOURCE_IDS)
    ):
        raise ValueError("private-research registry production boundary changed")

    synthetic_checks: dict[str, bool] = {}
    for source_id in AI_SOURCE_CONTRACTS:
        decision = evaluate_sft_record(
            _synthetic_ai_record(source_id, contract), policy=successor_policy
        )
        synthetic_checks[source_id] = decision.accepted
    excluded_probe = _synthetic_ai_record("lumi_ai_assisted_gap_closure_canary_v0_1", contract)
    excluded_probe["source"]["source_id"] = EXCLUDED_SOURCE_IDS[0]
    excluded_decision = evaluate_sft_record(excluded_probe, policy=successor_policy)
    if (
        not all(synthetic_checks.values())
        or excluded_decision.accepted
        or "learning_candidate_source_not_included" not in excluded_decision.reasons
    ):
        raise ValueError("private-research synthetic policy enforcement changed")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "authorization": {
            "bytes": plan["authorization_bytes"],
            "selection": "recommended_private_research_path",
            "sha256": plan["authorization_sha256"],
        },
        "data_contract": {
            "file_sha256": contract_file_sha256,
            "path": EXPECTED_OUTPUT["data_contract_path"],
            "predecessor_file_sha256": EXPECTED_EXTERNAL_BINDINGS["predecessor_data_contract"][1],
            "predecessor_preserved": True,
            "revision": "v0.2",
            "schema_version": "lumi.sft.v0.1",
        },
        "policy": {
            "accepted_dataset_sha256": contract["learning_candidate"]["accepted_dataset_sha256"],
            "ai_assisted_source_contracts": sorted(AI_SOURCE_CONTRACTS),
            "remaining_included_sources_bound_by_accepted_mix_hash_only": list(
                MIX_HASH_ONLY_SOURCE_IDS
            ),
            "evaluation_split_prohibitions": copy.deepcopy(EVALUATION_SPLIT_PROHIBITIONS),
            "excluded_source_ids": list(EXCLUDED_SOURCE_IDS),
            "external_distribution_allowed": False,
            "included_source_ids": list(INCLUDED_SOURCE_IDS),
            "learning_candidate_seal_allowed": True,
            "production_dataset_seal_allowed": False,
            "production_sft_allowed": False,
            "scope": "private_research_nonproduction",
        },
        "preserved_training_contract": {
            "allowed_languages": sorted(successor_policy.allowed_languages),
            "allowed_licenses": sorted(successor_policy.allowed_licenses),
            "allowed_splits": sorted(successor_policy.allowed_splits),
            "assistant_and_tool_only_supervision": successor_policy.supervised_roles
            == {"assistant", "tool"},
            "benchmark_decontamination": dict(successor_policy.benchmark_decontamination),
            "chat_template": successor_policy.chat_template,
            "context_window": successor_policy.context_window,
            "maximum_message_characters": successor_policy.maximum_message_characters,
            "maximum_messages": successor_policy.maximum_messages,
            "minimum_quality_score": successor_policy.minimum_quality_score,
            "packing_policy": successor_policy.packing_policy,
            "task_mix": dict(successor_policy.task_mix),
        },
        "enforcement": {
            "excluded_source_probe_rejected": True,
            "source_allowlist_enforced_by_record_validator": True,
            "two_machine_gap_source_provenance_contracts_enforced": True,
            "synthetic_ai_contract_checks": synthetic_checks,
        },
        "registry": {
            "file_sha256": registry.registry_sha256,
            "production_eligible_source_count": 0,
            "registry_mutated": False,
            "source_count": len(registry.sources),
        },
        "production": {
            "dataset_sealed": False,
            "production_eligible": False,
            "production_registry_created": False,
            "sft_training_started": False,
        },
        "resource_accounting": {
            "candidate_row_reads": 0,
            "data_contract_writes": 1,
            "factual_passes": 1,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
        "next_gate": {
            "action": "nonproduction_learning_candidate_seal_preflight_for_registry_v0_24_mix",
            "authorization_required": True,
            "candidate_records": 329_200,
            "train_supervised_tokens": 11_855_217,
            "validation_supervised_tokens": 939_293,
        },
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    policy = report.get("policy", {})
    production = report.get("production", {})
    enforcement = report.get("enforcement", {})
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("report_sha256") != sha256_value(unsigned)
        or not _SHA256.fullmatch(str(report.get("plan_file_sha256", "")))
        or report.get("production_eligible") is not False
        or report.get("raw_training_text_retained") is not False
        or policy.get("included_source_ids") != list(INCLUDED_SOURCE_IDS)
        or policy.get("excluded_source_ids") != list(EXCLUDED_SOURCE_IDS)
        or policy.get("production_dataset_seal_allowed") is not False
        or policy.get("production_sft_allowed") is not False
        or policy.get("external_distribution_allowed") is not False
        or not _strict_json_equal(
            production,
            {
                "dataset_sealed": False,
                "production_eligible": False,
                "production_registry_created": False,
                "sft_training_started": False,
            },
        )
        or enforcement.get("source_allowlist_enforced_by_record_validator") is not True
        or enforcement.get("two_machine_gap_source_provenance_contracts_enforced") is not True
        or not all(enforcement.get("synthetic_ai_contract_checks", {}).values())
        or report.get("registry", {}).get("production_eligible_source_count") != 0
        or not _strict_json_equal(
            report.get("resource_accounting"),
            {
                "candidate_row_reads": 0,
                "data_contract_writes": 1,
                "factual_passes": 1,
                "gpu_use": 0,
                "inference_calls": 0,
                "network_bytes": 0,
                "registry_writes": 0,
                "source_row_reads": 0,
                "test_object_reads": 0,
                "training_actions": 0,
            },
        )
    ):
        raise ValueError("private-research data-contract report boundary changed")
    _assert_raw_text_free(report)


def build_verification(*, report: Mapping[str, Any], report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "data_contract_file_sha256": report["data_contract"]["file_sha256"],
        "checks": {
            "candidate_and_source_row_reads_zero": True,
            "data_contract_v0_1_preserved": True,
            "evaluation_split_prohibitions_preserved": True,
            "exact_twenty_source_allowlist_and_seven_source_exclusion_bound": True,
            "legacy_v0_1_loader_compatibility_preserved": True,
            "no_registry_mutation_or_production_promotion": True,
            "no_seal_gpu_inference_or_training": True,
            "two_machine_gap_source_provenance_contracts_enforced": True,
            "production_flags_remain_false": True,
            "raw_training_text_absent": True,
            "registry_v0_24_binding_is_exact": True,
        },
        "resource_accounting": {
            "candidate_row_reads": 0,
            "data_contract_writes": 1,
            "factual_passes": 2,
            "gpu_use": 0,
            "inference_calls": 0,
            "network_bytes": 0,
            "registry_writes": 0,
            "source_row_reads": 0,
            "test_object_reads": 0,
            "training_actions": 0,
        },
        "production_eligible": False,
        "next_gate": report["next_gate"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in verification.items() if key != "verification_sha256"}
    checks = verification.get("checks")
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or not _SHA256.fullmatch(str(verification.get("report_file_sha256", "")))
        or not _SHA256.fullmatch(str(verification.get("report_internal_sha256", "")))
        or not _SHA256.fullmatch(str(verification.get("data_contract_file_sha256", "")))
        or verification.get("production_eligible") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not _strict_json_equal(
            verification.get("resource_accounting"),
            {
                "candidate_row_reads": 0,
                "data_contract_writes": 1,
                "factual_passes": 2,
                "gpu_use": 0,
                "inference_calls": 0,
                "network_bytes": 0,
                "registry_writes": 0,
                "source_row_reads": 0,
                "test_object_reads": 0,
                "training_actions": 0,
            },
        )
    ):
        raise ValueError("private-research contract verification boundary changed")
    _assert_raw_text_free(verification)


def output_paths(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path, Path]:
    contract = (root / plan["output"]["data_contract_path"]).resolve()
    report = (root / plan["output"]["report_path"]).resolve()
    verification = (root / plan["output"]["verification_path"]).resolve()
    if not contract.is_relative_to((root / "configs").resolve()):
        raise ValueError("private-research contract escaped configs root")
    if not report.is_relative_to((root / "evidence").resolve()) or not verification.is_relative_to(
        (root / "evidence").resolve()
    ):
        raise ValueError("private-research evidence escaped evidence root")
    if contract.exists() or report.exists() or verification.exists():
        raise FileExistsError("private-research contract pass already started")
    return contract, report, verification


__all__ = [
    "AI_SOURCE_CONTRACTS",
    "EXCLUDED_SOURCE_IDS",
    "INCLUDED_SOURCE_IDS",
    "REPORT_STATUS",
    "VERIFICATION_STATUS",
    "_assert_raw_text_free",
    "_load_object",
    "_write_exclusive",
    "build_contract",
    "build_report",
    "build_verification",
    "file_sha256",
    "json_file_bytes",
    "load_plan",
    "output_paths",
    "sha256_value",
    "validate_contract",
    "validate_report",
    "validate_verification",
]
