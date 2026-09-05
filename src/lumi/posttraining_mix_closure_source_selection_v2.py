"""Rights-aware v0.14 source selection for closing the Lumi 4B 10M SFT mix."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    _load_json,
    _verify_internal_sha,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-plan.v0.2"
REPORT_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-report.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-10m-mix-closure-source-selection-verification.v0.2"

EXPECTED_AUTHORIZATION = {
    "bulk_dataset_download_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "official_metadata_lookup_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "lumi_sft_10m_v0_14_mix_closure_source_selection_and_rights_preflight_only",
    "raw_text_free_plan_authorized": True,
    "unrelated_data_access_authorized": False,
}

EXPECTED_LIMITS = {
    "bulk_dataset_payload_bytes": 0,
    "maximum_network_bytes": 16 * 1024 * 1024,
    "maximum_runtime_minutes": 120,
}

EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.2.json",
    "verification_path": (
        "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.2.json"
    ),
}

EXPECTED_BINDINGS = frozenset(
    {
        "candidate_mix_reaudit",
        "candidate_mix_reaudit_verification",
        "data_contract",
        "implementation",
        "missing_mix_proposal",
        "prior_source_preflight",
        "prior_source_preflight_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)

EXPECTED_REAUDIT_FILE_SHA256 = "ab9d3575ae7de7c10858ef4abb72d667f986e1b514db49601eb044935008cd60"
EXPECTED_REAUDIT_VERIFICATION_FILE_SHA256 = (
    "f6d9c813945507e117799d9469ca0ce919234e37afd61c05d0cee6db3e252b61"
)
EXPECTED_REGISTRY_FILE_SHA256 = "a7310e469abb06cf4fbd12a72b2dcf2fbbebeac064e7b9182b994a9a77254e16"
EXPECTED_CURRENT_TRAIN_TOKENS = 7_254_668
EXPECTED_GAP_TO_10M = 2_745_332
EXPECTED_MINIMUM_TASK_CLOSURE = 2_968_674
EXPECTED_TARGET_TOKENS = 3_200_000
EXPECTED_PROJECTED_TOKENS = 10_454_668
SOURCE_CAP = 0.35

EXPECTED_SELECTED_SOURCE_IDS = {
    "lumi_first_party_multilingual_safety_tool_v0_3",
}
EXPECTED_CONDITIONAL_SOURCE_IDS = {
    "nvidia_helpsteer",
    "salesforce_xlam_function_calling_60k",
}
EXPECTED_EXCLUDED_SOURCE_IDS = {
    "allenai_tulu_3_sft_mixture",
    "allenai_wildguardmix",
    "huggingfacetb_smol_smoltalk",
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_value(unsigned)


def _source_map(rows: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    mapped = {
        row.get("source_id"): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    }
    if len(mapped) != len(rows):
        raise ValueError(f"{label} contains duplicate or invalid source ids")
    return mapped


def _verify_allocation(row: Mapping[str, Any], *, label: str) -> None:
    target = row.get("target_retained_train_supervised_tokens")
    if not isinstance(target, int) or target <= 0:
        raise ValueError(f"invalid retained target: {label}")
    for field in ("task_allocation", "language_allocation"):
        allocation = row.get(field)
        if not isinstance(allocation, Mapping) or sum(allocation.values()) != target:
            raise ValueError(f"{field} does not match target: {label}")


def _verify_official_evidence(plan: Mapping[str, Any]) -> None:
    rows = plan.get("official_evidence")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("official evidence inventory changed")
    allowed_prefixes = (
        "https://github.com/",
        "https://huggingface.co/datasets/",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("official evidence row must be an object")
        urls = row.get("urls")
        claims = row.get("claims")
        if not isinstance(urls, list) or not urls:
            raise ValueError("official evidence URL missing")
        if not isinstance(claims, list) or not claims:
            raise ValueError("official evidence claims missing")
        if any(not isinstance(url, str) or not url.startswith(allowed_prefixes) for url in urls):
            raise ValueError("non-official evidence URL present")


def _verify_source_decisions(plan: Mapping[str, Any]) -> None:
    selected = _source_map(plan.get("selected_lanes"), label="selected_lanes")
    if set(selected) != EXPECTED_SELECTED_SOURCE_IDS:
        raise ValueError("selected source set changed")
    for source_id, row in selected.items():
        _verify_allocation(row, label=source_id)
    if (
        sum(row["target_retained_train_supervised_tokens"] for row in selected.values())
        != EXPECTED_TARGET_TOKENS
    ):
        raise ValueError("selected source target changed")

    conditional = _source_map(plan.get("conditional_reserves"), label="conditional_reserves")
    if set(conditional) != EXPECTED_CONDITIONAL_SOURCE_IDS:
        raise ValueError("conditional reserve set changed")
    if any(row.get("counted_in_guaranteed_closure") is not False for row in conditional.values()):
        raise ValueError("conditional reserve counted in guaranteed closure")

    exclusions = _source_map(plan.get("exclusions"), label="exclusions")
    if set(exclusions) != EXPECTED_EXCLUDED_SOURCE_IDS:
        raise ValueError("excluded source set changed")
    if any(row.get("counted_in_guaranteed_closure") is not False for row in exclusions.values()):
        raise ValueError("excluded source counted in guaranteed closure")


def _proposal_floors(
    proposal: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    candidate = proposal.get("learning_candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("learning candidate policy missing")
    tasks = candidate.get("task_targets")
    languages = candidate.get("language_targets")
    if not isinstance(tasks, Mapping) or not isinstance(languages, Mapping):
        raise ValueError("learning candidate floors missing")
    task_floors = {key: int(value["minimum_tokens"]) for key, value in tasks.items()}
    language_floors = {key: int(value["minimum_tokens"]) for key, value in languages.items()}
    return task_floors, language_floors


def load_mix_closure_source_selection_plan_v2(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_metadata_only_v0_14_10m_mix_closure_source_selection",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"v0.14 mix-closure source-selection changed: {field}")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("v0.14 mix-closure binding set changed")
    resolved = {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    if bindings["candidate_mix_reaudit"].get("sha256") != EXPECTED_REAUDIT_FILE_SHA256:
        raise ValueError("authorized v0.14 re-audit report binding changed")
    if (
        bindings["candidate_mix_reaudit_verification"].get("sha256")
        != EXPECTED_REAUDIT_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("authorized v0.14 re-audit verification binding changed")
    if bindings["source_registry"].get("sha256") != EXPECTED_REGISTRY_FILE_SHA256:
        raise ValueError("v0.14 source registry binding changed")

    reaudit = _load_json(resolved["candidate_mix_reaudit"])
    reaudit_verification = _load_json(resolved["candidate_mix_reaudit_verification"])
    prior = _load_json(resolved["prior_source_preflight"])
    prior_verification = _load_json(resolved["prior_source_preflight_verification"])
    for value, field in (
        (reaudit, "report_sha256"),
        (reaudit_verification, "verification_sha256"),
        (prior, "report_sha256"),
        (prior_verification, "verification_sha256"),
    ):
        _verify_internal_sha(value, field)

    pool = reaudit.get("candidate_pool", {})
    if pool.get("accepted_train_supervised_tokens") != EXPECTED_CURRENT_TRAIN_TOKENS:
        raise ValueError("current candidate token total changed")
    if pool.get("gap_to_10m_train_supervised_tokens") != EXPECTED_GAP_TO_10M:
        raise ValueError("current 10M gap changed")
    closure = reaudit.get("closure", {})
    if sum(closure.get("task_floor_gaps", {}).values()) != EXPECTED_MINIMUM_TASK_CLOSURE:
        raise ValueError("minimum task closure changed")
    if sum(closure.get("language_floor_gaps", {}).values()) != EXPECTED_GAP_TO_10M:
        raise ValueError("minimum language closure changed")

    minimum = plan.get("minimum_accepted_closure")
    if not isinstance(minimum, Mapping):
        raise ValueError("minimum closure summary missing")
    if minimum.get("task_floor_gaps") != closure.get("task_floor_gaps"):
        raise ValueError("minimum task closure does not bind the v0.14 re-audit")
    if minimum.get("language_floor_gaps") != closure.get("language_floor_gaps"):
        raise ValueError("minimum language closure does not bind the v0.14 re-audit")

    _verify_source_decisions(plan)
    _verify_official_evidence(plan)
    _proposal_floors(_load_json(resolved["missing_mix_proposal"]))
    return plan


def _sum_allocations(
    current: Mapping[str, int],
    lanes: list[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, int]:
    totals = dict(current)
    for lane in lanes:
        for label, value in lane[key].items():
            totals[label] = totals.get(label, 0) + int(value)
    return dict(sorted(totals.items()))


def _project_source_tokens(
    current: Mapping[str, int], lanes: list[Mapping[str, Any]]
) -> tuple[dict[str, int], dict[str, float], int]:
    tokens = dict(current)
    for lane in lanes:
        source_id = lane["source_id"]
        tokens[source_id] = (
            tokens.get(source_id, 0) + lane["target_retained_train_supervised_tokens"]
        )
    total = sum(tokens.values())
    shares = {source_id: round(value / total, 8) for source_id, value in sorted(tokens.items())}
    return dict(sorted(tokens.items())), shares, total


def _project_family_shares(
    source_tokens: Mapping[str, int],
    groups: Mapping[str, list[str]],
    *,
    total: int,
) -> tuple[dict[str, int], dict[str, float]]:
    flattened = [source_id for source_ids in groups.values() for source_id in source_ids]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(source_tokens):
        raise ValueError("source family groups must cover every projected source exactly once")
    family_tokens = {
        family: sum(source_tokens[source_id] for source_id in source_ids)
        for family, source_ids in groups.items()
    }
    family_shares = {
        family: round(tokens / total, 8) for family, tokens in sorted(family_tokens.items())
    }
    return dict(sorted(family_tokens.items())), family_shares


def rebuild_mix_closure_source_selection_v2(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    reaudit_path = resolve_binding(
        repository_root,
        plan["bindings"]["candidate_mix_reaudit"],
        name="candidate_mix_reaudit",
    )
    proposal_path = resolve_binding(
        repository_root,
        plan["bindings"]["missing_mix_proposal"],
        name="missing_mix_proposal",
    )
    reaudit = _load_json(reaudit_path)
    proposal = _load_json(proposal_path)
    task_floors, language_floors = _proposal_floors(proposal)
    pool = reaudit["candidate_pool"]
    lanes = [dict(row) for row in plan["selected_lanes"]]

    projected_tasks = _sum_allocations(pool["train_by_task"], lanes, key="task_allocation")
    projected_languages = _sum_allocations(
        pool["train_by_language_mode"], lanes, key="language_allocation"
    )
    current_source_tokens = {
        source_id: metrics["by_split"]["train"]["supervised_tokens"]
        for source_id, metrics in reaudit["source_metrics"].items()
    }
    source_tokens, source_shares, projected_total = _project_source_tokens(
        current_source_tokens, lanes
    )
    family_tokens, family_shares = _project_family_shares(
        source_tokens,
        plan["source_family_groups"],
        total=projected_total,
    )
    task_gaps = {
        key: max(0, floor - projected_tasks.get(key, 0)) for key, floor in task_floors.items()
    }
    language_gaps = {
        key: max(0, floor - projected_languages.get(key, 0))
        for key, floor in language_floors.items()
    }
    source_violations = [key for key, share in source_shares.items() if share > SOURCE_CAP]
    family_violations = [key for key, share in family_shares.items() if share > SOURCE_CAP]

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(plan["bindings"]),
        "current_mix": {
            "accepted_train_supervised_tokens": pool["accepted_train_supervised_tokens"],
            "gap_to_10m_train_supervised_tokens": pool["gap_to_10m_train_supervised_tokens"],
            "language_floor_gaps": dict(reaudit["closure"]["language_floor_gaps"]),
            "source_cap_violations": list(reaudit["closure"]["source_cap_violations"]),
            "task_floor_gaps": dict(reaudit["closure"]["task_floor_gaps"]),
        },
        "decisions": {
            "conditional_reserves": [dict(row) for row in plan["conditional_reserves"]],
            "exclusions": [dict(row) for row in plan["exclusions"]],
            "selected_lanes": lanes,
        },
        "minimum_accepted_closure": dict(plan["minimum_accepted_closure"]),
        "network_accounting": {
            "bulk_dataset_payload_bytes": 0,
            "maximum_network_bytes": plan["limits"]["maximum_network_bytes"],
            "metadata_only": True,
            "official_metadata_resources_reviewed": len(plan["official_evidence"]),
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "official_evidence": [dict(row) for row in plan["official_evidence"]],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "dataset_payload_downloaded": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "gpu_used": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "projected_mix": {
            "added_train_supervised_tokens": sum(
                row["target_retained_train_supervised_tokens"] for row in lanes
            ),
            "all_language_floors_met": not any(language_gaps.values()),
            "all_source_caps_met": not source_violations,
            "all_source_family_caps_met": not family_violations,
            "all_task_floors_met": not any(task_gaps.values()),
            "buffer_above_minimum_task_closure": (
                EXPECTED_TARGET_TOKENS - EXPECTED_MINIMUM_TASK_CLOSURE
            ),
            "buffer_above_10m_floor": projected_total - 10_000_000,
            "language_floor_gaps": language_gaps,
            "source_cap": SOURCE_CAP,
            "source_cap_violations": source_violations,
            "source_family_cap_violations": family_violations,
            "source_family_shares": family_shares,
            "source_family_train_supervised_tokens": family_tokens,
            "source_shares": source_shares,
            "source_train_supervised_tokens": source_tokens,
            "task_floor_gaps": task_gaps,
            "train_by_language_mode": projected_languages,
            "train_by_task": projected_tasks,
            "train_supervised_tokens": projected_total,
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "first_party_generation_provider_terms_and_cost_gate_not_complete",
            "first_party_generation_and_curation_not_authorized_or_run",
            "candidate_registry_successor_not_materialized",
            "full_mix_cross_source_reaudit_not_complete",
            "frozen_validation_mix_not_sealed",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_gpu_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_v0_14_10m_mix_closure_source_selection_no_payload_or_training",
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_mix_closure_source_selection_report_v2(report)
    return report


def validate_mix_closure_source_selection_report_v2(
    report: Mapping[str, Any],
) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("v0.14 mix-closure source-selection report schema changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("v0.14 mix-closure source-selection report retained raw text")
    production = report.get("production", {})
    if any(production.values()):
        raise ValueError("v0.14 mix-closure source-selection authorized production")
    projected = report.get("projected_mix", {})
    if projected.get("added_train_supervised_tokens") != EXPECTED_TARGET_TOKENS:
        raise ValueError("v0.14 mix-closure retained target changed")
    if projected.get("train_supervised_tokens") != EXPECTED_PROJECTED_TOKENS:
        raise ValueError("v0.14 mix-closure projected total changed")
    for field in (
        "all_language_floors_met",
        "all_source_caps_met",
        "all_source_family_caps_met",
        "all_task_floors_met",
    ):
        if projected.get(field) is not True:
            raise ValueError(f"v0.14 mix-closure projection failed: {field}")
    if report.get("network_accounting", {}).get("bulk_dataset_payload_bytes") != 0:
        raise ValueError("v0.14 mix-closure preflight downloaded dataset payload")


def build_mix_closure_source_selection_verification_v2(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_mix_closure_source_selection_report_v2(report)
    projected = report["projected_mix"]
    verification: dict[str, Any] = {
        "checks": {
            "all_task_and_language_floors_close_with_buffer": True,
            "conditional_and_excluded_sources_not_counted": True,
            "current_v0_14_reaudit_and_registry_bindings_verified": True,
            "official_rights_and_provenance_evidence_is_metadata_only": True,
            "per_source_and_conservative_family_caps_are_below_0_35": True,
            "raw_training_text_absent": True,
            "zero_bulk_payload_api_gpu_registry_seal_or_sft_actions": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "projected_train_supervised_tokens": projected["train_supervised_tokens"],
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_v0_14_10m_mix_closure_source_selection_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(verification, "verification_sha256")
    return verification
