"""Rights-aware diversified source preflight for the Lumi 4B SFT mix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_preflight import sha256_value
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-diversified-source-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-diversified-source-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-diversified-source-preflight-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_payload_download_authorized": False,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "external_distribution_authorized": False,
    "local_raw_text_free_preflight_authorized": True,
    "official_metadata_lookup_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "diversified_rights_clear_non_math_source_preflight_only",
    "source_promotion_authorized": False,
    "unrelated_data_access_authorized": False,
}

EXPECTED_LIMITS = {
    "dataset_payload_bytes": 0,
    "maximum_metadata_pages_reviewed": 32,
    "maximum_runtime_minutes": 90,
    "official_external_sources_reviewed": 8,
}

EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-diversified-source-preflight-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-diversified-source-preflight-verification-v0.1.json"
    ),
}

EXPECTED_BINDINGS = frozenset(
    {
        "candidate_mix_reaudit",
        "candidate_mix_reaudit_verification",
        "data_contract",
        "implementation",
        "prior_shortlist_plan",
        "prior_shortlist_report",
        "prior_shortlist_verification",
        "runner",
        "source_registry",
        "test",
        "verifier",
    }
)

EXPECTED_CURRENT_MIX = {
    "accepted_train_supervised_tokens": 3_271_793,
    "gap_to_10m_train_supervised_tokens": 6_728_207,
    "train_by_language_mode": {
        "banglish": 6_345,
        "bengali_script": 532_033,
        "english": 2_728_599,
        "natural_code_switching": 4_816,
    },
    "train_by_task": {
        "coding": 54_439,
        "conversation": 201_762,
        "factual_qa": 141_516,
        "multilingual_bengali": 524_598,
        "reasoning_math_science": 2_269_713,
        "safety_helpfulness": 10_421,
        "tool_api": 9_413,
        "writing_summarization": 59_931,
    },
}

TASK_FLOORS = {
    "coding": 2_000_000,
    "conversation": 500_000,
    "factual_qa": 1_500_000,
    "multilingual_bengali": 1_000_000,
    "reasoning_math_science": 2_000_000,
    "safety_helpfulness": 800_000,
    "tool_api": 800_000,
    "writing_summarization": 1_200_000,
}

LANGUAGE_FLOORS = {
    "banglish": 800_000,
    "bengali_script": 1_000_000,
    "english": 7_800_000,
    "natural_code_switching": 400_000,
}

PRIMARY_LANE_IDS = {
    "lumi_first_party_multilingual_safety_tool_v0_2",
    "nvidia_opencodeinstruct",
    "openassistant_oasst2",
}

CONDITIONAL_DECISIONS = {
    "databricks_dolly_15k": "reserve_only_cc_by_sa_model_licensing_review_required",
    "nvidia_aegis_safety_v2": (
        "reserve_only_mixed_upstream_prompt_provenance_review_required"
    ),
    "nvidia_helpsteer2": (
        "reserve_only_sharegpt_prompt_provenance_review_required"
    ),
    "salesforce_xlam_function_calling_60k": (
        "conditional_owner_terms_acceptance_and_exact_revision_required"
    ),
}

EXCLUSION_DECISIONS = {
    "anthropic_hh_rlhf": "excluded_dataset_card_warns_against_dialogue_sft",
    "huggingfaceh4_no_robots": "excluded_noncommercial_license",
}

EXPECTED_NEXT_GATE = {
    "action": "bounded_cpu_only_opencodeinstruct_single_shard_acquisition",
    "authorization_required": True,
    "dataset_id": "nvidia/OpenCodeInstruct",
    "exact_objects": [
        {
            "git_blob_sha1": "8226aea38e6986a3a243ea161739ed3ad45ac600",
            "path": "README.md",
            "size_bytes": 4_377,
        },
        {
            "lfs_sha256": (
                "2839de0d6c4d1691710dbc63f43cd816e7c02a943f8e2c07087bbf5fa26172ce"
            ),
            "path": "data/train-00007-of-00050.parquet",
            "size_bytes": 113_867_035,
        },
    ],
    "maximum_decoded_bytes": 536_870_912,
    "maximum_network_bytes": 134_217_728,
    "maximum_retained_train_supervised_tokens": 2_300_000,
    "maximum_runtime_minutes": 180,
    "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
    "target_retained_train_supervised_tokens": 2_100_000,
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256_value(payload)


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


def _verify_official_evidence(plan: Mapping[str, Any]) -> None:
    rows = plan.get("official_evidence")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("official evidence inventory changed")
    allowed_prefixes = (
        "https://huggingface.co/datasets/",
        "https://github.com/",
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
        if any(
            not isinstance(url, str) or not url.startswith(allowed_prefixes)
            for url in urls
        ):
            raise ValueError("non-official evidence URL present")


def _verify_lanes(plan: Mapping[str, Any]) -> None:
    primary = _source_map(plan.get("primary_lanes"), label="primary_lanes")
    if set(primary) != PRIMARY_LANE_IDS:
        raise ValueError("primary source lane set changed")
    for source_id, row in primary.items():
        target = row.get("planning_train_supervised_tokens")
        task_allocation = row.get("task_allocation")
        language_allocation = row.get("language_allocation")
        if not isinstance(target, int) or target <= 0:
            raise ValueError(f"invalid target for {source_id}")
        if not isinstance(task_allocation, Mapping) or sum(task_allocation.values()) != target:
            raise ValueError(f"task allocation mismatch for {source_id}")
        if (
            not isinstance(language_allocation, Mapping)
            or sum(language_allocation.values()) != target
        ):
            raise ValueError(f"language allocation mismatch for {source_id}")

    conditional = _source_map(
        plan.get("conditional_lanes"), label="conditional_lanes"
    )
    if set(conditional) != set(CONDITIONAL_DECISIONS):
        raise ValueError("conditional source lane set changed")
    for source_id, decision in CONDITIONAL_DECISIONS.items():
        if conditional[source_id].get("decision") != decision:
            raise ValueError(f"conditional decision changed: {source_id}")

    exclusions = _source_map(plan.get("exclusions"), label="exclusions")
    if set(exclusions) != set(EXCLUSION_DECISIONS):
        raise ValueError("exclusion source set changed")
    for source_id, decision in EXCLUSION_DECISIONS.items():
        if exclusions[source_id].get("decision") != decision:
            raise ValueError(f"exclusion decision changed: {source_id}")


def load_diversified_source_preflight_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_metadata_only_diversified_source_preflight",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"diversified source preflight changed: {field}")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("diversified source preflight binding set changed")
    resolved = {
        name: resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    _verify_lanes(plan)
    _verify_official_evidence(plan)

    mix = _load_json(resolved["candidate_mix_reaudit"])
    mix_verification = _load_json(resolved["candidate_mix_reaudit_verification"])
    prior_report = _load_json(resolved["prior_shortlist_report"])
    prior_verification = _load_json(resolved["prior_shortlist_verification"])
    for value, field in (
        (mix, "report_sha256"),
        (mix_verification, "verification_sha256"),
        (prior_report, "report_sha256"),
        (prior_verification, "verification_sha256"),
    ):
        _verify_internal_sha(value, field)
    pool = mix.get("candidate_pool", {})
    for field, expected_value in EXPECTED_CURRENT_MIX.items():
        if pool.get(field) != expected_value:
            raise ValueError(f"candidate mix changed: {field}")
    closure = mix.get("closure", {})
    if closure.get("source_cap") != 0.35:
        raise ValueError("source cap changed")
    if closure.get("task_floor_gaps", {}).get("reasoning_math_science") != 0:
        raise ValueError("math floor is no longer closed")
    if prior_report.get("production", {}).get("dataset_payload_downloaded") is not False:
        raise ValueError("prior shortlist unexpectedly downloaded payload")
    return plan


def _sum_allocations(
    current: Mapping[str, int],
    primary_lanes: list[Mapping[str, Any]],
    *,
    allocation_key: str,
) -> dict[str, int]:
    result = dict(current)
    for lane in primary_lanes:
        allocation = lane[allocation_key]
        for key, value in allocation.items():
            result[key] = result.get(key, 0) + int(value)
    return dict(sorted(result.items()))


def _project_source_shares(
    current_source_tokens: Mapping[str, int],
    primary_lanes: list[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, float], int]:
    totals = dict(current_source_tokens)
    for lane in primary_lanes:
        totals[lane["source_id"]] = (
            totals.get(lane["source_id"], 0)
            + lane["planning_train_supervised_tokens"]
        )
    total = sum(totals.values())
    shares = {key: round(value / total, 8) for key, value in sorted(totals.items())}
    return dict(sorted(totals.items())), shares, total


def rebuild_diversified_source_preflight(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    bindings = plan["bindings"]
    mix_path = resolve_binding(
        repository_root, bindings["candidate_mix_reaudit"], name="candidate_mix_reaudit"
    )
    mix = _load_json(mix_path)
    pool = mix["candidate_pool"]
    primary = [dict(row) for row in plan["primary_lanes"]]
    projected_tasks = _sum_allocations(
        pool["train_by_task"], primary, allocation_key="task_allocation"
    )
    projected_languages = _sum_allocations(
        pool["train_by_language_mode"],
        primary,
        allocation_key="language_allocation",
    )
    current_source_tokens = {
        source_id: metrics["by_split"]["train"]["supervised_tokens"]
        for source_id, metrics in mix["source_metrics"].items()
    }
    source_tokens, source_shares, projected_total = _project_source_shares(
        current_source_tokens, primary
    )
    projected_task_gaps = {
        key: max(0, floor - projected_tasks.get(key, 0))
        for key, floor in TASK_FLOORS.items()
    }
    projected_language_gaps = {
        key: max(0, floor - projected_languages.get(key, 0))
        for key, floor in LANGUAGE_FLOORS.items()
    }
    projected_source_cap_violations = [
        source_id for source_id, share in source_shares.items() if share > 0.35
    ]
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": dict(bindings),
        "current_mix": {
            **EXPECTED_CURRENT_MIX,
            "source_cap": 0.35,
            "source_cap_violations": list(mix["closure"]["source_cap_violations"]),
        },
        "decisions": {
            "conditional_lanes": [dict(row) for row in plan["conditional_lanes"]],
            "exclusions": [dict(row) for row in plan["exclusions"]],
            "primary_lanes": primary,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "official_evidence": [dict(row) for row in plan["official_evidence"]],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "dataset_payload_downloaded": False,
            "dataset_sealed": False,
            "external_model_api_used": False,
            "gpu_used": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "projected_mix": {
            "all_language_floors_met": not any(projected_language_gaps.values()),
            "all_source_caps_met": not projected_source_cap_violations,
            "all_task_floors_met": not any(projected_task_gaps.values()),
            "language_floor_gaps": projected_language_gaps,
            "source_cap": 0.35,
            "source_cap_violations": projected_source_cap_violations,
            "source_shares": source_shares,
            "source_train_supervised_tokens": source_tokens,
            "task_floor_gaps": projected_task_gaps,
            "train_by_language_mode": projected_languages,
            "train_by_task": projected_tasks,
            "train_supervised_tokens": projected_total,
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "separate_authorization_for_each_dataset_payload_acquisition",
            "opencode_quality_execution_pii_decontamination_and_attribution_not_complete",
            "oasst2_tree_selection_quality_pii_decontamination_and_attribution_not_complete",
            "first_party_multilingual_generation_contract_and_owner_review_not_complete",
            "xlam_terms_not_accepted_and_exact_revision_not_pinned",
            "aya_near_duplicate_canonical_selection_not_materialized",
            "full_mix_cross_source_deduplication_not_materialized",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_gpu_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_diversified_non_math_plan_no_payload_or_training",
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_diversified_source_preflight_report(report)
    return report


def validate_diversified_source_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("diversified source report schema changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("diversified source report retained raw text")
    production = report.get("production", {})
    if any(production.values()):
        raise ValueError("diversified source report authorized production action")
    projected = report.get("projected_mix", {})
    if projected.get("train_supervised_tokens", 0) < 10_000_000:
        raise ValueError("diversified source plan does not reach 10M")
    if projected.get("all_task_floors_met") is not True:
        raise ValueError("diversified source plan does not close task floors")
    if projected.get("all_language_floors_met") is not True:
        raise ValueError("diversified source plan does not close language floors")
    if projected.get("all_source_caps_met") is not True:
        raise ValueError("diversified source plan does not close source cap")


def build_diversified_source_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_diversified_source_preflight_report(report)
    verification: dict[str, Any] = {
        "next_gate": dict(plan["next_gate"]),
        "plan_file_sha256": report["plan_file_sha256"],
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_sha256": report_file_sha256,
        "report_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_diversified_source_preflight_nonproduction",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    return verification


def stable_evidence_digest(rows: list[Mapping[str, Any]]) -> str:
    """Return a stable digest for a metadata evidence inventory."""

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
