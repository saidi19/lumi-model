"""Fail-closed bulk-source shortlist for the Lumi 4B learning SFT candidate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-bulk-source-shortlist-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-bulk-source-shortlist-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-bulk-source-shortlist-verification.v0.1"

DEEPMIND_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
OPENCODE_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
OPENCODE_CANARY_OBJECT = {
    "lfs_sha256": "2839de0d6c4d1691710dbc63f43cd816e7c02a943f8e2c07087bbf5fa26172ce",
    "path": "data/train-00007-of-00050.parquet",
    "size_bytes": 113_867_035,
}

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "dataset_payload_download_authorized": False,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "local_raw_text_free_preflight_authorized": True,
    "official_metadata_lookup_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "rights_clear_bulk_math_and_coding_source_shortlist_only",
    "source_promotion_authorized": False,
}

EXPECTED_LIMITS = {
    "dataset_payload_bytes": 0,
    "maximum_metadata_network_bytes": 2_097_152,
    "maximum_runtime_minutes": 60,
    "official_sources_reviewed": 5,
}

EXPECTED_OUTPUT = {
    "production_promotion_permitted": False,
    "raw_training_text_in_outputs": False,
    "report_path": "evidence/lumi-4b-sft-bulk-source-shortlist-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-bulk-source-shortlist-verification-v0.1.json"
    ),
}

EXPECTED_BINDINGS = {
    "candidate_mix_reaudit",
    "candidate_mix_reaudit_verification",
    "data_contract",
    "deepmind_canary_report",
    "deepmind_canary_verification",
    "deepmind_promotion_receipt",
    "deepmind_promotion_verification",
    "implementation",
    "missing_mix_proposal",
    "runner",
    "source_registry",
    "test",
    "verifier",
}

EXPECTED_SOURCE_DECISIONS = {
    "bigcode_self_oss_instruct_50k": "blocked_existing_row_level_license_mapping_gap",
    "bigcode_the_stack_dedup": "blocked_gated_terms_and_per_row_license_compliance",
    "deepmind_mathematics_dataset": "recommended_next_separate_local_generation_gate",
    "huggingfacetb_smoltalk": "excluded_mixed_upstream_subsets_and_teacher_provenance",
    "nvidia_opencodeinstruct": "recommended_second_separate_bounded_curation_gate",
}

EXPECTED_GAPS = {
    "accepted_train_supervised_tokens": 1_068_485,
    "coding": 1_945_561,
    "gap_to_10m_train_supervised_tokens": 8_931_515,
    "reasoning_math_science": 2_133_595,
}

EXPECTED_NEXT_GATE = {
    "action": "bounded_cpu_only_deepmind_mathematics_scale_generation",
    "authorization_required": True,
    "maximum_generated_records": 250_000,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 240,
    "official_test_or_test_extra_generation_authorized": False,
    "source_id": "deepmind_mathematics_dataset",
    "target_train_supervised_tokens": {
        "maximum": 2_300_000,
        "minimum": 2_150_000,
        "planning_target": 2_200_000,
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_value(value: Mapping[str, Any], *, excluded_key: str | None = None) -> str:
    payload = dict(value)
    if excluded_key is not None:
        payload.pop(excluded_key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"missing internal hash: {field}")
    if sha256_value(value, excluded_key=field) != observed:
        raise ValueError(f"internal hash mismatch: {field}")


def _resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid binding: {name}")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"binding escapes repository: {name}")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"binding mismatch: {name}")
    return path


def _shortlist_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    shortlist = plan.get("shortlist")
    if not isinstance(shortlist, list):
        raise ValueError("shortlist must be a list")
    mapped = {
        item.get("source_id"): item
        for item in shortlist
        if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
    }
    if set(mapped) != set(EXPECTED_SOURCE_DECISIONS):
        raise ValueError("shortlist source set changed")
    for source_id, decision in EXPECTED_SOURCE_DECISIONS.items():
        if mapped[source_id].get("decision") != decision:
            raise ValueError(f"shortlist decision changed: {source_id}")
    return mapped


def load_bulk_source_shortlist_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "current_priority_gaps": EXPECTED_GAPS,
        "limits": EXPECTED_LIMITS,
        "next_gate": EXPECTED_NEXT_GATE,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_official_metadata_preflight_no_dataset_payload_or_training",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"bulk-source shortlist plan changed: {field}")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("bulk-source shortlist binding set changed")
    resolved = {
        name: _resolve_binding(repository_root, binding, name=name)
        for name, binding in bindings.items()
    }
    shortlist = _shortlist_map(plan)
    deepmind = shortlist["deepmind_mathematics_dataset"]
    if deepmind.get("revision") != DEEPMIND_REVISION:
        raise ValueError("DeepMind revision changed")
    opencode = shortlist["nvidia_opencodeinstruct"]
    if opencode.get("revision") != OPENCODE_REVISION:
        raise ValueError("OpenCodeInstruct revision changed")
    if opencode.get("bounded_canary_object") != OPENCODE_CANARY_OBJECT:
        raise ValueError("OpenCodeInstruct canary object changed")

    official = plan.get("official_evidence")
    if (
        not isinstance(official, list)
        or len(official) != EXPECTED_LIMITS["official_sources_reviewed"]
    ):
        raise ValueError("official evidence inventory changed")
    for item in official:
        if not isinstance(item, Mapping):
            raise ValueError("official evidence row must be an object")
        urls = item.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("official evidence URL missing")
        if any(
            not isinstance(url, str)
            or not url.startswith(("https://github.com/", "https://huggingface.co/", "https://arxiv.org/"))
            for url in urls
        ):
            raise ValueError("non-official evidence URL present")

    reaudit = _load_json(resolved["candidate_mix_reaudit"])
    reaudit_verification = _load_json(resolved["candidate_mix_reaudit_verification"])
    deepmind_canary = _load_json(resolved["deepmind_canary_report"])
    deepmind_verification = _load_json(resolved["deepmind_canary_verification"])
    deepmind_promotion = _load_json(resolved["deepmind_promotion_receipt"])
    deepmind_promotion_verification = _load_json(
        resolved["deepmind_promotion_verification"]
    )
    for value, field in (
        (reaudit, "report_sha256"),
        (reaudit_verification, "verification_sha256"),
        (deepmind_canary, "report_sha256"),
        (deepmind_verification, "verification_sha256"),
        (deepmind_promotion, "receipt_sha256"),
        (deepmind_promotion_verification, "verification_sha256"),
    ):
        _verify_internal_sha(value, field)
    if reaudit.get("status") != (
        "verified_registry_v0_9_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("candidate-mix re-audit status changed")
    if deepmind_canary.get("status") != "verified_cpu_only_nonproduction_adapter_canary":
        raise ValueError("DeepMind canary status changed")
    if deepmind_canary.get("source", {}).get("revision") != DEEPMIND_REVISION:
        raise ValueError("DeepMind canary source binding changed")
    if deepmind_canary.get("canary", {}).get("generation", {}).get("replay_byte_exact") is not True:
        raise ValueError("DeepMind deterministic replay did not pass")

    registry = _load_json(resolved["source_registry"])
    registry_sources = {
        item.get("source_id"): item
        for item in registry.get("sources", [])
        if isinstance(item, Mapping)
    }
    registered_deepmind = registry_sources.get("deepmind_mathematics_dataset")
    if not isinstance(registered_deepmind, Mapping):
        raise ValueError("DeepMind is not registered")
    if registered_deepmind.get("revision") != DEEPMIND_REVISION:
        raise ValueError("registered DeepMind revision changed")
    contract = _load_json(resolved["data_contract"])
    if "cc-by-4.0" not in contract.get("allowed_licenses", []):
        raise ValueError("OpenCodeInstruct license is outside the active allowed list")
    return plan


def rebuild_bulk_source_shortlist(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    bindings = plan["bindings"]
    reaudit = _load_json(
        _resolve_binding(
            repository_root,
            bindings["candidate_mix_reaudit"],
            name="candidate_mix_reaudit",
        )
    )
    deepmind_canary = _load_json(
        _resolve_binding(
            repository_root,
            bindings["deepmind_canary_report"],
            name="deepmind_canary_report",
        )
    )
    train_measurement = deepmind_canary["canary"]["token_measurement"]["by_split"]["train"]
    generated = deepmind_canary["canary"]["generation"]["generated_records"]
    retained = deepmind_canary["canary"]["retained_records"]
    average_tokens = train_measurement["supervised_tokens"] / train_measurement["records"]
    planning_target = plan["next_gate"]["target_train_supervised_tokens"]["planning_target"]
    estimated_retained_rows = math.ceil(planning_target / average_tokens)
    estimated_generated_rows = math.ceil(estimated_retained_rows / (retained / generated))
    projected_after_two_lanes = (
        EXPECTED_GAPS["accepted_train_supervised_tokens"] + planning_target + 2_100_000
    )

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            name: {
                "path": binding["path"],
                "sha256": binding["sha256"],
            }
            for name, binding in sorted(bindings.items())
        },
        "current_mix": {
            "accepted_train_supervised_tokens": reaudit["candidate_pool"][
                "accepted_train_supervised_tokens"
            ],
            "coding_gap": reaudit["closure"]["task_floor_gaps"]["coding"],
            "gap_to_10m_train_supervised_tokens": reaudit["candidate_pool"][
                "gap_to_10m_train_supervised_tokens"
            ],
            "reasoning_math_science_gap": reaudit["closure"]["task_floor_gaps"][
                "reasoning_math_science"
            ],
        },
        "decisions": [dict(item) for item in plan["shortlist"]],
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "official_evidence": [dict(item) for item in plan["official_evidence"]],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_promotions": 0,
            "dataset_payload_downloaded": False,
            "dataset_sealed": False,
            "gpu_used": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "projections": {
            "deepmind_canary_retention_fraction": retained / generated,
            "deepmind_observed_train_supervised_tokens_per_retained_record": average_tokens,
            "deepmind_planning_generated_rows": estimated_generated_rows,
            "deepmind_planning_retained_rows": estimated_retained_rows,
            "deepmind_planning_train_supervised_tokens": planning_target,
            "deepmind_share_at_10m_floor_after_expansion": (
                (train_measurement["supervised_tokens"] + planning_target) / 10_000_000
            ),
            "opencode_curation_retention_cap_train_supervised_tokens": 2_100_000,
            "projected_train_supervised_tokens_after_both_priority_lanes": (
                projected_after_two_lanes
            ),
            "remaining_gap_to_10m_after_both_priority_lanes": (
                10_000_000 - projected_after_two_lanes
            ),
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "separate_owner_authorization_for_deepmind_scale_generation",
            "deepmind_scale_quality_dedup_and_owner_review_not_complete",
            "separate_owner_authorization_for_opencode_exact_shard_acquisition",
            "opencode_row_quality_execution_pii_decontamination_and_attribution_not_complete",
            "factual_writing_safety_tool_and_multilingual_gaps_remain_after_priority_lanes",
            "full_mix_source_cap_and_cross_source_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_gpu_sft_not_authorized",
        ],
        "report_sha256": "",
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_bulk_source_shortlist_deepmind_then_opencode_no_payload_or_training",
    }
    report["report_sha256"] = sha256_value(report, excluded_key="report_sha256")
    return report


def validate_bulk_source_shortlist_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("bulk-source shortlist report schema changed")
    if report.get("status") != (
        "verified_bulk_source_shortlist_deepmind_then_opencode_no_payload_or_training"
    ):
        raise ValueError("bulk-source shortlist report status changed")
    _verify_internal_sha(report, "report_sha256")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("raw training text must not be retained")
    production = report.get("production")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("production action appeared in shortlist report")
    if report.get("next_gate") != EXPECTED_NEXT_GATE:
        raise ValueError("next gate changed")
    if report.get("current_mix") != {
        "accepted_train_supervised_tokens": 1_068_485,
        "coding_gap": 1_945_561,
        "gap_to_10m_train_supervised_tokens": 8_931_515,
        "reasoning_math_science_gap": 2_133_595,
    }:
        raise ValueError("current candidate gaps changed")
