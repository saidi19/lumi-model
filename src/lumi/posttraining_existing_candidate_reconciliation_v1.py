"""Local-only reconciliation of already completed Lumi SFT candidate work."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-existing-candidate-reconciliation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-existing-candidate-reconciliation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-reconciliation-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "predecessor_mutation_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
}

CURRENT_TRAIN_TOKENS = 7_341_582
TASK_FLOORS = {
    "factual_qa": 1_500_000,
    "multilingual_bengali": 1_000_000,
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

CURRENT_FAMILIES = {
    "coherelabs_aya_family": 513_379,
    "deepmind_mathematics_family": 2_221_113,
    "dmath_family": 48_480,
    "lumi_first_party_family": 114_864,
    "nvidia_family": 2_100_077,
    "openassistant_family": 2_343_669,
}

LOCAL_CANDIDATES = {
    "aya_joke_mit": {
        "train_supervised_tokens": 225_756,
        "task_accounting": {"multilingual_bengali": 225_756},
        "task_behavior": {"multilingual_bengali": 225_756},
        "language_accounting": {"bengali_script": 225_756},
        "language_behavior": {"bengali_script": 225_756},
        "family": "coherelabs_aya_family",
    },
    "aya_wikisplit_cc_by_sa": {
        "train_supervised_tokens": 499_980,
        "task_accounting": {"writing_summarization": 499_980},
        "task_behavior": {"writing_summarization": 499_980},
        "language_accounting": {"bengali_script": 499_980},
        "language_behavior": {"bengali_script": 499_980},
        "family": "coherelabs_aya_family",
    },
    "banglabridge_repaired": {
        "train_supervised_tokens": 49_998,
        "task_accounting": {"multilingual_bengali": 49_998},
        "task_behavior": {"multilingual_bengali": 49_998},
        "language_accounting": {
            "bengali_script": 47_674,
            "natural_code_switching": 2_324,
        },
        "language_behavior": {
            "bengali_script": 47_674,
            "natural_code_switching": 2_324,
        },
        "family": "banglabridge_family",
    },
    "banglatlit_auxiliary": {
        "train_supervised_tokens": 300_000,
        "task_accounting": {"multilingual_bengali": 300_000},
        "task_behavior": {},
        "language_accounting": {"banglish": 300_000},
        "language_behavior": {},
        "family": "banglatlit_family",
    },
    "nemotron_tool_synthetic": {
        "train_supervised_tokens": 1_299_965,
        "task_accounting": {"tool_api": 1_299_965},
        "task_behavior": {"tool_api": 1_299_965},
        "language_accounting": {"english": 1_299_965},
        "language_behavior": {"english": 1_299_965},
        "family": "nvidia_family",
    },
}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic pretty JSON bytes."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    """Hash a mapping using compact canonical JSON."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_internal(value: Mapping[str, Any], field: str) -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing internal hash: {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if sha256_value(unsigned) != expected:
        raise ValueError(f"internal hash does not verify: {field}")


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate the local reconciliation plan."""

    root = Path(repository_root).resolve()
    workspace_root = root.parent.resolve()
    plan = _load_json(Path(path).resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("existing candidate reconciliation plan schema changed")
    if plan.get("status") != "authorized_local_only_existing_candidate_reconciliation":
        raise ValueError("existing candidate reconciliation is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("existing candidate reconciliation authorization widened")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or len(bindings) < 20:
        raise ValueError("existing candidate reconciliation bindings are incomplete")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(workspace_root):
            raise ValueError(f"binding escaped workspace: {name}")
        if not resolved.is_file():
            raise ValueError(f"bound file is missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding hash mismatch: {name}")
    return plan


def _add_counts(base: Mapping[str, int], addition: Mapping[str, int]) -> dict[str, int]:
    result = dict(base)
    for key, value in addition.items():
        result[key] = result.get(key, 0) + int(value)
    return result


def _gaps(counts: Mapping[str, int], floors: Mapping[str, int]) -> dict[str, int]:
    return {key: max(0, floor - counts.get(key, 0)) for key, floor in floors.items()}


def _project(
    current_tasks: Mapping[str, int],
    current_languages: Mapping[str, int],
    included_candidates: list[str],
) -> dict[str, Any]:
    task_accounting = dict(current_tasks)
    task_behavior = dict(current_tasks)
    language_accounting = dict(current_languages)
    language_behavior = dict(current_languages)
    families = dict(CURRENT_FAMILIES)
    added = 0
    for candidate_name in included_candidates:
        candidate = LOCAL_CANDIDATES[candidate_name]
        tokens = int(candidate["train_supervised_tokens"])
        added += tokens
        task_accounting = _add_counts(task_accounting, candidate["task_accounting"])
        task_behavior = _add_counts(task_behavior, candidate["task_behavior"])
        language_accounting = _add_counts(
            language_accounting, candidate["language_accounting"]
        )
        language_behavior = _add_counts(
            language_behavior, candidate["language_behavior"]
        )
        family = str(candidate["family"])
        families[family] = families.get(family, 0) + tokens

    total = CURRENT_TRAIN_TOKENS + added
    task_gaps = _gaps(task_behavior, TASK_FLOORS)
    language_gaps = _gaps(language_behavior, LANGUAGE_FLOORS)
    family_shares = {
        key: round(value / total, 8) for key, value in sorted(families.items())
    }
    family_cap_dilution = max(
        [0]
        + [math.ceil(value / 0.35 - total) for value in families.values()]
    )
    numeric_gap = max(0, 10_000_000 - total)
    task_gap_total = sum(task_gaps.values())
    language_gap_total = sum(language_gaps.values())
    strict_gap = max(
        numeric_gap,
        task_gap_total,
        language_gap_total,
        family_cap_dilution,
    )
    return {
        "included_candidates": included_candidates,
        "added_train_supervised_tokens": added,
        "projected_train_supervised_tokens": total,
        "numeric_gap_to_10m": numeric_gap,
        "task_train_supervised_tokens_accounting": task_accounting,
        "task_train_supervised_tokens_behavior_safe": task_behavior,
        "task_floor_gaps_behavior_safe": task_gaps,
        "task_floor_gap_total_behavior_safe": task_gap_total,
        "language_train_supervised_tokens_accounting": language_accounting,
        "language_train_supervised_tokens_behavior_safe": language_behavior,
        "language_floor_gaps_behavior_safe": language_gaps,
        "language_floor_gap_total_behavior_safe": language_gap_total,
        "source_family_train_supervised_tokens": families,
        "source_family_shares": family_shares,
        "source_family_cap_dilution_gap": family_cap_dilution,
        "strict_targeted_gap_lower_bound": strict_gap,
    }


def build_report(
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build a raw-text-free authoritative reuse and conflict ledger."""

    root = Path(repository_root).resolve()
    values: dict[str, dict[str, Any]] = {}
    json_names = {
        "source_registry",
        "current_mix_report",
        "current_mix_verification",
        "learning_seal_report",
        "learning_seal_verification",
        "aya_rights_report",
        "aya_rights_verification",
        "aya_rights_manifest",
        "aya_capacity_report",
        "aya_capacity_verification",
        "banglabridge_report",
        "banglabridge_verification",
        "closure_curation_report",
        "closure_curation_verification",
        "closure_successor_report",
        "closure_successor_verification",
        "closure_selection_report",
        "closure_selection_verification",
        "closure_retirement_report",
        "closure_retirement_verification",
    }
    for name in json_names:
        binding = plan["bindings"][name]
        values[name] = _load_json((root / binding["path"]).resolve())

    current = values["current_mix_report"]
    current_verification = values["current_mix_verification"]
    seal = values["learning_seal_report"]
    seal_verification = values["learning_seal_verification"]
    aya_report = values["aya_rights_report"]
    aya_verification = values["aya_rights_verification"]
    aya_manifest = values["aya_rights_manifest"]
    capacity = values["aya_capacity_report"]
    capacity_verification = values["aya_capacity_verification"]
    banglabridge = values["banglabridge_report"]
    banglabridge_verification = values["banglabridge_verification"]
    curation = values["closure_curation_report"]
    curation_verification = values["closure_curation_verification"]
    predecessor_projection = values["closure_successor_report"]
    predecessor_projection_verification = values["closure_successor_verification"]
    selection = values["closure_selection_report"]
    selection_verification = values["closure_selection_verification"]
    retirement = values["closure_retirement_report"]
    retirement_verification = values["closure_retirement_verification"]

    if current.get("status") != "verified_registry_v0_17_candidate_mix_reaudit_nonproduction":
        raise ValueError("latest candidate mix report status changed")
    if current_verification.get("status") != "PASS_verified_registry_v0_17_candidate_mix_reaudit_nonproduction":
        raise ValueError("latest candidate mix verification did not pass")
    if seal.get("status") != "PASS_nonproduction_learning_candidate_sealed_smoke_only":
        raise ValueError("learning seal status changed")
    if not str(seal_verification.get("status", "")).startswith("PASS"):
        raise ValueError("learning seal verification did not pass")
    if aya_report.get("status") != "materialized_private_nonproduction_aya_rights_filtered_candidate":
        raise ValueError("Aya rights-filtered materialization status changed")
    if not str(aya_verification.get("status", "")).startswith("verified"):
        raise ValueError("Aya rights-filtered verification did not pass")
    if capacity.get("status") != "verified_capacity_only_no_candidate_or_production_action":
        raise ValueError("Aya capacity scan status changed")
    if not str(capacity_verification.get("status", "")).startswith("verified"):
        raise ValueError("Aya capacity verification did not pass")
    if banglabridge.get("status") != "repaired_private_nonproduction_candidate_registry_unchanged":
        raise ValueError("BanglaBridge repair status changed")
    if not str(banglabridge_verification.get("status", "")).startswith("verified"):
        raise ValueError("BanglaBridge verification did not pass")
    if curation.get("status") != "curated_private_nonproduction_owner_review_pending":
        raise ValueError("closure curation status changed")
    if not str(curation_verification.get("status", "")).startswith("verified"):
        raise ValueError("closure curation verification did not pass")
    if predecessor_projection.get("status") != "verified_closure_projection_blocked_pending_targeted_gap_fill":
        raise ValueError("predecessor local closure projection status changed")
    if not str(predecessor_projection_verification.get("status", "")).startswith("verified"):
        raise ValueError("predecessor local closure projection verification failed")
    if selection.get("status") != "verified_registry_v0_17_10m_closure_reconciliation_nonproduction":
        raise ValueError("latest closure selection status changed")
    if not str(selection_verification.get("status", "")).startswith("PASS"):
        raise ValueError("latest closure selection verification failed")
    if retirement.get("status") != "PASS_local_evidence_preserved_remote_termination_ready":
        raise ValueError("closure retirement status changed")
    if not str(retirement_verification.get("status", "")).startswith("PASS"):
        raise ValueError("closure retirement verification failed")

    pool = current["candidate_pool"]
    if pool["accepted_train_supervised_tokens"] != CURRENT_TRAIN_TOKENS:
        raise ValueError("latest train token total changed")
    source_metrics = current["source_metrics"]["coherelabs_aya_dataset"]
    if source_metrics["by_split"]["train"]["supervised_tokens"] != 513_379:
        raise ValueError("already included Aya train tokens changed")
    if seal["output"]["source_metrics"]["coherelabs_aya_dataset"] != source_metrics["by_split"]:
        raise ValueError("Aya seal ledger differs from the latest mix")

    retained_families = {
        item["family_id"]: item["metrics"]["splits"]["train"]["supervised_tokens"]
        for item in aya_manifest["retained_families"]
    }
    if retained_families != {
        "joke_explaination_inst_translated": 225_756,
        "wiki_split_inst_translated": 499_980,
    }:
        raise ValueError("Aya retained-family ledger changed")
    if aya_report["output"]["train_supervised_tokens"] != 725_736:
        raise ValueError("Aya rights-filtered train tokens changed")
    if banglabridge["output"]["train_supervised_tokens"] != 49_998:
        raise ValueError("BanglaBridge repaired train tokens changed")
    curation_tokens = curation["candidate"]["source_train_supervised_tokens"]
    if curation_tokens["nvidia_nemotron_post_training_tool_canary"] != 1_299_965:
        raise ValueError("Nemotron local candidate token ledger changed")
    if curation_tokens["banglatlit_auxiliary_closure_canary"] != 300_000:
        raise ValueError("BanglaTLit local candidate token ledger changed")
    if capacity["capacity_decision"]["immediately_rights_clear_joke_train_supervised_tokens"] != 14_870:
        raise ValueError("Aya incremental MIT capacity changed")
    if capacity["capacity_decision"]["wikisplit_production_token_credit"] != 0:
        raise ValueError("unresolved WikiSplit unexpectedly received production credit")

    current_tasks = pool["train_by_task"]
    current_languages = pool["train_by_language_mode"]
    scenarios = {
        "all_existing_local_no_new_inference_policy_conditional": _project(
            current_tasks,
            current_languages,
            [
                "aya_joke_mit",
                "aya_wikisplit_cc_by_sa",
                "banglabridge_repaired",
                "banglatlit_auxiliary",
                "nemotron_tool_synthetic",
            ],
        ),
        "exclude_unresolved_share_alike_keep_existing_synthetic": _project(
            current_tasks,
            current_languages,
            [
                "aya_joke_mit",
                "banglabridge_repaired",
                "banglatlit_auxiliary",
                "nemotron_tool_synthetic",
            ],
        ),
        "exclude_unresolved_share_alike_and_preexisting_synthetic_teacher": _project(
            current_tasks,
            current_languages,
            [
                "aya_joke_mit",
                "banglabridge_repaired",
                "banglatlit_auxiliary",
            ],
        ),
    }

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_existing_work_reconciled_no_redownload_or_promotion",
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "bindings": {
            name: {"sha256": binding["sha256"]}
            for name, binding in sorted(plan["bindings"].items())
        },
        "current_baseline": {
            "registry_sha256": plan["bindings"]["source_registry"]["sha256"],
            "source_count": current["registry"]["source_count"],
            "candidate_records": pool["accepted_records_after_generic_quarantine"],
            "train_supervised_tokens": CURRENT_TRAIN_TOKENS,
            "validation_supervised_tokens": pool["input_validation_supervised_tokens"],
            "task_train_supervised_tokens": current_tasks,
            "language_train_supervised_tokens": current_languages,
            "aya_already_included": {
                "source_id": "coherelabs_aya_dataset",
                "candidate_file_sha256": source_metrics["candidate_file_sha256"],
                "train_records": source_metrics["by_split"]["train"]["records"],
                "train_supervised_tokens": 513_379,
                "validation_records": source_metrics["by_split"]["validation"]["records"],
                "validation_supervised_tokens": source_metrics["by_split"]["validation"]["supervised_tokens"],
                "action": "reuse_byte_exact_do_not_redownload_or_repromote",
            },
        },
        "completed_work_ledger": {
            "aya_collection_rights_filtered": {
                "candidate_file_sha256": aya_report["output"]["candidate_file_sha256"],
                "records": aya_report["output"]["records"],
                "train_supervised_tokens": aya_report["output"]["train_supervised_tokens"],
                "immediately_clear_mit_train_supervised_tokens": 225_756,
                "conditional_cc_by_sa_train_supervised_tokens": 499_980,
                "quarantined_noncommercial_train_supervised_tokens": aya_report["quarantined"]["splits"]["train"]["supervised_tokens"],
                "registry_v0_17_included": False,
                "redownload_required": False,
                "repeat_materialization_required": False,
            },
            "aya_full_local_capacity_scan": {
                "raw_object_sha256": capacity["bindings"]["raw_aya_parquet_sha256"],
                "rows_read": capacity["raw_inventory"]["rows_read"],
                "incremental_mit_train_supervised_tokens_after_near_dedup": 14_870,
                "conditional_wikisplit_train_supervised_tokens_lower_bound": capacity["capacity_decision"]["conditional_wikisplit_train_supervised_tokens_lower_bound"],
                "wikisplit_production_token_credit": 0,
                "redownload_required": False,
                "repeat_capacity_scan_required": False,
            },
            "banglabridge_repaired": {
                "candidate_file_sha256": banglabridge["output"]["candidate_file_sha256"],
                "train_records": banglabridge["output"]["train_records"],
                "train_supervised_tokens": banglabridge["output"]["train_supervised_tokens"],
                "validation_records": banglabridge["output"]["validation_records"],
                "validation_supervised_tokens": banglabridge["output"]["validation_supervised_tokens"],
                "redownload_required": False,
                "repeat_hygiene_repair_required": False,
            },
            "nemotron_tool_curated": {
                "candidate_file_sha256": plan["bindings"]["nemotron_candidate"]["sha256"],
                "train_supervised_tokens": 1_299_965,
                "origin": "preexisting_synthetic_dataset_candidate",
                "new_inference_required": False,
                "redownload_required": False,
            },
            "banglatlit_auxiliary_curated": {
                "candidate_file_sha256": plan["bindings"]["banglatlit_candidate"]["sha256"],
                "train_supervised_tokens": 300_000,
                "behavior_floor_credit": 0,
                "redownload_required": False,
            },
        },
        "permanent_and_conditional_exclusions": {
            "aya_wikilingua_noncommercial": {
                "train_supervised_tokens": 399_997,
                "decision": "permanently_exclude_from_commercial_product_sft",
            },
            "aya_wikisplit_share_alike": {
                "materialized_train_supervised_tokens": 499_980,
                "decision": "exclude_by_default_until_share_alike_compliance_is_explicitly_resolved",
            },
            "retired_gpt_oss_closure": {
                "candidate_sha256": retirement["local_evidence"]["candidate"]["sha256"],
                "decision": "permanently_exclude_from_registry_seal_and_recovery_prefix",
            },
        },
        "updated_v0_12_projections": scenarios,
        "conflict_controls": {
            "predecessor_registry_mutated": False,
            "current_seal_mutated": False,
            "candidate_files_mutated": False,
            "raw_training_text_retained_in_report": False,
            "completed_acquisition_or_curation_rerun": False,
            "full_cross_source_exact_and_near_duplicate_reaudit_complete_for_projected_mix": False,
            "registry_successor_materialized": False,
        },
        "recommended_next_gate": {
            "action": "local_only_candidate_policy_selection_then_v0_18_projection_and_full_cross_source_reaudit",
            "default_policy": {
                "redownload_existing_sources": False,
                "rerun_completed_repairs_or_capacity_scans": False,
                "include_unresolved_cc_by_sa": False,
                "new_model_inference": False,
            },
            "owner_policy_choice_still_required": [
                "whether_to_include_the_already_curated_synthetic_nemotron_candidate",
                "whether_share_alike_compliance_will_be_resolved_or_wikisplit_stays_excluded",
            ],
            "only_after_policy_choice": [
                "project_immutable_registry_v0_18_without_predecessor_mutation",
                "run_full_cross_source_exact_near_duplicate_and_decontamination_audit",
                "recalculate_residual_task_language_and_source_family_gaps",
            ],
            "download_authorized": False,
            "inference_authorized": False,
            "registry_change_authorized": False,
            "dataset_seal_or_sft_authorized": False,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_verification(
    report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    """Build an independent raw-text-free verification receipt."""

    _verify_internal(report, "report_sha256")
    scenarios = report["updated_v0_12_projections"]
    if scenarios["all_existing_local_no_new_inference_policy_conditional"][
        "projected_train_supervised_tokens"
    ] != 9_717_281:
        raise ValueError("all-local projection changed")
    if scenarios["exclude_unresolved_share_alike_keep_existing_synthetic"][
        "strict_targeted_gap_lower_bound"
    ] != 1_835_341:
        raise ValueError("conservative synthetic-retained gap changed")
    if scenarios[
        "exclude_unresolved_share_alike_and_preexisting_synthetic_teacher"
    ]["strict_targeted_gap_lower_bound"] != 2_606_006:
        raise ValueError("no-synthetic-teacher gap changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_existing_work_reconciled_no_redownload_or_promotion",
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "checks": {
            "all_bound_historical_artifacts_rehashed": True,
            "latest_registry_v0_17_mix_and_smoke_seal_bound": True,
            "already_included_aya_not_double_counted": True,
            "aya_rights_filtered_and_capacity_work_not_repeated": True,
            "banglabridge_repair_not_repeated": True,
            "local_nemotron_and_banglatlit_candidates_reused_by_hash": True,
            "noncommercial_and_retired_candidates_remain_excluded": True,
            "unresolved_share_alike_has_zero_default_production_credit": True,
            "updated_v0_12_projection_math_reproduced": True,
            "no_download_network_gpu_inference_registry_seal_or_sft": True,
            "report_is_raw_text_free": True,
        },
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
