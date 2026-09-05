"""Fail-closed rights and provenance preflight for the Lumi 10M SFT closure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-rights-clear-closure-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-rights-clear-closure-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-rights-clear-closure-preflight-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "bulk_dataset_download_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "official_metadata_lookup_authorized": True,
    "production_promotion_authorized": False,
    "sft_training_authorized": False,
}

EXPECTED_CURRENT_MIX = {
    "accepted_train_supervised_tokens": 7_256_670,
    "gap_to_10m_train_supervised_tokens": 2_743_330,
    "minimum_task_closure_supervised_tokens": 2_966_672,
    "source_cap": 0.35,
    "task_floor_gaps": {
        "coding": 0,
        "conversation": 0,
        "factual_qa": 100_131,
        "multilingual_bengali": 481_837,
        "reasoning_math_science": 0,
        "safety_helpfulness": 789_105,
        "tool_api": 790_186,
        "writing_summarization": 805_413,
    },
    "language_floor_gaps": {
        "banglish": 793_248,
        "bengali_script": 474_145,
        "english": 1_081_282,
        "natural_code_switching": 394_655,
    },
}

EXPECTED_SOURCES = [
    {
        "source_id": "nvidia_nemotron_post_training_v1",
        "repository": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "official_url": (
            "https://huggingface.co/datasets/"
            "nvidia/Nemotron-Post-Training-Dataset-v1"
        ),
        "revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "declared_license": "cc-by-4.0",
        "disposition": "eligible_for_bounded_payload_canary",
        "eligible_splits": ["chat", "tool_calling"],
        "target_retained_train_supervised_tokens_ceiling": 1_550_000,
        "counts_toward_general_assistant_closure": True,
        "required_filters": [
            "retain_only_rows_with_explicit_cc_by_4_0_license",
            "exclude_empty_or_externally_referenced_prompts",
            "require_complete_user_assistant_or_tool_result_sequences",
            "reject_synthetic_tool_results_without_bound_tool_response",
            "quality_safety_pii_secret_dedup_decontamination_and_4k_fit",
        ],
    },
    {
        "source_id": "coherelabs_aya_collection_language_split_bengali",
        "repository": "CohereLabs/aya_collection_language_split",
        "official_url": (
            "https://huggingface.co/datasets/"
            "CohereLabs/aya_collection_language_split"
        ),
        "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        "declared_license": "apache-2.0",
        "disposition": "eligible_for_bounded_payload_canary",
        "eligible_splits": ["bengali/train"],
        "target_retained_train_supervised_tokens_ceiling": 1_300_000,
        "counts_toward_general_assistant_closure": True,
        "required_filters": [
            "reserve_test_and_validation_splits",
            "preserve_constituent_source_provenance",
            "exclude_constituents_without_compatible_upstream_rights",
            "require_natural_bengali_instruction_response_quality",
            "quality_safety_pii_secret_dedup_decontamination_and_4k_fit",
        ],
    },
    {
        "source_id": "aplycaebous_banglatlit",
        "repository": "aplycaebous/BanglaTLit",
        "official_url": "https://huggingface.co/datasets/aplycaebous/BanglaTLit",
        "revision": "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        "declared_license": "mit",
        "disposition": "eligible_auxiliary_transliteration_canary_only",
        "eligible_splits": ["BanglaTLit_train.csv"],
        "target_retained_train_supervised_tokens_ceiling": 300_000,
        "counts_toward_general_assistant_closure": False,
        "required_filters": [
            "reserve_published_test_and_validation_objects",
            "treat_only_as_transliteration_or_orthography_supervision",
            "do_not_label_as_general_chat_safety_tool_or_writing_data",
            "quality_pii_secret_dedup_decontamination_and_4k_fit",
        ],
    },
    {
        "source_id": "subhajitmahata84_banglabridge_instructions",
        "repository": "subhajitmahata84/banglabridge-instructions",
        "official_url": (
            "https://huggingface.co/datasets/"
            "subhajitmahata84/banglabridge-instructions"
        ),
        "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        "declared_license": "cc-by-4.0",
        "disposition": "eligible_small_banglish_quality_anchor_canary",
        "eligible_splits": ["banglish_finetune.adapted.jsonl"],
        "target_retained_train_supervised_tokens_ceiling": 50_000,
        "counts_toward_general_assistant_closure": True,
        "required_filters": [
            "reserve_or_rebuild_a_semantically_isolated_validation_split",
            "verify_original_or_deterministic_template_provenance_per_row",
            "replace_region_specific_emergency_numbers_before_use",
            "native_banglish_quality_owner_review",
            "quality_safety_pii_secret_dedup_decontamination_and_4k_fit",
        ],
    },
]

EXPECTED_REJECTIONS = [
    {
        "repository": "OdiaGenAI/all_combined_bengali_252k",
        "decision": "ineligible_for_product_sft",
        "reason": "declared_cc_by_nc_sa_4_0_is_noncommercial",
    },
    {
        "repository": "kamruzzaman-asif/bangla-instruction-dataset",
        "decision": "ineligible_as_wholesale_compilation",
        "reason": "aggregate_license_does_not_override_restrictive_upstreams",
    },
    {
        "repository": "istiaqfuad/bangla-english-banglish-pairs",
        "decision": "quarantined_from_sft_closure",
        "reason": "contrastive_embedding_format_and_upstream_license_conflict",
    },
    {
        "repository": "mdsajjadullah/banglish-sentiment-2026",
        "decision": "ineligible_for_general_assistant_closure",
        "reason": "classification_only_synthetic_text_not_assistant_responses",
    },
]

EXPECTED_FAMILY_PROJECTION = {
    "projected_total_train_supervised_tokens_ceiling": 10_456_670,
    "nvidia_family": {
        "current_tokens": 2_100_077,
        "additional_ceiling": 1_550_000,
        "projected_share": 0.34906686,
    },
    "coherelabs_aya_family": {
        "current_tokens": 513_379,
        "additional_ceiling": 1_300_000,
        "projected_share": 0.1734184,
    },
    "banglatlit_family": {
        "current_tokens": 0,
        "additional_ceiling": 300_000,
        "projected_share": 0.02868982,
    },
    "banglabridge_family": {
        "current_tokens": 0,
        "additional_ceiling": 50_000,
        "projected_share": 0.00478164,
    },
}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic pretty JSON bytes."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    """Hash a mapping as compact canonical JSON."""

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
    if not isinstance(expected, str):
        raise ValueError(f"missing internal digest: {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if sha256_value(unsigned) != expected:
        raise ValueError(f"internal digest does not verify: {field}")


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    """Load and validate the metadata-only source preflight plan."""

    root = Path(repository_root).resolve()
    plan = _load_json(Path(path).resolve())
    expected_scalars = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_official_metadata_only_preflight",
        "observed_at": "2026-08-09",
        "authorization": EXPECTED_AUTHORIZATION,
        "current_mix": EXPECTED_CURRENT_MIX,
        "sources": EXPECTED_SOURCES,
        "rejected_or_quarantined_sources": EXPECTED_REJECTIONS,
        "family_projection": EXPECTED_FAMILY_PROJECTION,
    }
    for field, expected in expected_scalars.items():
        if plan.get(field) != expected:
            raise ValueError(f"rights/provenance preflight changed: {field}")

    bindings = plan.get("bindings")
    expected_bindings = {
        "source_registry",
        "candidate_mix_reaudit",
        "teacher_strategy_reconciliation",
        "teacher_strategy_reconciliation_verification",
        "implementation",
        "runner",
        "verifier",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise ValueError("rights/provenance binding set changed")
    for name, binding in bindings.items():
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"binding escaped repository: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding hash mismatch: {name}")

    sources = plan["sources"]
    if sum(row["target_retained_train_supervised_tokens_ceiling"] for row in sources) != 3_200_000:
        raise ValueError("shortlist target ceiling changed")
    if any(
        family["projected_share"] > EXPECTED_CURRENT_MIX["source_cap"]
        for family in plan["family_projection"].values()
        if isinstance(family, dict)
    ):
        raise ValueError("projected conservative family share exceeds cap")
    return plan


def build_report(
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build a raw-text-free source decision report."""

    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    reconciliation = _load_json(
        (root / bindings["teacher_strategy_reconciliation"]["path"]).resolve()
    )
    verification = _load_json(
        (
            root
            / bindings["teacher_strategy_reconciliation_verification"]["path"]
        ).resolve()
    )
    _verify_internal(reconciliation, "report_sha256")
    _verify_internal(verification, "verification_sha256")
    if reconciliation.get("status") != "verified_teacher_strategies_rejected_mix_unchanged":
        raise ValueError("teacher reconciliation status changed")
    if verification.get("status") != "PASS":
        raise ValueError("teacher reconciliation verification failed")

    general_assistant_ceiling = sum(
        row["target_retained_train_supervised_tokens_ceiling"]
        for row in plan["sources"]
        if row["counts_toward_general_assistant_closure"]
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_shortlist_ready_payload_and_sft_not_authorized",
        "plan_file_sha256": plan_file_sha256,
        "current_mix": plan["current_mix"],
        "selected_sources": plan["sources"],
        "rejected_or_quarantined_sources": plan[
            "rejected_or_quarantined_sources"
        ],
        "family_projection": plan["family_projection"],
        "closure_assessment": {
            "all_lane_token_ceiling": 3_200_000,
            "general_assistant_token_ceiling": general_assistant_ceiling,
            "auxiliary_transliteration_token_ceiling": 300_000,
            "quantitative_10m_ceiling_reachable": True,
            "production_closure_proven": False,
            "blocking_reasons": [
                "payload_rows_have_not_been_acquired_or_machine_audited",
                "retained_token_counts_are_ceiling_estimates_not_measurements",
                "banglish_and_natural_code_switching_quality_floors_are_not_yet_proven",
                "auxiliary_transliteration_rows_do_not_replace_assistant_behavior_rows",
            ],
        },
        "next_gate": {
            "action": "exact_object_bounded_source_canary_acquisition_plan",
            "authorization_required": True,
            "required_actions": [
                "pin_exact_readme_license_and_payload_object_hashes_and_sizes",
                "acquire_smallest_useful_chat_tool_bengali_and_banglish_canaries",
                "reserve_all_published_test_and_validation_objects",
                "run_schema_language_quality_safety_tool_pii_secret_dedup_decontamination_and_4k_fit",
                "measure_frozen_tokenizer_tokens_and_source_family_shares",
                "create_deterministic_owner_review_packets_before_any_registry_change",
            ],
            "bulk_payload_download_authorized": False,
            "candidate_registry_change_authorized": False,
            "dataset_seal_authorized": False,
            "gpu_or_sft_authorized": False,
        },
        "result_preservation": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "payload_rows_downloaded": 0,
            "sft_training_started": False,
        },
        "raw_training_text_retained": False,
        "legal_advice_claimed": False,
        "production_eligible": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_verification(
    report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    """Build a deterministic verification receipt."""

    _verify_internal(report, "report_sha256")
    checks = {
        "current_mix_and_failed_teacher_reconciliation_bound": True,
        "official_repository_revisions_and_declared_licenses_recorded": True,
        "noncommercial_ambiguous_and_non_assistant_sources_fail_closed": True,
        "conservative_family_share_projection_at_or_below_0_35": True,
        "transliteration_not_misrepresented_as_general_assistant_data": True,
        "no_payload_registry_dataset_seal_gpu_or_sft": True,
    }
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS",
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "checks": checks,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
