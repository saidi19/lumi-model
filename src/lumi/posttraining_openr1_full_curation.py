"""Fail-closed local preflight for full OpenR1 curation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-openr1-full-curation-plan.v0.1"
PREFLIGHT_SCHEMA_VERSION = "lumi.sft-openr1-full-curation-preflight.v0.1"

SOURCE_ID = "openr1_math_220k"
DATASET_ID = "open-r1/OpenR1-Math-220k"
SOURCE_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"

DATA_OBJECTS = [f"data/train-{index:05d}-of-00010.parquet" for index in range(10)]
EXACT_OBJECTS = [
    *({"kind": "data", "path": path, "split": "train"} for path in DATA_OBJECTS),
    {"kind": "documentation", "path": "README.md"},
]
EXPECTED_FEATURES = [
    "problem",
    "solution",
    "answer",
    "problem_type",
    "question_type",
    "source",
    "uuid",
    "is_reasoning_complete",
    "generations",
    "correctness_math_verify",
    "correctness_llama",
    "finish_reasons",
    "correctness_count",
    "messages",
]
EXPECTED_SOURCE = {
    "config": "default",
    "dataset_id": DATASET_ID,
    "dataset_size_bytes": 4_964_543_659,
    "declared_license_id": "apache-2.0",
    "download_size_bytes": 2_149_897_914,
    "revision": SOURCE_REVISION,
    "rows": 93_733,
    "source_id": SOURCE_ID,
    "split": "train",
}
EXPECTED_LIMITS = {
    "context_window": 4_096,
    "expected_data_download_bytes": 2_149_897_914,
    "expected_decoded_data_bytes": 4_964_543_659,
    "maximum_data_objects": 10,
    "maximum_decoded_bytes": 5_368_709_120,
    "maximum_network_bytes": 2_415_919_104,
    "maximum_objects": 11,
    "maximum_retained_supervised_tokens": 6_600_000,
    "maximum_runtime_minutes": 360,
    "maximum_source_rows": 93_733,
    "owner_review_sample_records": 32,
}
EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": False,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "local_openr1_full_curation_preflight_only",
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_ACQUISITION = {
    "exact_objects": EXACT_OBJECTS,
    "mode": "future_revision_pinned_fetch_requires_separate_authorization",
    "object_sizes_and_hashes_must_be_recorded_during_fetch": True,
    "reject_redirect_to_different_revision": True,
    "reject_unlisted_objects": True,
    "require_source_head_before_and_after": True,
}
EXPECTED_POLICY = {
    "assistant_trace_field": "generations",
    "assistant_trace_must_differ_from_reference_solution": True,
    "benchmark_decontamination_required": True,
    "cross_split_exact_and_near_deduplication_required": True,
    "external_image_or_url_dependencies_allowed": False,
    "global_pii_filter_modified": False,
    "maximum_final_train_supervised_token_share": 0.35,
    "origin_label": "synthetic",
    "problem_group_split_required": True,
    "record_level_attribution_required": True,
    "record_level_quality_review_required": True,
    "reference_solution_field_allowed_as_assistant_target": False,
    "require_aligned_generation_verifier_vectors": True,
    "require_nonempty_upstream_problem_source": True,
    "require_reasoning_complete": True,
    "require_verifier_field": "correctness_math_verify",
    "require_verifier_pass": True,
    "secondary_pii_reconciliation": {
        "allowed_match_role": "assistant",
        "contact_keyword_scope": "same_line_as_phone_match",
        "cross_message_join_is_not_a_source_text_match": True,
        "math_context_required": True,
        "other_sensitive_detector_hits_allowed": False,
        "quarantine_before_reclassification": True,
    },
    "source_specific_exception_is_global_override": False,
    "upstream_source_rights_map_required_before_promotion": True,
    "validation_fraction": 0.05,
}
EXPECTED_OUTPUT = {
    "candidate_materialization_permitted": False,
    "preflight_report_contains_raw_training_text": False,
    "preflight_report_path": (
        "evidence/lumi-4b-sft-openr1-full-curation-preflight-v0.1.json"
    ),
    "production_promotion_permitted": False,
    "raw_dataset_in_git_permitted": False,
}
EXPECTED_BINDING_PATHS = {
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "curation_implementation": "src/lumi/posttraining_openr1_full_curation.py",
    "curation_runner": "scripts/check_openr1_full_curation_plan.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "final_base_binding": "configs/lumi-4b-final-base-binding-v0.1.json",
    "missing_mix_proposal": "configs/lumi-4b-sft-missing-mix-proposal-v0.1.json",
    "pinned_metadata": (
        "../lumi-artifacts/4b-sft-source-canary-v0.1/"
        "5d1b59ab15bd03f23f2fd3d4e7ef931041d22a93ac0887a2ad77a289d5908f02/"
        "attempt-004/raw/openr1_math_220k-metadata-post.json"
    ),
    "secondary_pii_evidence": "evidence/lumi-4b-sft-openr1-pii-secondary-v0.1.json",
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.4.json",
    "verified_trace_evidence": (
        "evidence/lumi-4b-sft-openr1-verified-trace-audit-v0.1.json"
    ),
    "verified_trace_implementation": "src/lumi/posttraining_openr1.py",
}
REMAINING_BLOCKERS = [
    "full_dataset_not_acquired_or_curated",
    "full_dataset_secondary_pii_validation_not_run",
    "full_dataset_4096_token_fit_not_measured",
    "upstream_problem_source_rights_and_attribution_not_resolved_per_row",
    "record_level_quality_review_not_complete",
    "deterministic_owner_quality_sample_not_reviewed",
    "candidate_source_registry_promotion_not_authorized",
    "production_dataset_seal_not_authorized",
]

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


def json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent
    relative_path = binding.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"OpenR1 full-curation binding path is invalid: {name}")
    resolved = (root / relative_path).resolve()
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError(f"OpenR1 full-curation binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"OpenR1 full-curation binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"OpenR1 full-curation binding changed: {name}")
    return resolved


def load_openr1_full_curation_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "authorization": EXPECTED_AUTHORIZATION,
        "source": EXPECTED_SOURCE,
        "limits": EXPECTED_LIMITS,
        "acquisition": EXPECTED_ACQUISITION,
        "policy": EXPECTED_POLICY,
        "output": EXPECTED_OUTPUT,
        "status": "local_preflight_ready_acquisition_requires_authorization",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"OpenR1 full-curation plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("OpenR1 full-curation bindings must be an object")
    if set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("OpenR1 full-curation binding set changed")
    for name, expected_path in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"OpenR1 full-curation binding path changed: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def summarize_pinned_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if metadata.get("id") != DATASET_ID or metadata.get("sha") != SOURCE_REVISION:
        raise ValueError("OpenR1 pinned metadata identity changed")
    if metadata.get("private") is not False or metadata.get("gated") not in {
        False,
        None,
    }:
        raise ValueError("OpenR1 pinned metadata access boundary changed")
    card = metadata.get("cardData")
    if not isinstance(card, Mapping) or card.get("license") != "apache-2.0":
        raise ValueError("OpenR1 declared dataset license changed")
    dataset_info = card.get("dataset_info")
    if not isinstance(dataset_info, list):
        raise ValueError("OpenR1 dataset_info is missing")
    default = next(
        (
            item
            for item in dataset_info
            if isinstance(item, Mapping) and item.get("config_name") == "default"
        ),
        None,
    )
    if not isinstance(default, Mapping):
        raise ValueError("OpenR1 default config metadata is missing")
    features = default.get("features")
    feature_names = [
        item.get("name") for item in features if isinstance(item, Mapping)
    ] if isinstance(features, list) else []
    if feature_names != EXPECTED_FEATURES:
        raise ValueError("OpenR1 default feature schema changed")
    splits = default.get("splits")
    if not isinstance(splits, list) or len(splits) != 1:
        raise ValueError("OpenR1 default split metadata changed")
    split = splits[0]
    if not isinstance(split, Mapping) or dict(split) != {
        "name": "train",
        "num_bytes": EXPECTED_SOURCE["dataset_size_bytes"],
        "num_examples": EXPECTED_SOURCE["rows"],
    }:
        raise ValueError("OpenR1 default train split changed")
    if default.get("download_size") != EXPECTED_SOURCE["download_size_bytes"]:
        raise ValueError("OpenR1 compressed size changed")
    if default.get("dataset_size") != EXPECTED_SOURCE["dataset_size_bytes"]:
        raise ValueError("OpenR1 decoded size changed")

    siblings = metadata.get("siblings")
    sibling_paths = {
        item.get("rfilename") for item in siblings if isinstance(item, Mapping)
    } if isinstance(siblings, list) else set()
    expected_paths = {item["path"] for item in EXACT_OBJECTS}
    if not expected_paths.issubset(sibling_paths):
        raise ValueError("OpenR1 exact acquisition object is missing")
    return {
        "config": "default",
        "data_object_count": len(DATA_OBJECTS),
        "dataset_id": DATASET_ID,
        "decoded_data_bytes": EXPECTED_SOURCE["dataset_size_bytes"],
        "declared_license_id": "apache-2.0",
        "download_data_bytes": EXPECTED_SOURCE["download_size_bytes"],
        "exact_object_paths_sha256": sha256_value(
            sorted(item["path"] for item in EXACT_OBJECTS)
        ),
        "feature_names_sha256": sha256_value(EXPECTED_FEATURES),
        "objects": len(EXACT_OBJECTS),
        "revision": SOURCE_REVISION,
        "rows": EXPECTED_SOURCE["rows"],
        "split": "train",
    }


def build_openr1_full_curation_preflight(
    *,
    metadata: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    secondary_pii_evidence: Mapping[str, Any],
    verified_trace_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    summary = summarize_pinned_metadata(metadata)
    if verified_trace_evidence.get("report_sha256") != (
        "4d5cdab5179154138650059da9044d19826f6510a6fb749a58bd195637431a8d"
    ):
        raise ValueError("OpenR1 verified-trace evidence changed")
    if secondary_pii_evidence.get("report_sha256") != (
        "7761ab5800608658c0ea99d6b2a1d03e19c2c46c85b1641e647789ef1069fb6a"
    ):
        raise ValueError("OpenR1 secondary PII evidence changed")
    report = {
        "acquisition_authorized": False,
        "authorization": dict(plan["authorization"]),
        "bounded_slice_evidence": {
            "encoded_4k_records": verified_trace_evidence["encoding_metrics"][
                "accepted"
            ],
            "global_filter_modified": secondary_pii_evidence[
                "global_pii_filter_modified"
            ],
            "normal_filter_records": verified_trace_evidence["filter_metrics"][
                "accepted"
            ],
            "secondary_unresolved_records": secondary_pii_evidence[
                "diagnostic_metrics"
            ]["unresolved_records"],
            "selected_verified_trace_records": verified_trace_evidence[
                "selection_metrics"
            ]["selected_records"],
        },
        "decision": "exact_full_acquisition_requires_separate_bounded_authorization",
        "metadata_summary": summary,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": REMAINING_BLOCKERS,
        "rights_boundary": {
            "dataset_card_declared_license": "apache-2.0",
            "dataset_level_license_treated_as_per_row_rights_proof": False,
            "raw_text_free_upstream_source_inventory_required": True,
            "records_quarantined_until_source_rights_map": True,
        },
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "local_preflight_passed_no_download_no_promotion",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_openr1_full_curation_preflight(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 full-curation preflight report")
    if report.get("acquisition_authorized") is not False:
        raise ValueError("OpenR1 preflight cannot authorize acquisition")
    if report.get("production_eligible") is not False:
        raise ValueError("OpenR1 preflight cannot establish production eligibility")
    if report.get("production_promotions") != 0:
        raise ValueError("OpenR1 preflight cannot promote records")
    if report.get("raw_text_retained") is not False:
        raise ValueError("OpenR1 preflight report must be raw-text-free")
    if report.get("remaining_production_blockers") != REMAINING_BLOCKERS:
        raise ValueError("OpenR1 preflight blocker set changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    if seal != sha256_value(unsealed):
        raise ValueError("OpenR1 full-curation preflight seal mismatch")
