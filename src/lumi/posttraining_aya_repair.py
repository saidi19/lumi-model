"""Fail-closed local repair preflight for the pinned Aya Bengali subset."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .posttraining_oasst1_full_run import (
    deterministic_owner_sample,
    file_sha256,
    sha256_value,
)

PLAN_SCHEMA_VERSION = "lumi.sft-aya-bengali-repair-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-bengali-repair-preflight-report.v0.1"

SOURCE_ID = "coherelabs_aya_dataset"
DATASET_ID = "CohereLabs/aya_dataset"
SOURCE_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
SOURCE_LICENSE = "apache-2.0"

EXPECTED_STATUS = "local_repair_preflight_only"
EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "dataset_promotion_authorized": False,
    "external_api_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_pinned_aya_bengali_local_repair_preflight_only",
    "unrelated_source_access_authorized": False,
}
EXPECTED_LIMITS = {
    "expected_input_bytes": 1_990_313,
    "expected_page_objects": 23,
    "expected_source_rows": 1_534,
    "maximum_runtime_minutes": 30,
    "owner_sample_records": 32,
}
EXPECTED_SOURCE = {
    "config": "default",
    "dataset_id": DATASET_ID,
    "license_id": SOURCE_LICENSE,
    "revision": SOURCE_REVISION,
    "source_id": SOURCE_ID,
    "split": "train",
}
EXPECTED_POLICY = {
    "allowed_annotation_types": ["original-annotations", "re-annotations"],
    "concentration_selection": "content_hash_order_cap_offender_to_all_other_rows",
    "deduplication_key": "unicode_nfc_trimmed_prompt_target_pair",
    "deduplication_tiebreak": "least_frequent_annotator_then_full_payload_sha256",
    "language": "bn",
    "maximum_annotator_fraction": 0.5,
    "normalization": [
        "unicode_nfc",
        "strip_outer_whitespace",
        "remove_exactly_one_terminal_target_pipe_artifact",
    ],
    "post_repair_gates": [
        "lumi_schema_validation",
        "pii_and_secret_scan",
        "exact_and_normalized_deduplication",
        "benchmark_decontamination",
        "frozen_4096_token_fit",
        "assistant_only_loss_encoding",
        "final_annotator_concentration_cap",
        "deterministic_owner_quality_sample",
    ],
    "require_bengali_script_in_prompt_and_target": True,
    "require_exact_language_labels": True,
    "split": "train",
    "task_category": "multilingual_bengali",
}
EXPECTED_OUTPUT = {
    "candidate_records_materialized": False,
    "owner_sample_materialized": False,
    "production_promotion_permitted": False,
    "raw_training_text_in_report": False,
    "report_path": "evidence/lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json",
}
EXPECTED_BINDING_NAMES = frozenset(
    {
        "benchmark_registry",
        "candidate_source_registry",
        "chat_template",
        "data_contract",
        "filter_implementation",
        "format_implementation",
        "input_audit_report",
        "input_inventory",
        "missing_mix_proposal",
        "reconciliation_receipt",
        "repair_implementation",
        "repair_runner",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TERMINAL_PIPE = re.compile(r"[\t ]*\|[\t ]*\Z")


@dataclass(frozen=True)
class AyaRepairSelection:
    """Private in-memory repaired rows and a raw-text-free metric summary."""

    candidates: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_workspace_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    """Resolve one hash-bound file inside the shared workspace."""

    root = Path(repository_root).resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Aya repair binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError("Aya repair binding escapes the workspace")
    if not resolved.is_file():
        raise FileNotFoundError(f"Aya repair binding is missing: {relative}")
    expected = _require_sha256(binding.get("sha256"), f"binding {relative}")
    if file_sha256(resolved) != expected:
        raise ValueError(f"Aya repair binding changed: {relative}")
    return resolved


def load_aya_repair_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact non-authorizing local Aya repair preflight plan."""

    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya repair preflight plan")
    if plan.get("status") != EXPECTED_STATUS:
        raise ValueError("Aya repair preflight status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("Aya repair preflight observation date changed")
    for field, expected in {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "output": EXPECTED_OUTPUT,
        "policy": EXPECTED_POLICY,
        "source": EXPECTED_SOURCE,
    }.items():
        if plan.get(field) != expected:
            raise ValueError(f"Aya repair preflight {field} changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("Aya repair preflight binding inventory changed")
    for name in sorted(EXPECTED_BINDING_NAMES):
        binding = bindings[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya repair binding is invalid: {name}")
        resolve_workspace_binding(repository_root, binding)
    return plan


def verify_aya_input_receipts(
    *,
    audit_report: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> None:
    """Require the previously sealed full-count audit and reconciliation."""

    if audit_report.get("schema_version") != "lumi.sft-aya-bengali-source-audit-report.v0.1":
        raise ValueError("Aya source audit schema changed")
    audit_seal = _require_sha256(audit_report.get("report_sha256"), "Aya audit seal")
    if (
        sha256_value({key: value for key, value in audit_report.items() if key != "report_sha256"})
        != audit_seal
    ):
        raise ValueError("Aya source audit seal is invalid")
    if (
        audit_report.get("source_rows") != 1_534
        or audit_report.get("production_promotions") != 0
        or audit_report.get("raw_text_retained") is not False
    ):
        raise ValueError("Aya source audit boundary changed")

    if reconciliation.get("schema_version") != (
        "lumi.sft-aya-bengali-audit-reconciliation-report.v0.1"
    ):
        raise ValueError("Aya reconciliation schema changed")
    reconciliation_seal = _require_sha256(
        reconciliation.get("report_sha256"), "Aya reconciliation seal"
    )
    if (
        sha256_value(
            {key: value for key, value in reconciliation.items() if key != "report_sha256"}
        )
        != reconciliation_seal
    ):
        raise ValueError("Aya reconciliation seal is invalid")
    if (
        reconciliation.get("retrieval_integrity", {}).get("fetched_rows") != 1_534
        or reconciliation.get("production_promotions") != 0
        or reconciliation.get("raw_text_retained") is not False
    ):
        raise ValueError("Aya reconciliation boundary changed")


def _contains_bengali(value: str) -> bool:
    return any("\u0980" <= character <= "\u09ff" for character in value)


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _normalize_target(value: str) -> tuple[str, bool]:
    normalized = _normalize_text(value)
    repaired = _TERMINAL_PIPE.search(normalized) is not None
    if repaired:
        normalized = _TERMINAL_PIPE.sub("", normalized).rstrip()
    return normalized, repaired


def _cap_annotator_concentration(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Deterministically cap any single annotator at one half of final rows."""

    selected = list(candidates)
    before = Counter(str(row["annotator_sha256"]) for row in selected)
    removed: Counter[str] = Counter()
    while selected:
        counts = Counter(str(row["annotator_sha256"]) for row in selected)
        annotator, count = counts.most_common(1)[0]
        other_rows = len(selected) - count
        if count <= other_rows:
            break
        offender = sorted(
            (row for row in selected if row["annotator_sha256"] == annotator),
            key=lambda row: str(row["content_sha256"]),
        )
        keep = {str(row["record_id"]) for row in offender[:other_rows]}
        next_selected: list[Mapping[str, Any]] = []
        for row in selected:
            if row["annotator_sha256"] != annotator or row["record_id"] in keep:
                next_selected.append(row)
            else:
                removed[annotator] += 1
        selected = next_selected

    after = Counter(str(row["annotator_sha256"]) for row in selected)
    maximum = max(after.values(), default=0)
    fraction = maximum / len(selected) if selected else 0.0
    return selected, {
        "maximum_fraction_after": fraction,
        "maximum_records_after": maximum,
        "records_after": len(selected),
        "records_before": len(candidates),
        "records_removed": sum(removed.values()),
        "top_annotator_records_before": max(before.values(), default=0),
        "unique_annotators_after": len(after),
        "unique_annotators_before": len(before),
    }


def select_aya_repair_candidates(
    wrapped_rows: Sequence[Mapping[str, Any]],
) -> AyaRepairSelection:
    """Normalize, content-address, deduplicate, and label exact Aya Bengali rows."""

    if len(wrapped_rows) != EXPECTED_LIMITS["expected_source_rows"]:
        raise ValueError("Aya repair source row count changed")
    rejected: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    annotator_pool: Counter[str] = Counter()
    repaired_terminal_pipes = 0

    for wrapped in wrapped_rows:
        if not isinstance(wrapped, Mapping) or not isinstance(wrapped.get("row"), Mapping):
            rejected["invalid_row_wrapper"] += 1
            continue
        row = wrapped["row"]
        if set(row) != {
            "annotation_type",
            "inputs",
            "language",
            "language_code",
            "targets",
            "user_id",
        }:
            rejected["unexpected_row_schema"] += 1
            continue
        if row.get("language") != "Bengali" or row.get("language_code") != "ben":
            rejected["language_label_mismatch"] += 1
            continue
        if row.get("annotation_type") not in EXPECTED_POLICY["allowed_annotation_types"]:
            rejected["annotation_type"] += 1
            continue
        prompt = row.get("inputs")
        target = row.get("targets")
        user_id = row.get("user_id")
        if not all(isinstance(value, str) and value.strip() for value in (prompt, target, user_id)):
            rejected["empty_or_invalid_required_field"] += 1
            continue
        prompt = _normalize_text(prompt)
        target, repaired = _normalize_target(target)
        if repaired:
            repaired_terminal_pipes += 1
        if not prompt or not target:
            rejected["empty_after_normalization"] += 1
            continue
        if not _contains_bengali(prompt) or not _contains_bengali(target):
            rejected["missing_bengali_script"] += 1
            continue

        annotator = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        pair_sha256 = sha256_value([prompt, target])
        full_payload_sha256 = sha256_value(
            {
                "annotation_type": row["annotation_type"],
                "annotator_sha256": annotator,
                "prompt": prompt,
                "target": target,
            }
        )
        annotator_pool[annotator] += 1
        normalized.append(
            {
                "annotation_type": row["annotation_type"],
                "annotator_sha256": annotator,
                "content_sha256": pair_sha256,
                "full_payload_sha256": full_payload_sha256,
                "prompt": prompt,
                "record_id": f"aya-ben-{pair_sha256[:32]}",
                "target": target,
            }
        )

    by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_pair[row["content_sha256"]].append(row)
    deduplicated: list[dict[str, Any]] = []
    for rows in by_pair.values():
        rows.sort(
            key=lambda row: (
                annotator_pool[row["annotator_sha256"]],
                row["full_payload_sha256"],
            )
        )
        deduplicated.append(rows[0])
        rejected["normalized_exact_pair_duplicate"] += len(rows) - 1
    deduplicated.sort(key=lambda row: row["record_id"])

    metrics = {
        "deduplicated_rows": len(deduplicated),
        "input_rows": len(wrapped_rows),
        "normalized_rows": len(normalized),
        "rejection_counts": dict(sorted(rejected.items())),
        "terminal_pipe_artifacts_repaired": repaired_terminal_pipes,
        "unique_annotators_before_final_gates": len(annotator_pool),
    }
    return AyaRepairSelection(candidates=tuple(deduplicated), metrics=metrics)


def build_aya_sft_records(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert repaired rows to the Lumi schema without claiming owner review."""

    records: list[dict[str, Any]] = []
    annotators: dict[str, str] = {}
    transform_sha256 = sha256_value({"policy": EXPECTED_POLICY, "revision": SOURCE_REVISION})
    for candidate in candidates:
        record_id = str(candidate["record_id"])
        content_sha256 = str(candidate["content_sha256"])
        records.append(
            {
                "language": "bn",
                "messages": [
                    {"content": candidate["prompt"], "role": "user"},
                    {"content": candidate["target"], "role": "assistant"},
                ],
                "provenance": {
                    "consent_basis": "source_level_open_license",
                    "content_sha256": content_sha256,
                    "transform_sha256": transform_sha256,
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_method": "aya_upstream_human_annotation_plus_lumi_machine_repair_v0.1",
                    "review_status": "accepted",
                    "score": 0.75,
                    "secret_scan": "passed",
                },
                "record_id": record_id,
                "safety": {
                    "policy_version": "lumi-sft-basic-pattern-scan-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": DATASET_ID,
                    "document_id": content_sha256,
                    "license_id": SOURCE_LICENSE,
                    "revision": SOURCE_REVISION,
                    "source_id": SOURCE_ID,
                    "uri": (
                        f"https://huggingface.co/datasets/{DATASET_ID}/tree/"
                        f"{SOURCE_REVISION}#sha256={content_sha256}"
                    ),
                },
                "split": "train",
                "task_category": "multilingual_bengali",
            }
        )
        annotators[record_id] = str(candidate["annotator_sha256"])
    return records, annotators


def finalize_aya_repair_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    annotators: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Apply the final concentration cap after all other machine gates."""

    annotated = [
        {
            **dict(record),
            "annotator_sha256": annotators[str(record["record_id"])],
            "content_sha256": record["provenance"]["content_sha256"],
        }
        for record in records
    ]
    capped, metrics = _cap_annotator_concentration(annotated)
    final_records = []
    for row in capped:
        clean = dict(row)
        clean.pop("annotator_sha256")
        clean.pop("content_sha256")
        final_records.append(clean)
    final_records.sort(key=lambda row: str(row["record_id"]))
    return final_records, metrics


def build_aya_repair_report(
    *,
    bindings: Mapping[str, str],
    concentration_metrics: Mapping[str, Any],
    filter_report: Mapping[str, Any],
    final_records: Sequence[Mapping[str, Any]],
    owner_sample_records: int,
    plan_file_sha256: str,
    selection_metrics: Mapping[str, Any],
    token_metrics: Mapping[str, Any],
    annotators: Mapping[str, str],
) -> dict[str, Any]:
    """Build the sealed raw-text-free local repair preflight receipt."""

    for name, digest in bindings.items():
        _require_sha256(digest, name)
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if concentration_metrics.get("maximum_fraction_after", 1.0) > 0.5:
        raise ValueError("Aya final annotator concentration exceeds the policy")
    owner_sample = deterministic_owner_sample(
        final_records,
        maximum_records=owner_sample_records,
    )
    final_ids = sorted(str(record["record_id"]) for record in final_records)
    attribution_ledger = [
        {
            "annotator_sha256": annotators[record_id],
            "record_id": record_id,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
        }
        for record_id in final_ids
    ]
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": dict(sorted(bindings.items())),
        "candidate": {
            "attribution_ledger_materialized": False,
            "attribution_ledger_sha256": sha256_value(attribution_ledger),
            "materialized": False,
            "record_ids_sha256": sha256_value(final_ids),
            "records": len(final_records),
            "supervised_tokens": token_metrics["supervised_tokens"],
        },
        "filter": {
            "accepted_before_4k_and_concentration": filter_report["accepted"],
            "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
            "rejected": filter_report["rejected"],
            "rejection_counts": filter_report["rejection_counts"],
            "report_sha256": filter_report["report_sha256"],
        },
        "observed_at": "2026-08-06",
        "owner_review": {
            "complete": False,
            "record_count": len(owner_sample),
            "sample_record_ids_sha256": sha256_value(
                sorted(str(record["record_id"]) for record in owner_sample)
            ),
            "status": "candidate_materialization_and_owner_review_not_authorized",
        },
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_promotion_authorized": False,
            "dataset_seal_authorized": False,
            "production_eligible": False,
            "sft_training_authorized": False,
        },
        "raw_training_text_present": False,
        "repair": {
            "concentration": dict(concentration_metrics),
            "selection": dict(selection_metrics),
        },
        "remaining_blockers": [
            "candidate_records_and_attribution_manifest_not_materialized",
            "deterministic_owner_quality_sample_not_reviewed",
            "full_record_level_instruction_compliance_review_not_complete",
            "cross_source_deduplication_and_full_10m_to_30m_mix_not_complete",
            "candidate_registry_promotion_not_authorized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": dict(EXPECTED_SOURCE),
        "status": "local_repair_preflight_passed_materialization_requires_authorization",
        "token_measurement": dict(token_metrics),
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_report_seal(report: Mapping[str, Any]) -> None:
    """Independently validate the report seal and non-production boundary."""

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Aya repair report schema changed")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    if (
        sha256_value({key: value for key, value in report.items() if key != "report_sha256"})
        != seal
    ):
        raise ValueError("Aya repair report seal is invalid")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("Aya repair report retained raw text")
    production = report.get("production")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("Aya repair preflight crossed a production boundary")
