"""Raw-text-free reconciliation for Aya dataset-viewer pagination evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from lumi.posttraining_source_quality import sha256_value

REPORT_SCHEMA_VERSION = "lumi.sft-aya-bengali-audit-reconciliation-report.v0.1"


def _bengali_frequency(statistics: Mapping[str, Any]) -> int:
    entries = statistics.get("statistics")
    if statistics.get("partial") is not False or not isinstance(entries, list):
        raise ValueError("Aya statistics response is partial or invalid")
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("column_name") != "language_code":
            continue
        column = entry.get("column_statistics")
        frequencies = column.get("frequencies") if isinstance(column, Mapping) else None
        value = frequencies.get("ben") if isinstance(frequencies, Mapping) else None
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Aya statistics response omitted the Bengali frequency")


def build_aya_audit_reconciliation_report(
    *,
    bindings: Mapping[str, str],
    observed_at: str,
    original_report: Mapping[str, Any],
    statistics: Mapping[str, Any],
    wrapped_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Correctly separate source duplicates from unstable viewer row indices."""

    if original_report.get("schema_version") != "lumi.sft-aya-bengali-source-audit-report.v0.1":
        raise ValueError("Aya original audit schema changed")
    original_without_seal = {
        key: value for key, value in original_report.items() if key != "report_sha256"
    }
    if original_report.get("report_sha256") != sha256_value(original_without_seal):
        raise ValueError("Aya original audit seal is invalid")
    if original_report.get("production_promotions") != 0:
        raise ValueError("Aya original audit unexpectedly promoted data")
    if original_report.get("raw_text_retained") is not False:
        raise ValueError("Aya original audit retained raw text")
    for name, digest in bindings.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Aya reconciliation binding is invalid")

    official_bengali_rows = _bengali_frequency(statistics)
    if not 1 <= official_bengali_rows <= 1534 or len(wrapped_rows) != official_bengali_rows:
        raise ValueError("Aya reconciliation row boundary changed")

    row_index_payloads: dict[int, set[str]] = defaultdict(set)
    row_index_counts: Counter[int] = Counter()
    full_payload_counts: Counter[str] = Counter()
    full_payload_indices: dict[str, set[int]] = defaultdict(set)
    pair_counts: Counter[str] = Counter()
    for wrapped in wrapped_rows:
        if not isinstance(wrapped, Mapping):
            raise ValueError("Aya reconciliation row wrapper is invalid")
        row_index = wrapped.get("row_idx")
        row = wrapped.get("row")
        if not isinstance(row_index, int) or not isinstance(row, Mapping):
            raise ValueError("Aya reconciliation row is invalid")
        if row.get("language_code") != "ben":
            raise ValueError("Aya reconciliation contains a non-Bengali row")
        payload_digest = sha256_value(row)
        pair_digest = sha256_value([row.get("inputs"), row.get("targets")])
        row_index_counts[row_index] += 1
        row_index_payloads[row_index].add(payload_digest)
        full_payload_counts[payload_digest] += 1
        full_payload_indices[payload_digest].add(row_index)
        pair_counts[pair_digest] += 1

    row_index_collision_groups = sum(count > 1 for count in row_index_counts.values())
    row_index_collision_excess = sum(count - 1 for count in row_index_counts.values())
    distinct_content_index_collisions = sum(
        len(payloads) > 1 for payloads in row_index_payloads.values()
    )
    exact_full_duplicate_groups = sum(count > 1 for count in full_payload_counts.values())
    exact_full_duplicate_excess = sum(
        count - 1 for count in full_payload_counts.values()
    )
    duplicate_groups_with_distinct_indices = sum(
        len(full_payload_indices[digest]) > 1
        for digest, count in full_payload_counts.items()
        if count > 1
    )
    exact_pair_duplicate_excess = sum(count - 1 for count in pair_counts.values())

    original_duplicate_metrics = original_report.get("duplicate_metrics")
    if not isinstance(original_duplicate_metrics, Mapping):
        raise ValueError("Aya original duplicate metrics are missing")
    if original_duplicate_metrics.get("duplicate_row_indices") != row_index_collision_excess:
        raise ValueError("Aya row-index collision count changed")
    if original_duplicate_metrics.get("duplicate_pair_records") != exact_pair_duplicate_excess:
        raise ValueError("Aya pair duplicate count changed")

    provenance = original_report.get("provenance_metrics")
    formatting = original_report.get("format_and_content_metrics")
    language = original_report.get("language_metrics")
    if not all(isinstance(value, Mapping) for value in (provenance, formatting, language)):
        raise ValueError("Aya original quality metrics are missing")
    top_annotator_records = provenance.get("top_annotator_records")
    trailing_pipe_records = formatting.get("trailing_pipe_records")
    if not isinstance(top_annotator_records, int) or not isinstance(
        trailing_pipe_records, int
    ):
        raise ValueError("Aya robustness inputs are invalid")
    half_limit_records = official_bengali_rows // 2
    concentration_margin_records = top_annotator_records - half_limit_records
    uncertainty_records = row_index_collision_excess

    blockers = [
        blocker
        for blocker in original_report.get("remaining_production_blockers", [])
        if blocker != "full_bengali_subset_contains_duplicate_row_indices"
    ]
    pagination_blocker = (
        "dataset_viewer_pagination_index_instability_requires_"
        "content_addressed_materialization"
    )
    if pagination_blocker not in blockers:
        blockers.insert(2, pagination_blocker)

    report: dict[str, Any] = {
        "authorization": {
            "dataset_promotion_authorized": False,
            "external_api_spend_authorized": False,
            "network_access_used": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "existing_aya_audit_retrieval_reconciliation_only",
        },
        "bindings": dict(sorted(bindings.items())),
        "decision": "full_count_quality_audit_complete_with_pagination_caveat",
        "duplicate_interpretation": {
            "exact_full_payload_duplicate_excess": exact_full_duplicate_excess,
            "exact_full_payload_duplicate_groups": exact_full_duplicate_groups,
            "exact_pair_duplicate_excess": exact_pair_duplicate_excess,
            "full_duplicate_groups_with_distinct_wrapper_indices": (
                duplicate_groups_with_distinct_indices
            ),
            "unique_full_payloads": len(full_payload_counts),
        },
        "observed_at": observed_at,
        "original_quality_metrics": {
            "format_and_content": dict(formatting),
            "language": dict(language),
            "provenance": dict(provenance),
        },
        "original_report_sha256": original_report["report_sha256"],
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": blockers,
        "retrieval_integrity": {
            "content_multiset_sha256": sha256_value(sorted(full_payload_counts.elements())),
            "dataset_viewer_wrapper_index_stable": False,
            "distinct_content_index_collision_groups": distinct_content_index_collisions,
            "fetched_rows": len(wrapped_rows),
            "official_statistics_bengali_rows": official_bengali_rows,
            "row_index_collision_excess": row_index_collision_excess,
            "row_index_collision_groups": row_index_collision_groups,
            "unique_wrapper_row_indices": len(row_index_counts),
        },
        "robustness": {
            "concentration_failure_margin_records": concentration_margin_records,
            "concentration_verdict_robust_to_index_uncertainty": (
                concentration_margin_records > uncertainty_records
            ),
            "index_uncertainty_records": uncertainty_records,
            "trailing_artifact_verdict_robust_to_index_uncertainty": (
                trailing_pipe_records > uncertainty_records
            ),
        },
        "row_level_human_review_claimed": False,
        "schema_version": REPORT_SCHEMA_VERSION,
    }
    report["report_sha256"] = sha256_value(report)
    return report
