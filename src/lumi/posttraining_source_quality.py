"""Raw-text-free quality and provenance audit for bounded Lumi SFT sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_QUALITY_AUDIT_PLAN_SCHEMA_VERSION = (
    "lumi.sft-source-quality-audit-plan.v0.1"
)
SOURCE_QUALITY_AUDIT_REPORT_SCHEMA_VERSION = (
    "lumi.sft-source-quality-audit-report.v0.1"
)
OASST1_REPAIR_AUDIT_REPORT_SCHEMA_VERSION = (
    "lumi.sft-oasst1-repair-audit-report.v0.1"
)
AYA_BENGALI_AUDIT_PLAN_SCHEMA_VERSION = (
    "lumi.sft-aya-bengali-source-audit-plan.v0.1"
)
AYA_BENGALI_AUDIT_REPORT_SCHEMA_VERSION = (
    "lumi.sft-aya-bengali-source-audit-report.v0.1"
)
EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": False,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_bounded_canary_quality_and_provenance_audit_only",
}
EXPECTED_SOURCE_IDS = frozenset(
    {
        "coherelabs_aya_dataset",
        "openassistant_oasst1",
        "openr1_math_220k",
    }
)
EXPECTED_AYA_BENGALI_AUDIT_AUTHORIZATION = {
    "bounded_public_dataset_download_authorized": True,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "pinned_aya_bengali_full_subset_cpu_only_source_audit",
    "unrelated_dataset_access_authorized": False,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


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


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            records.append(value)
    return records


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def load_source_quality_audit_plan(path: str | Path) -> dict[str, Any]:
    """Validate a non-authorizing audit plan for already-fetched canary rows."""

    plan = _load_json(path)
    if plan.get("schema_version") != SOURCE_QUALITY_AUDIT_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported source quality audit plan schema")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("source quality audit authorization changed")
    if plan.get("output") != {
        "contains_raw_training_text": False,
        "production_promotion_permitted": False,
        "record_level_human_review_claimed": False,
    }:
        raise ValueError("source quality audit output boundary changed")

    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("source quality audit must bind exactly three sources")
    if {source.get("source_id") for source in sources} != EXPECTED_SOURCE_IDS:
        raise ValueError("source quality audit source set changed")
    for source in sources:
        if not isinstance(source.get("dataset_id"), str):
            raise ValueError("source quality audit dataset_id is invalid")
        revision = source.get("revision")
        if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
            raise ValueError("source quality audit revision is not pinned")

    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("source quality audit inputs are missing")
    canary = inputs.get("source_canary")
    ai_candidate = inputs.get("ai_assisted_candidate")
    if not isinstance(canary, Mapping) or not isinstance(ai_candidate, Mapping):
        raise ValueError("source quality audit input binding is invalid")
    if canary.get("accepted_records") != 72 or ai_candidate.get("records") != 320:
        raise ValueError("source quality audit record boundary changed")
    for field in (
        "accepted_jsonl_sha256",
        "internal_report_sha256",
        "report_file_sha256",
    ):
        _require_sha256(canary.get(field), f"source_canary.{field}")
    _require_sha256(
        ai_candidate.get("canonical_records_sha256"),
        "ai_assisted_candidate.canonical_records_sha256",
    )
    _require_sha256(
        ai_candidate.get("file_sha256"), "ai_assisted_candidate.file_sha256"
    )
    _require_sha256(
        ai_candidate.get("source_revision"), "ai_assisted_candidate.source_revision"
    )
    raw_files = canary.get("raw_row_files")
    if not isinstance(raw_files, Mapping) or set(raw_files) != EXPECTED_SOURCE_IDS:
        raise ValueError("source quality audit raw row file set changed")
    for source_id, digest in raw_files.items():
        _require_sha256(digest, f"raw_row_files.{source_id}")
    return plan


def load_aya_bengali_audit_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate the exact bounded full-subset Aya Bengali audit plan."""

    plan_path = Path(path)
    root = Path(repository_root).resolve()
    plan = _load_json(plan_path)
    if plan.get("schema_version") != AYA_BENGALI_AUDIT_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya Bengali source audit plan schema")
    if plan.get("authorization") != EXPECTED_AYA_BENGALI_AUDIT_AUTHORIZATION:
        raise ValueError("Aya Bengali audit authorization boundary changed")
    if plan.get("limits") != {
        "expected_filtered_rows": 1534,
        "maximum_fetched_rows": 1534,
        "maximum_network_responses": 50,
        "maximum_single_response_bytes": 8 * 1024 * 1024,
        "maximum_total_response_bytes": 64 * 1024 * 1024,
        "page_size": 50,
    }:
        raise ValueError("Aya Bengali audit limits changed")
    if plan.get("output") != {
        "production_promotion_permitted": False,
        "raw_pages_must_be_outside_git": True,
        "report_contains_raw_training_text": False,
        "row_level_human_review_claimed": False,
    }:
        raise ValueError("Aya Bengali audit output boundary changed")

    source = plan.get("source")
    if source != {
        "config": "default",
        "dataset_id": "CohereLabs/aya_dataset",
        "filter": "\"language_code\"='ben'",
        "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "source_id": "coherelabs_aya_dataset",
        "split": "train",
    }:
        raise ValueError("Aya Bengali audit source binding changed")

    criteria = plan.get("criteria")
    if criteria != {
        "attribution_manifest_required_before_promotion": True,
        "maximum_top_annotator_fraction": 0.5,
        "record_level_human_review_required_before_promotion": True,
        "require_all_language_codes_bengali": True,
        "require_no_empty_prompt_or_target": True,
        "require_no_exact_duplicate_pairs": True,
        "require_no_trailing_pipe_artifacts": True,
        "require_unique_row_indices": True,
    }:
        raise ValueError("Aya Bengali audit criteria changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "audit_implementation",
        "audit_runner",
        "sft_contract",
        "source_registry",
    }:
        raise ValueError("Aya Bengali audit bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya Bengali audit binding is invalid: {name}")
        relative_path = binding.get("path")
        expected_sha256 = _require_sha256(
            binding.get("sha256"),
            f"bindings.{name}.sha256",
        )
        if not isinstance(relative_path, str):
            raise ValueError(f"Aya Bengali audit binding path is invalid: {name}")
        bound_path = (root / relative_path).resolve()
        if root not in bound_path.parents:
            raise ValueError(f"Aya Bengali audit binding escapes repository: {name}")
        if file_sha256(bound_path) != expected_sha256:
            raise ValueError(f"Aya Bengali audit binding changed: {name}")
    return plan


def _raw_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    wrapped_rows = value.get("rows")
    if not isinstance(wrapped_rows, list):
        raise ValueError("raw source payload omitted rows")
    rows: list[dict[str, Any]] = []
    for wrapped in wrapped_rows:
        if not isinstance(wrapped, Mapping):
            raise ValueError("raw source wrapper is invalid")
        row = wrapped.get("row")
        row_index = wrapped.get("row_idx")
        if not isinstance(row, Mapping) or not isinstance(row_index, int):
            raise ValueError("raw source row is invalid")
        rows.append({**row, "_row_index": row_index})
    return rows


def _accepted_source_records(
    records: Sequence[Mapping[str, Any]], source_id: str
) -> list[Mapping[str, Any]]:
    selected = [
        record
        for record in records
        if isinstance(record.get("source"), Mapping)
        and record["source"].get("source_id") == source_id
    ]
    if len(selected) != 24:
        raise ValueError(f"representative source record count changed: {source_id}")
    return selected


def audit_oasst1(
    accepted: Sequence[Mapping[str, Any]], raw_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure context and rank safety without retaining conversation text."""

    rows = _raw_rows(raw_payload)
    by_id = {
        row.get("message_id"): row
        for row in rows
        if isinstance(row.get("message_id"), str)
    }
    rank_counts: Counter[str] = Counter()
    tree_counts: Counter[str] = Counter()
    followup_without_history = 0
    missing_bindings = 0
    for record in accepted:
        source = record.get("source")
        document_id = source.get("document_id") if isinstance(source, Mapping) else None
        answer = by_id.get(document_id)
        parent = by_id.get(answer.get("parent_id")) if isinstance(answer, Mapping) else None
        if not isinstance(answer, Mapping) or not isinstance(parent, Mapping):
            missing_bindings += 1
            continue
        if parent.get("parent_id") is not None:
            followup_without_history += 1
        rank = answer.get("rank")
        rank_counts["null" if rank is None else str(rank)] += 1
        tree_id = answer.get("message_tree_id")
        if isinstance(tree_id, str):
            tree_counts[hashlib.sha256(tree_id.encode()).hexdigest()] += 1

    blockers = []
    if followup_without_history:
        blockers.append("current_pair_adapter_drops_required_ancestor_history")
    if rank_counts.get("null", 0):
        blockers.append("unranked_replies_are_admitted_by_current_adapter")
    if missing_bindings:
        blockers.append("accepted_record_raw_binding_missing")
    blockers.extend(
        [
            "full_thread_selection_and_quality_policy_not_sealed",
            "attribution_manifest_not_built",
        ]
    )
    return {
        "accepted_records": len(accepted),
        "decision": "blocked_after_representative_audit",
        "distinct_message_trees": len(tree_counts),
        "followup_pairs_missing_ancestor_history": followup_without_history,
        "largest_tree_record_count": max(tree_counts.values(), default=0),
        "missing_raw_bindings": missing_bindings,
        "production_blockers": blockers,
        "rank_counts": dict(sorted(rank_counts.items())),
        "root_direct_pairs": len(accepted) - followup_without_history - missing_bindings,
    }


def build_oasst1_repair_audit_report(
    *,
    accepted: Sequence[Mapping[str, Any]],
    adapter_implementation_sha256: str,
    filter_report: Mapping[str, Any],
    metadata_sha256: str,
    observed_at: str,
    raw_payload: Mapping[str, Any],
    raw_rows_sha256: str,
    source_revision: str,
    target_records: int,
) -> dict[str, Any]:
    """Seal a raw-text-free structural receipt for the repaired OASST1 adapter."""

    raw_rows = _raw_rows(raw_payload)
    if raw_payload.get("partial") is not False:
        raise ValueError("OASST1 repair audit requires a complete bounded response")
    if not 1 <= len(raw_rows) <= 100:
        raise ValueError("OASST1 repair raw-row boundary changed")
    if not 1 <= len(accepted) <= target_records <= 24:
        raise ValueError("OASST1 repair accepted-record boundary changed")
    for name, digest in {
        "adapter_implementation_sha256": adapter_implementation_sha256,
        "metadata_sha256": metadata_sha256,
        "raw_rows_sha256": raw_rows_sha256,
    }.items():
        _require_sha256(digest, name)
    if not isinstance(source_revision, str) or _GIT_SHA.fullmatch(source_revision) is None:
        raise ValueError("OASST1 repair source revision is not pinned")
    if filter_report.get("accepted") != len(accepted):
        raise ValueError("OASST1 repair filter accepted count changed")
    if filter_report.get("rejected") != 0:
        raise ValueError("OASST1 repair filter rejected a candidate")
    _require_sha256(filter_report.get("report_sha256"), "filter_report.report_sha256")

    structural = audit_oasst1(accepted, raw_payload)
    rank_counts = structural["rank_counts"]
    structural_passed = (
        structural["followup_pairs_missing_ancestor_history"] == 0
        and structural["missing_raw_bindings"] == 0
        and rank_counts == {"0": len(accepted)}
        and structural["root_direct_pairs"] == len(accepted)
    )
    if not structural_passed:
        raise ValueError("OASST1 repaired adapter did not close structural findings")

    record_ids = [record.get("record_id") for record in accepted]
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("OASST1 repair record ID is invalid")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("OASST1 repair record IDs are not unique")

    tasks = Counter(record.get("task_category") for record in accepted)
    languages = Counter(record.get("language") for record in accepted)
    report: dict[str, Any] = {
        "accepted_candidate_records": len(accepted),
        "accepted_record_ids_sha256": sha256_value(record_ids),
        "adapter": "oasst1_root_direct_rank0_v0.2",
        "authorization": {
            "bulk_data_download_authorized": False,
            "dataset_promotion_authorized": False,
            "external_api_spend_authorized": False,
            "network_access_used": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "existing_pinned_oasst1_slice_structural_reaudit_only",
        },
        "bindings": {
            "adapter_implementation_sha256": adapter_implementation_sha256,
            "filter_report_sha256": filter_report["report_sha256"],
            "metadata_sha256": metadata_sha256,
            "raw_rows_sha256": raw_rows_sha256,
            "source_revision": source_revision,
        },
        "closed_findings": [
            "current_pair_adapter_drops_required_ancestor_history",
            "unranked_replies_are_admitted_by_current_adapter",
        ],
        "decision": "structural_repair_passed_source_still_blocked",
        "distributions": {
            "language": dict(sorted(languages.items())),
            "task": dict(sorted(tasks.items())),
        },
        "input_rows": len(raw_rows),
        "observed_at": observed_at,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": structural["production_blockers"],
        "schema_version": OASST1_REPAIR_AUDIT_REPORT_SCHEMA_VERSION,
        "structural_metrics": {
            "followup_pairs_missing_ancestor_history": 0,
            "missing_raw_bindings": 0,
            "rank_counts": rank_counts,
            "root_direct_pairs": structural["root_direct_pairs"],
        },
        "target_records": target_records,
        "total_source_rows": raw_payload.get("num_rows_total"),
    }
    report["report_sha256"] = sha256_value(report)
    return report


def audit_aya(
    accepted: Sequence[Mapping[str, Any]], raw_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure Bengali annotation concentration and simple format artifacts."""

    rows = _raw_rows(raw_payload)
    by_index = {row["_row_index"]: row for row in rows}
    annotator_counts: Counter[str] = Counter()
    annotation_types: Counter[str] = Counter()
    trailing_pipe_records = 0
    missing_bindings = 0
    for record in accepted:
        source = record.get("source")
        document_id = source.get("document_id") if isinstance(source, Mapping) else None
        if not isinstance(document_id, str) or not document_id.startswith("train-row-"):
            missing_bindings += 1
            continue
        try:
            row_index = int(document_id.removeprefix("train-row-"))
        except ValueError:
            missing_bindings += 1
            continue
        row = by_index.get(row_index)
        if not isinstance(row, Mapping):
            missing_bindings += 1
            continue
        user_id = row.get("user_id")
        annotation_type = row.get("annotation_type")
        target = row.get("targets")
        if isinstance(user_id, str):
            annotator_counts[hashlib.sha256(user_id.encode()).hexdigest()] += 1
        if isinstance(annotation_type, str):
            annotation_types[annotation_type] += 1
        if isinstance(target, str) and target.rstrip().endswith("|"):
            trailing_pipe_records += 1

    top_count = max(annotator_counts.values(), default=0)
    top_fraction = top_count / len(accepted) if accepted else 0.0
    blockers = []
    if top_fraction > 0.5:
        blockers.append("representative_slice_top_annotator_exceeds_half")
    if trailing_pipe_records:
        blockers.append("representative_slice_contains_trailing_format_artifacts")
    if missing_bindings:
        blockers.append("accepted_record_raw_binding_missing")
    blockers.extend(
        [
            "full_bengali_subset_annotator_concentration_not_measured",
            "full_record_level_instruction_compliance_review_not_complete",
            "attribution_manifest_not_built",
        ]
    )
    return {
        "accepted_records": len(accepted),
        "annotation_type_counts": dict(sorted(annotation_types.items())),
        "decision": "blocked_after_representative_audit",
        "missing_raw_bindings": missing_bindings,
        "production_blockers": blockers,
        "top_annotator_fraction": top_fraction,
        "top_annotator_records": top_count,
        "trailing_format_artifact_records": trailing_pipe_records,
        "unique_annotators": len(annotator_counts),
    }


def _nearest_rank_summary(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"maximum": 0, "median": 0, "minimum": 0, "p95": 0}
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        index = max(0, ((len(ordered) * percent + 99) // 100) - 1)
        return ordered[min(index, len(ordered) - 1)]

    return {
        "maximum": ordered[-1],
        "median": percentile(50),
        "minimum": ordered[0],
        "p95": percentile(95),
    }


def _bengali_text_counts(value: str) -> tuple[int, int, bool]:
    alphabetic = 0
    bengali_alphabetic = 0
    has_bengali = False
    for character in value:
        if "\u0980" <= character <= "\u09ff":
            has_bengali = True
            if character.isalpha():
                bengali_alphabetic += 1
        if character.isalpha():
            alphabetic += 1
    return alphabetic, bengali_alphabetic, has_bengali


def build_aya_bengali_source_audit_report(
    *,
    expected_rows: int,
    maximum_top_annotator_fraction: float,
    metadata_post_sha256: str,
    metadata_pre_sha256: str,
    observed_at: str,
    page_inventory: Sequence[Mapping[str, Any]],
    plan_sha256: str,
    response_bytes: int,
    response_count: int,
    source_revision: str,
    wrapped_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a raw-text-free full-subset machine audit for Aya Bengali."""

    if not 1 <= expected_rows <= 1534:
        raise ValueError("Aya Bengali expected row boundary changed")
    if len(wrapped_rows) != expected_rows:
        raise ValueError("Aya Bengali fetched row count is incomplete")
    if maximum_top_annotator_fraction != 0.5:
        raise ValueError("Aya Bengali annotator concentration threshold changed")
    if not isinstance(observed_at, str) or not observed_at:
        raise ValueError("Aya Bengali audit observation date is missing")
    if not isinstance(response_bytes, int) or not 0 < response_bytes <= 64 * 1024 * 1024:
        raise ValueError("Aya Bengali response byte boundary changed")
    if not isinstance(response_count, int) or not 1 <= response_count <= 50:
        raise ValueError("Aya Bengali network response boundary changed")
    if not isinstance(source_revision, str) or _GIT_SHA.fullmatch(source_revision) is None:
        raise ValueError("Aya Bengali source revision is not pinned")
    for name, digest in {
        "metadata_post_sha256": metadata_post_sha256,
        "metadata_pre_sha256": metadata_pre_sha256,
        "plan_sha256": plan_sha256,
    }.items():
        _require_sha256(digest, name)

    expected_offset = 0
    page_rows = 0
    normalized_inventory: list[dict[str, Any]] = []
    for page_number, page in enumerate(page_inventory):
        if not isinstance(page, Mapping):
            raise ValueError("Aya Bengali page inventory entry is invalid")
        filename = page.get("filename")
        offset = page.get("offset")
        rows = page.get("rows")
        size_bytes = page.get("size_bytes")
        digest = _require_sha256(page.get("sha256"), "page.sha256")
        if filename != f"aya-bengali-page-{page_number:03d}.json":
            raise ValueError("Aya Bengali page filename sequence changed")
        if offset != expected_offset:
            raise ValueError("Aya Bengali page offset sequence changed")
        if not isinstance(rows, int) or not 1 <= rows <= 100:
            raise ValueError("Aya Bengali page row count is invalid")
        if not isinstance(size_bytes, int) or not 0 < size_bytes <= 8 * 1024 * 1024:
            raise ValueError("Aya Bengali page byte count is invalid")
        normalized_inventory.append(
            {
                "filename": filename,
                "offset": offset,
                "rows": rows,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        page_rows += rows
        expected_offset += rows
    if not 1 <= len(normalized_inventory) <= 31 or page_rows != expected_rows:
        raise ValueError("Aya Bengali page inventory is incomplete")

    row_indices: list[int] = []
    annotators: Counter[str] = Counter()
    annotation_types: Counter[str] = Counter()
    prompt_hashes: Counter[str] = Counter()
    target_hashes: Counter[str] = Counter()
    pair_hashes: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    empty_prompts = 0
    empty_targets = 0
    invalid_rows = 0
    language_code_mismatches = 0
    language_label_mismatches = 0
    missing_annotators = 0
    identical_prompt_target = 0
    trailing_pipe_records = 0
    url_records = 0
    prompt_records_with_bengali = 0
    target_records_with_bengali = 0
    prompt_alphabetic = 0
    prompt_bengali_alphabetic = 0
    target_alphabetic = 0
    target_bengali_alphabetic = 0

    for wrapped in wrapped_rows:
        if not isinstance(wrapped, Mapping):
            invalid_rows += 1
            continue
        row_index = wrapped.get("row_idx")
        row = wrapped.get("row")
        if not isinstance(row_index, int) or not isinstance(row, Mapping):
            invalid_rows += 1
            continue
        row_indices.append(row_index)
        if row.get("language_code") != "ben":
            language_code_mismatches += 1
        if row.get("language") != "Bengali":
            language_label_mismatches += 1

        annotation_type = row.get("annotation_type")
        if annotation_type in {"original-annotations", "re-annotations"}:
            annotation_types[str(annotation_type)] += 1
        else:
            annotation_types["other_or_missing"] += 1
        user_id = row.get("user_id")
        if isinstance(user_id, str) and user_id:
            annotators[hashlib.sha256(user_id.encode()).hexdigest()] += 1
        else:
            missing_annotators += 1

        prompt = row.get("inputs")
        target = row.get("targets")
        if not isinstance(prompt, str):
            empty_prompts += 1
            prompt = ""
        elif not prompt.strip():
            empty_prompts += 1
        if not isinstance(target, str):
            empty_targets += 1
            target = ""
        elif not target.strip():
            empty_targets += 1

        prompt_lengths.append(len(prompt))
        target_lengths.append(len(target))
        prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
        target_digest = hashlib.sha256(target.encode()).hexdigest()
        prompt_hashes[prompt_digest] += 1
        target_hashes[target_digest] += 1
        pair_hashes[sha256_value([prompt_digest, target_digest])] += 1
        if prompt == target:
            identical_prompt_target += 1
        if target.rstrip().endswith("|"):
            trailing_pipe_records += 1
        if re.search(r"https?://", prompt) or re.search(r"https?://", target):
            url_records += 1

        alphabetic, bengali, has_bengali = _bengali_text_counts(prompt)
        prompt_alphabetic += alphabetic
        prompt_bengali_alphabetic += bengali
        if has_bengali:
            prompt_records_with_bengali += 1
        alphabetic, bengali, has_bengali = _bengali_text_counts(target)
        target_alphabetic += alphabetic
        target_bengali_alphabetic += bengali
        if has_bengali:
            target_records_with_bengali += 1

    duplicate_row_indices = len(row_indices) - len(set(row_indices))
    duplicate_prompt_records = sum(count - 1 for count in prompt_hashes.values())
    duplicate_target_records = sum(count - 1 for count in target_hashes.values())
    duplicate_pair_records = sum(count - 1 for count in pair_hashes.values())
    top_annotator_records = max(annotators.values(), default=0)
    top_annotator_fraction = top_annotator_records / expected_rows

    blockers: list[str] = []
    closed_findings: list[str] = []
    if top_annotator_fraction <= maximum_top_annotator_fraction:
        closed_findings.append("full_bengali_subset_annotator_concentration_within_limit")
    else:
        blockers.append("full_bengali_subset_top_annotator_exceeds_half")
    if trailing_pipe_records:
        blockers.append("full_bengali_subset_contains_trailing_pipe_artifacts")
    else:
        closed_findings.append("full_bengali_subset_contains_no_trailing_pipe_artifacts")
    if empty_prompts or empty_targets or invalid_rows:
        blockers.append("full_bengali_subset_contains_empty_or_invalid_rows")
    if language_code_mismatches or language_label_mismatches:
        blockers.append("full_bengali_subset_language_labels_are_inconsistent")
    if duplicate_row_indices:
        blockers.append("full_bengali_subset_contains_duplicate_row_indices")
    if duplicate_pair_records:
        blockers.append("full_bengali_subset_exact_pairs_require_deduplication")
    blockers.extend(
        [
            "full_record_level_instruction_compliance_review_not_complete",
            "attribution_manifest_not_built",
        ]
    )

    report: dict[str, Any] = {
        "authorization": {
            **EXPECTED_AYA_BENGALI_AUDIT_AUTHORIZATION,
            "network_response_bytes": response_bytes,
            "network_responses": response_count,
        },
        "bindings": {
            "metadata_post_sha256": metadata_post_sha256,
            "metadata_pre_sha256": metadata_pre_sha256,
            "page_inventory_sha256": sha256_value(normalized_inventory),
            "plan_sha256": plan_sha256,
            "source_revision": source_revision,
        },
        "closed_findings": closed_findings,
        "decision": "full_subset_machine_audit_complete_source_still_blocked",
        "duplicate_metrics": {
            "duplicate_pair_records": duplicate_pair_records,
            "duplicate_prompt_records": duplicate_prompt_records,
            "duplicate_row_indices": duplicate_row_indices,
            "duplicate_target_records": duplicate_target_records,
        },
        "format_and_content_metrics": {
            "empty_prompt_records": empty_prompts,
            "empty_target_records": empty_targets,
            "identical_prompt_target_records": identical_prompt_target,
            "invalid_row_wrappers": invalid_rows,
            "prompt_character_lengths": _nearest_rank_summary(prompt_lengths),
            "target_character_lengths": _nearest_rank_summary(target_lengths),
            "trailing_pipe_records": trailing_pipe_records,
            "url_records": url_records,
        },
        "language_metrics": {
            "language_code_mismatches": language_code_mismatches,
            "language_label_mismatches": language_label_mismatches,
            "prompt_bengali_alphabetic_fraction": (
                prompt_bengali_alphabetic / prompt_alphabetic
                if prompt_alphabetic
                else 0.0
            ),
            "prompt_records_with_bengali": prompt_records_with_bengali,
            "target_bengali_alphabetic_fraction": (
                target_bengali_alphabetic / target_alphabetic
                if target_alphabetic
                else 0.0
            ),
            "target_records_with_bengali": target_records_with_bengali,
        },
        "observed_at": observed_at,
        "page_count": len(normalized_inventory),
        "production_eligible": False,
        "production_promotions": 0,
        "provenance_metrics": {
            "annotation_type_counts": dict(sorted(annotation_types.items())),
            "attributed_rows": sum(annotators.values()),
            "missing_annotator_rows": missing_annotators,
            "top_annotator_fraction": top_annotator_fraction,
            "top_annotator_records": top_annotator_records,
            "unique_annotators": len(annotators),
        },
        "raw_text_retained": False,
        "remaining_production_blockers": blockers,
        "row_level_human_review_claimed": False,
        "schema_version": AYA_BENGALI_AUDIT_REPORT_SCHEMA_VERSION,
        "source_rows": expected_rows,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def audit_openr1(
    accepted: Sequence[Mapping[str, Any]], raw_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure trace-field, completeness, dependency, and source risks."""

    rows = _raw_rows(raw_payload)
    by_uuid = {
        row.get("uuid"): row for row in rows if isinstance(row.get("uuid"), str)
    }
    source_counts: Counter[str] = Counter()
    reference_solution_matches = 0
    mixed_completeness = 0
    image_dependencies = 0
    external_urls = 0
    short_solutions = 0
    missing_bindings = 0
    for record in accepted:
        source = record.get("source")
        document_id = source.get("document_id") if isinstance(source, Mapping) else None
        row = by_uuid.get(document_id)
        messages = record.get("messages")
        if not isinstance(row, Mapping) or not isinstance(messages, list) or not messages:
            missing_bindings += 1
            continue
        answer = messages[-1]
        answer_text = answer.get("content") if isinstance(answer, Mapping) else None
        solution = row.get("solution")
        problem = row.get("problem")
        if isinstance(answer_text, str) and answer_text == solution:
            reference_solution_matches += 1
        completeness = row.get("is_reasoning_complete")
        if not isinstance(completeness, list) or not completeness or not all(
            item is True for item in completeness
        ):
            mixed_completeness += 1
        combined = "".join(
            item for item in (problem, solution) if isinstance(item, str)
        )
        if "![](" in combined:
            image_dependencies += 1
        if re.search(r"https?://", combined):
            external_urls += 1
        if isinstance(solution, str) and len(solution) < 80:
            short_solutions += 1
        upstream_source = row.get("source")
        if isinstance(upstream_source, str):
            source_counts[upstream_source] += 1

    blockers = []
    if reference_solution_matches:
        blockers.append("current_adapter_uses_reference_solution_not_verified_generation")
    if mixed_completeness:
        blockers.append("representative_slice_contains_mixed_completeness_vectors")
    if image_dependencies:
        blockers.append("representative_slice_contains_external_image_dependencies")
    if short_solutions:
        blockers.append("representative_slice_contains_answer_only_or_short_solutions")
    if missing_bindings:
        blockers.append("accepted_record_raw_binding_missing")
    blockers.extend(
        [
            "underlying_problem_source_provenance_not_resolved_per_row",
            "full_dataset_4096_token_fit_not_measured",
            "attribution_manifest_not_built",
        ]
    )
    return {
        "accepted_assistant_matches_reference_solution": reference_solution_matches,
        "accepted_records": len(accepted),
        "decision": "blocked_after_representative_audit",
        "external_image_dependency_records": image_dependencies,
        "external_url_records": external_urls,
        "missing_raw_bindings": missing_bindings,
        "mixed_generation_completeness_records": mixed_completeness,
        "production_blockers": blockers,
        "short_reference_solution_records": short_solutions,
        "upstream_source_counts": dict(sorted(source_counts.items())),
    }


def _distribution(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(field)
        if isinstance(value, str):
            counts[value] += 1
    return dict(sorted(counts.items()))


def build_candidate_coverage_gap(
    *,
    source_canary_records: Sequence[Mapping[str, Any]],
    ai_candidate_records: Sequence[Mapping[str, Any]],
    target_mix: Mapping[str, float],
) -> dict[str, Any]:
    """Describe the diagnostic pool without presenting it as a trainable mix."""

    all_records = [*source_canary_records, *ai_candidate_records]
    task_counts = Counter(_distribution(all_records, "task_category"))
    total = len(all_records)
    task_share = {
        task: task_counts.get(task, 0) / total if total else 0.0
        for task in sorted(target_mix)
    }
    zero_coverage = [task for task in sorted(target_mix) if task_counts.get(task, 0) == 0]
    sparse = [
        task
        for task in sorted(target_mix)
        if 0 < task_share[task] < float(target_mix[task]) * 0.25
    ]
    return {
        "ai_assisted_candidate_records": len(ai_candidate_records),
        "combined_diagnostic_records": total,
        "diagnostic_only_not_a_trainable_mix": True,
        "source_canary_records": len(source_canary_records),
        "sparse_below_one_quarter_of_target_share": sparse,
        "task_counts": dict(sorted(task_counts.items())),
        "task_share": task_share,
        "target_share": dict(sorted(target_mix.items())),
        "zero_coverage_tasks": zero_coverage,
    }


def run_source_quality_audit(
    *,
    plan: Mapping[str, Any],
    canary_attempt: str | Path,
    ai_candidate_jsonl: str | Path,
    target_mix: Mapping[str, float],
) -> dict[str, Any]:
    """Audit the exact existing canary and emit no source text."""

    attempt = Path(canary_attempt)
    accepted_path = attempt / "accepted-canary.jsonl"
    report_path = attempt / "canary-report.json"
    raw_directory = attempt / "raw"
    inputs = plan["inputs"]
    canary_binding = inputs["source_canary"]
    ai_binding = inputs["ai_assisted_candidate"]
    if file_sha256(accepted_path) != canary_binding["accepted_jsonl_sha256"]:
        raise ValueError("source canary accepted artifact changed")
    if file_sha256(report_path) != canary_binding["report_file_sha256"]:
        raise ValueError("source canary report file changed")
    if file_sha256(ai_candidate_jsonl) != ai_binding["file_sha256"]:
        raise ValueError("AI-assisted candidate artifact changed")

    canary_report = _load_json(report_path)
    if canary_report.get("report_sha256") != canary_binding["internal_report_sha256"]:
        raise ValueError("source canary internal report seal changed")
    if canary_report.get("accepted_artifact", {}).get("sha256") != canary_binding[
        "accepted_jsonl_sha256"
    ]:
        raise ValueError("source canary report no longer binds accepted artifact")

    raw_paths = {
        "coherelabs_aya_dataset": raw_directory / "coherelabs_aya_dataset-rows.json",
        "openassistant_oasst1": raw_directory / "openassistant_oasst1-rows.json",
        "openr1_math_220k": raw_directory / "openr1_math_220k-rows.json",
    }
    for source_id, path in raw_paths.items():
        if file_sha256(path) != canary_binding["raw_row_files"][source_id]:
            raise ValueError(f"source canary raw rows changed: {source_id}")

    accepted = _load_jsonl(accepted_path)
    ai_candidate = _load_jsonl(ai_candidate_jsonl)
    if len(accepted) != canary_binding["accepted_records"]:
        raise ValueError("source canary accepted record count changed")
    if len(ai_candidate) != ai_binding["records"]:
        raise ValueError("AI-assisted candidate record count changed")
    if sha256_value(ai_candidate) != ai_binding["canonical_records_sha256"]:
        raise ValueError("AI-assisted candidate canonical record seal changed")
    record_ids = [record.get("record_id") for record in accepted]
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("source canary record ID is invalid")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("source canary record IDs are not unique")

    source_results = {
        "coherelabs_aya_dataset": audit_aya(
            _accepted_source_records(accepted, "coherelabs_aya_dataset"),
            _load_json(raw_paths["coherelabs_aya_dataset"]),
        ),
        "openassistant_oasst1": audit_oasst1(
            _accepted_source_records(accepted, "openassistant_oasst1"),
            _load_json(raw_paths["openassistant_oasst1"]),
        ),
        "openr1_math_220k": audit_openr1(
            _accepted_source_records(accepted, "openr1_math_220k"),
            _load_json(raw_paths["openr1_math_220k"]),
        ),
    }
    issue_count = sum(
        len(result["production_blockers"]) for result in source_results.values()
    )
    report = {
        "audit_complete": True,
        "audit_method": (
            "codex_assisted_structural_and_provenance_audit_with_deterministic_metrics"
        ),
        "authorization": plan["authorization"],
        "bindings": {
            "ai_assisted_candidate_canonical_sha256": ai_binding[
                "canonical_records_sha256"
            ],
            "ai_assisted_candidate_file_sha256": ai_binding["file_sha256"],
            "ai_assisted_candidate_source_revision": ai_binding["source_revision"],
            "source_canary_accepted_sha256": canary_binding["accepted_jsonl_sha256"],
            "source_canary_report_sha256": canary_binding["internal_report_sha256"],
        },
        "candidate_coverage_gap": build_candidate_coverage_gap(
            source_canary_records=accepted,
            ai_candidate_records=ai_candidate,
            target_mix=target_mix,
        ),
        "human_record_level_review_claimed": False,
        "observed_at": plan["observed_at"],
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "representative_audit_outcome": "complete_failed_closed",
        "schema_version": SOURCE_QUALITY_AUDIT_REPORT_SCHEMA_VERSION,
        "source_results": source_results,
        "total_production_blocker_findings": issue_count,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def load_source_quality_audit_evidence(path: str | Path) -> dict[str, Any]:
    """Validate tracked raw-text-free evidence from the representative audit."""

    evidence = _load_json(path)
    if evidence.get("schema_version") != SOURCE_QUALITY_AUDIT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported source quality audit report schema")
    if evidence.get("audit_complete") is not True:
        raise ValueError("source quality audit is incomplete")
    if evidence.get("representative_audit_outcome") != "complete_failed_closed":
        raise ValueError("source quality audit outcome changed")
    if evidence.get("production_promotions") != 0:
        raise ValueError("source quality audit may not promote sources")
    if evidence.get("human_record_level_review_claimed") is not False:
        raise ValueError("source quality audit falsely claims human review")
    if evidence.get("raw_training_text_retained") is not False:
        raise ValueError("source quality audit evidence retained raw text")
    source_results = evidence.get("source_results")
    if not isinstance(source_results, Mapping) or set(source_results) != EXPECTED_SOURCE_IDS:
        raise ValueError("source quality audit result set changed")
    if any(
        result.get("decision") != "blocked_after_representative_audit"
        for result in source_results.values()
        if isinstance(result, Mapping)
    ):
        raise ValueError("source quality audit decision changed")
    supplied = evidence.get("report_sha256")
    _require_sha256(supplied, "report_sha256")
    unsigned = dict(evidence)
    unsigned.pop("report_sha256")
    if sha256_value(unsigned) != supplied:
        raise ValueError("source quality audit report seal changed")
    return evidence
