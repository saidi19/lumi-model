"""Fail-closed OpenR1 verified-generation selection and audit helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

OPENR1_AUDIT_PLAN_SCHEMA_VERSION = (
    "lumi.sft-openr1-verified-trace-audit-plan.v0.1"
)
OPENR1_AUDIT_REPORT_SCHEMA_VERSION = (
    "lumi.sft-openr1-verified-trace-audit-report.v0.1"
)
EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": False,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "existing_pinned_openr1_slice_verified_trace_reaudit_only",
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_SOURCE = {
    "config": "default",
    "dataset_id": "open-r1/OpenR1-Math-220k",
    "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
    "source_id": "openr1_math_220k",
    "split": "train",
}
EXPECTED_LIMITS = {
    "context_window": 4096,
    "maximum_raw_rows": 24,
    "maximum_selected_records": 24,
}
EXPECTED_CRITERIA = {
    "allow_external_image_dependency": False,
    "allow_external_url_dependency": False,
    "attribution_manifest_required_before_promotion": True,
    "full_dataset_4096_fit_required_before_promotion": True,
    "record_level_quality_review_required_before_promotion": True,
    "require_aligned_generation_verifier_vectors": True,
    "require_nonempty_upstream_problem_source": True,
    "require_reasoning_complete": True,
    "require_trace_different_from_reference_solution": True,
    "require_verified_generation_field": "generations",
    "require_verifier_pass": "correctness_math_verify",
}
EXPECTED_OUTPUT = {
    "contains_raw_training_text": False,
    "production_promotion_permitted": False,
    "selected_records_persisted": False,
}
EXPECTED_BINDINGS = frozenset(
    {
        "audit_implementation",
        "audit_runner",
        "chat_template",
        "metadata_post",
        "metadata_pre",
        "raw_rows",
        "sft_contract",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
)

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URL = re.compile(r"https?://", re.IGNORECASE)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)|<img\b", re.IGNORECASE)


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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def load_openr1_audit_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load and verify the exact non-network OpenR1 re-audit plan."""

    root = Path(repository_root).resolve()
    workspace = root.parent
    plan = _load_json(path)
    if plan.get("schema_version") != OPENR1_AUDIT_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 verified-trace audit plan")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenR1 audit authorization boundary changed")
    if plan.get("source") != EXPECTED_SOURCE:
        raise ValueError("OpenR1 audit source binding changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenR1 audit limits changed")
    if plan.get("criteria") != EXPECTED_CRITERIA:
        raise ValueError("OpenR1 audit criteria changed")
    if plan.get("output") != EXPECTED_OUTPUT:
        raise ValueError("OpenR1 audit output boundary changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenR1 audit binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenR1 audit binding must be an object: {name}")
        relative_path = binding.get("path")
        expected_sha256 = _require_sha256(
            binding.get("sha256"),
            f"bindings.{name}.sha256",
        )
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"OpenR1 audit binding path is invalid: {name}")
        resolved = (root / relative_path).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"OpenR1 audit binding escapes the workspace: {name}")
        if not resolved.is_file():
            raise ValueError(f"OpenR1 audit binding is missing: {name}")
        if file_sha256(resolved) != expected_sha256:
            raise ValueError(f"OpenR1 audit binding changed: {name}")
    return plan


def _dependency_flags(text: str) -> tuple[bool, bool]:
    return _MARKDOWN_IMAGE.search(text) is not None, _URL.search(text) is not None


def _make_record(
    *,
    generation_index: int,
    problem: str,
    revision: str,
    row_index: int,
    trace: str,
    upstream_source: str,
    uuid: str,
) -> dict[str, Any]:
    messages = [
        {"content": problem, "role": "user"},
        {"content": trace, "role": "assistant"},
    ]
    identity = {
        "dataset_id": EXPECTED_SOURCE["dataset_id"],
        "generation_index": generation_index,
        "revision": revision,
        "row_index": row_index,
        "uuid": uuid,
    }
    record_digest = sha256_value(identity)
    transform = {
        **identity,
        "adapter": "openr1_verified_generation_v0.2",
        "output_schema": "lumi.sft.v0.1",
        "trace_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
    }
    return {
        "generation_provenance": {
            "generation_index": generation_index,
            "reasoning_complete": True,
            "trace_field": "generations",
            "upstream_source_sha256": hashlib.sha256(
                upstream_source.encode("utf-8")
            ).hexdigest(),
            "verifier_field": "correctness_math_verify",
            "verifier_passed": True,
        },
        "language": "en",
        "messages": messages,
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": sha256_value(messages),
            "transform_sha256": sha256_value(transform),
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": "openr1_verified_generation_deterministic_gate_v0.2",
            "review_status": "accepted",
            "score": 0.9,
            "secret_scan": "passed",
        },
        "record_id": f"openr1_math_220k-{record_digest[:24]}",
        "safety": {
            "policy_version": "lumi-sft-canary-basic-pattern-scan-v0.1",
            "risk_labels": [],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": EXPECTED_SOURCE["dataset_id"],
            "document_id": uuid,
            "license_id": "apache-2.0",
            "revision": revision,
            "source_id": EXPECTED_SOURCE["source_id"],
            "uri": (
                "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/"
                f"viewer/default/train?row={row_index}"
            ),
        },
        "split": "train",
        "task_category": "reasoning_math_science",
    }


def select_verified_openr1_records(
    wrapped_rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    maximum_selected_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select verifier-passed generated traces and emit only aggregate metrics."""

    if _GIT_SHA.fullmatch(revision) is None:
        raise ValueError("OpenR1 source revision is not pinned")
    if not 1 <= len(wrapped_rows) <= EXPECTED_LIMITS["maximum_raw_rows"]:
        raise ValueError("OpenR1 raw row boundary changed")
    if not 1 <= maximum_selected_records <= EXPECTED_LIMITS["maximum_selected_records"]:
        raise ValueError("OpenR1 selected-record boundary changed")

    records: list[dict[str, Any]] = []
    row_indices: list[int] = []
    uuids: list[str] = []
    source_counts: Counter[str] = Counter()
    generation_index_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    candidate_trace_count = 0
    exact_reference_match_candidates = 0
    problem_image_dependencies = 0
    problem_url_dependencies = 0
    trace_image_dependencies = 0
    trace_url_dependencies = 0
    aligned_vector_rows = 0
    complete_row_provenance = 0

    for wrapped in wrapped_rows:
        if not isinstance(wrapped, Mapping):
            rejection_counts["invalid_row_wrapper"] += 1
            continue
        row = wrapped.get("row")
        row_index = wrapped.get("row_idx")
        if not isinstance(row, Mapping) or not isinstance(row_index, int):
            rejection_counts["invalid_row_binding"] += 1
            continue
        row_indices.append(row_index)

        problem = row.get("problem")
        solution = row.get("solution")
        uuid = row.get("uuid")
        upstream_source = row.get("source")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (problem, solution, uuid, upstream_source)
        ):
            rejection_counts["missing_row_level_provenance_or_text"] += 1
            continue
        uuids.append(uuid)
        complete_row_provenance += 1
        source_counts[upstream_source] += 1

        problem_image, problem_url = _dependency_flags(problem)
        if problem_image:
            problem_image_dependencies += 1
        if problem_url:
            problem_url_dependencies += 1
        if problem_image or problem_url:
            rejection_counts["problem_external_dependency"] += 1
            continue

        generations = row.get("generations")
        completeness = row.get("is_reasoning_complete")
        verifier = row.get("correctness_math_verify")
        if not all(isinstance(value, list) for value in (generations, completeness, verifier)):
            rejection_counts["missing_generation_verifier_vectors"] += 1
            continue
        if not generations or not (
            len(generations) == len(completeness) == len(verifier)
        ):
            rejection_counts["misaligned_generation_verifier_vectors"] += 1
            continue
        aligned_vector_rows += 1

        selected: tuple[int, str] | None = None
        for generation_index, trace in enumerate(generations):
            if (
                completeness[generation_index] is not True
                or verifier[generation_index] is not True
                or not isinstance(trace, str)
                or not trace.strip()
            ):
                continue
            candidate_trace_count += 1
            if trace == solution:
                exact_reference_match_candidates += 1
                continue
            trace_image, trace_url = _dependency_flags(trace)
            if trace_image:
                trace_image_dependencies += 1
            if trace_url:
                trace_url_dependencies += 1
            if trace_image or trace_url:
                continue
            selected = (generation_index, trace)
            break

        if selected is None:
            rejection_counts["no_safe_verified_complete_generation"] += 1
            continue
        generation_index, trace = selected
        generation_index_counts[str(generation_index)] += 1
        records.append(
            _make_record(
                generation_index=generation_index,
                problem=problem,
                revision=revision,
                row_index=row_index,
                trace=trace,
                upstream_source=upstream_source,
                uuid=uuid,
            )
        )
        if len(records) == maximum_selected_records:
            break

    duplicate_row_indices = len(row_indices) - len(set(row_indices))
    duplicate_uuids = len(uuids) - len(set(uuids))
    record_ids = [record["record_id"] for record in records]
    duplicate_selected_records = len(record_ids) - len(set(record_ids))
    metrics = {
        "aligned_generation_verifier_rows": aligned_vector_rows,
        "candidate_verified_complete_traces": candidate_trace_count,
        "complete_row_provenance_records": complete_row_provenance,
        "duplicate_raw_row_indices": duplicate_row_indices,
        "duplicate_raw_uuids": duplicate_uuids,
        "duplicate_selected_records": duplicate_selected_records,
        "exact_reference_match_candidates_rejected": exact_reference_match_candidates,
        "problem_external_image_records": problem_image_dependencies,
        "problem_external_url_records": problem_url_dependencies,
        "raw_rows": len(wrapped_rows),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_generation_index_counts": dict(sorted(generation_index_counts.items())),
        "selected_record_ids_sha256": sha256_value(sorted(record_ids)),
        "selected_records": len(records),
        "trace_external_image_candidates_rejected": trace_image_dependencies,
        "trace_external_url_candidates_rejected": trace_url_dependencies,
        "upstream_source_counts": dict(sorted(source_counts.items())),
    }
    return records, metrics


def build_openr1_audit_report(
    *,
    encoding_report: Mapping[str, Any],
    filter_report: Mapping[str, Any],
    metadata_sha256: str,
    observed_at: str,
    plan_sha256: str,
    raw_rows_sha256: str,
    selection_metrics: Mapping[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    """Build a sealed raw-text-free receipt for the bounded repaired adapter."""

    for name, digest in {
        "metadata_sha256": metadata_sha256,
        "plan_sha256": plan_sha256,
        "raw_rows_sha256": raw_rows_sha256,
    }.items():
        _require_sha256(digest, name)
    if _GIT_SHA.fullmatch(source_revision) is None:
        raise ValueError("OpenR1 report source revision is not pinned")
    selected = selection_metrics.get("selected_records")
    if not isinstance(selected, int) or selected < 0:
        raise ValueError("OpenR1 selected-record metric is invalid")
    filter_accepted = filter_report.get("accepted")
    filter_rejected = filter_report.get("rejected")
    if (
        not isinstance(filter_accepted, int)
        or not isinstance(filter_rejected, int)
        or filter_accepted < 0
        or filter_rejected < 0
        or filter_accepted + filter_rejected != selected
    ):
        raise ValueError("OpenR1 filter result does not reconcile selected records")
    if encoding_report.get("total") != filter_accepted:
        raise ValueError("OpenR1 encoding input does not match filtered records")

    blockers: list[str] = []
    rejection_counts = selection_metrics.get("rejection_counts")
    if isinstance(rejection_counts, Mapping) and rejection_counts:
        blockers.append("bounded_slice_contains_rows_without_safe_verified_trace")
    if encoding_report.get("rejected"):
        blockers.append("bounded_slice_contains_4096_token_overflow")
    if filter_report.get("rejected"):
        blockers.append("bounded_slice_contains_normal_filter_rejections")
    blockers.extend(
        [
            "full_dataset_4096_token_fit_not_measured",
            "underlying_problem_source_rights_and_attribution_not_resolved_per_row",
            "record_level_quality_review_not_complete",
            "attribution_manifest_not_built",
        ]
    )
    report = {
        "adapter_repair": {
            "reference_solution_field_used": False,
            "selected_trace_field": "generations",
            "verifier_field": "correctness_math_verify",
        },
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "metadata_sha256": metadata_sha256,
            "plan_sha256": plan_sha256,
            "raw_rows_sha256": raw_rows_sha256,
            "source_revision": source_revision,
        },
        "decision": "bounded_verified_trace_selection_completed_source_still_blocked",
        "encoding_metrics": dict(encoding_report),
        "filter_metrics": {
            "accepted": filter_report.get("accepted"),
            "rejected": filter_report.get("rejected"),
            "rejection_counts": filter_report.get("rejection_counts", {}),
            "report_sha256": filter_report.get("report_sha256"),
        },
        "observed_at": observed_at,
        "production_eligible": False,
        "production_promotions": 0,
        "raw_text_retained": False,
        "remaining_production_blockers": blockers,
        "schema_version": OPENR1_AUDIT_REPORT_SCHEMA_VERSION,
        "selection_metrics": dict(selection_metrics),
    }
    report["report_sha256"] = sha256_value(report)
    return report
