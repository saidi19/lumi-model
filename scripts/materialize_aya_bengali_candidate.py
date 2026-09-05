#!/usr/bin/env python3
"""Materialize the exact approved Aya Bengali private candidate and review packet."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from lumi.posttraining_aya_materialization import (
    EXPECTED_PREFLIGHT,
    build_attribution_manifest,
    build_materialization_report,
    load_aya_materialization_plan,
    load_exact_aya_pages,
    render_owner_review_html,
    validate_materialization_report,
    verify_exact_preflight,
)
from lumi.posttraining_aya_repair import (
    build_aya_sft_records,
    finalize_aya_repair_candidates,
    resolve_workspace_binding,
    select_aya_repair_candidates,
    verify_aya_input_receipts,
)
from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst1_full_run import (
    deterministic_owner_sample,
    file_sha256,
    load_runtime_components,
    measure_token_fit,
    sha256_value,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya materialization artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _artifact_entry(
    *,
    name: str,
    path: Path,
    private_root: Path,
    contains_raw_training_text: bool,
) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "contains_raw_training_text": contains_raw_training_text,
        "name": name,
        "path": path.relative_to(private_root.parent.parent.parent).as_posix(),
        "sha256": file_sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_aya_materialization_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    preflight_path = resolve_workspace_binding(root, bindings["preflight_report"])
    preflight = _load_json(preflight_path)
    if file_sha256(preflight_path) != EXPECTED_PREFLIGHT["file_sha256"]:
        raise ValueError("Aya preflight file differs from owner approval")
    verify_exact_preflight(preflight)

    inventory_path = resolve_workspace_binding(root, bindings["input_inventory"])
    audit_path = resolve_workspace_binding(root, bindings["input_audit_report"])
    reconciliation_path = resolve_workspace_binding(root, bindings["reconciliation_receipt"])
    verify_aya_input_receipts(
        audit_report=_load_json(audit_path),
        reconciliation=_load_json(reconciliation_path),
    )
    wrapped_rows = load_exact_aya_pages(
        inventory_path,
        expected_objects=plan["limits"]["source_page_objects"],
        expected_rows=plan["limits"]["source_rows"],
    )
    selection = select_aya_repair_candidates(wrapped_rows)
    records, annotators = build_aya_sft_records(selection.candidates)
    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, _filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=source_registry,
    )
    four_k_accepted, _initial_token_metrics = measure_token_fit(
        structurally_accepted,
        template=template,
        tokenizer=tokenizer,
    )
    final_records, _concentration_metrics = finalize_aya_repair_candidates(
        four_k_accepted,
        annotators=annotators,
    )
    final_records, token_metrics = measure_token_fit(
        final_records,
        template=template,
        tokenizer=tokenizer,
    )
    final_records = [dict(record) for record in final_records]
    final_records.sort(key=lambda row: row["record_id"])
    final_ids = sorted(str(record["record_id"]) for record in final_records)
    if (
        len(final_records) != plan["limits"]["maximum_candidate_records"]
        or token_metrics["supervised_tokens"] != plan["limits"]["maximum_supervised_tokens"]
        or sha256_value(final_ids) != EXPECTED_PREFLIGHT["record_ids_sha256"]
        or token_metrics["token_ledger_sha256"] != EXPECTED_PREFLIGHT["token_ledger_sha256"]
    ):
        raise ValueError("Aya materialization did not reproduce the approved candidate")

    candidates_by_id = {str(row["record_id"]): row for row in selection.candidates}
    attribution = build_attribution_manifest(
        final_records,
        candidates_by_id=candidates_by_id,
    )
    owner_sample = deterministic_owner_sample(
        final_records,
        maximum_records=plan["limits"]["owner_sample_records"],
    )
    sample_ids_sha256 = sha256_value(sorted(str(record["record_id"]) for record in owner_sample))
    expected_sample_sha256 = preflight["owner_review"]["sample_record_ids_sha256"]
    if sample_ids_sha256 != expected_sample_sha256:
        raise ValueError("Aya owner-review sample differs from the approved preflight")

    private_root = (root / plan["output"]["private_artifact_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    if private_root.exists() and any(private_root.rglob("*")):
        raise FileExistsError(f"refusing non-empty Aya private artifact root: {private_root}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite Aya materialization report: {report_path}")

    candidate_path = private_root / plan["output"]["candidate_filename"]
    attribution_path = private_root / plan["output"]["attribution_manifest_filename"]
    review_path = private_root / plan["output"]["owner_review_filename"]
    _write_bytes_atomic(candidate_path, _jsonl_bytes(final_records))
    _write_bytes_atomic(attribution_path, _jsonl_bytes(attribution))
    review_html = render_owner_review_html(
        owner_sample,
        attribution_file_sha256=file_sha256(attribution_path),
        candidate_file_sha256=file_sha256(candidate_path),
        sample_record_ids_sha256=sample_ids_sha256,
    )
    _write_bytes_atomic(review_path, review_html.encode("utf-8"))

    artifacts = [
        _artifact_entry(
            name="accepted_records",
            path=candidate_path,
            private_root=private_root,
            contains_raw_training_text=True,
        ),
        _artifact_entry(
            name="attribution_manifest",
            path=attribution_path,
            private_root=private_root,
            contains_raw_training_text=False,
        ),
        _artifact_entry(
            name="owner_review_packet",
            path=review_path,
            private_root=private_root,
            contains_raw_training_text=True,
        ),
    ]
    report = build_materialization_report(
        artifacts=artifacts,
        candidate_records=len(final_records),
        owner_sample_record_ids_sha256=sample_ids_sha256,
        owner_sample_records=len(owner_sample),
        plan_file_sha256=file_sha256(plan_path),
        supervised_tokens=token_metrics["supervised_tokens"],
    )
    elapsed = time.monotonic() - started
    if elapsed > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("Aya materialization exceeded the authorized runtime")
    report["runtime_seconds"] = round(elapsed, 3)
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    validate_materialization_report(report)
    _write_bytes_atomic(
        report_path,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        "PASS: Aya Bengali private candidate materialized "
        f"records={len(final_records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"private_artifacts={len(artifacts)} owner_sample={len(owner_sample)} "
        f"production_eligible=False report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
