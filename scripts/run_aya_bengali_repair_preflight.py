#!/usr/bin/env python3
"""Run the local-only Aya Bengali repair simulation over exact existing pages."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from lumi.posttraining_aya_repair import (
    build_aya_repair_report,
    build_aya_sft_records,
    finalize_aya_repair_candidates,
    load_aya_repair_plan,
    resolve_workspace_binding,
    select_aya_repair_candidates,
    validate_report_seal,
    verify_aya_input_receipts,
)
from lumi.posttraining_data import filter_sft_records
from lumi.posttraining_oasst1_full_run import (
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


def _write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya repair report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_exact_pages(inventory_path: Path, *, limits: dict[str, int]) -> list[dict[str, Any]]:
    inventory = _load_json(inventory_path)
    if (
        inventory.get("rows") != limits["expected_source_rows"]
        or inventory.get("response_bytes") != limits["expected_input_bytes"]
    ):
        raise ValueError("Aya repair input inventory boundary changed")
    pages = inventory.get("page_inventory")
    if not isinstance(pages, list) or len(pages) != limits["expected_page_objects"]:
        raise ValueError("Aya repair page inventory changed")
    raw_root = inventory_path.parent / "raw"
    wrapped_rows: list[dict[str, Any]] = []
    expected_offset = 0
    for number, item in enumerate(pages):
        if not isinstance(item, dict):
            raise ValueError("Aya repair page inventory item is invalid")
        filename = f"aya-bengali-page-{number:03d}.json"
        if item.get("filename") != filename or item.get("offset") != expected_offset:
            raise ValueError("Aya repair page sequence changed")
        page_path = raw_root / filename
        if file_sha256(page_path) != item.get("sha256"):
            raise ValueError(f"Aya repair page hash changed: {filename}")
        if page_path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"Aya repair page size changed: {filename}")
        payload = _load_json(page_path)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != item.get("rows"):
            raise ValueError(f"Aya repair page row count changed: {filename}")
        wrapped_rows.extend(rows)
        expected_offset += len(rows)
    if len(wrapped_rows) != limits["expected_source_rows"]:
        raise ValueError("Aya repair exact page set is incomplete")
    return wrapped_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_aya_repair_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    inventory_path = resolve_workspace_binding(root, bindings["input_inventory"])
    audit_path = resolve_workspace_binding(root, bindings["input_audit_report"])
    reconciliation_path = resolve_workspace_binding(root, bindings["reconciliation_receipt"])
    verify_aya_input_receipts(
        audit_report=_load_json(audit_path),
        reconciliation=_load_json(reconciliation_path),
    )
    wrapped_rows = _load_exact_pages(inventory_path, limits=plan["limits"])
    selection = select_aya_repair_candidates(wrapped_rows)
    records, annotators = build_aya_sft_records(selection.candidates)

    policy, source_registry, benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    structurally_accepted, filter_report = filter_sft_records(
        records,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=source_registry,
    )
    four_k_accepted, initial_token_metrics = measure_token_fit(
        structurally_accepted,
        template=template,
        tokenizer=tokenizer,
    )
    final_records, concentration_metrics = finalize_aya_repair_candidates(
        four_k_accepted,
        annotators=annotators,
    )
    final_records, token_metrics = measure_token_fit(
        final_records,
        template=template,
        tokenizer=tokenizer,
    )
    token_metrics["accepted_before_final_concentration"] = initial_token_metrics["accepted_records"]
    token_metrics["overlength_rejected_before_final_concentration"] = initial_token_metrics[
        "overlength_rejected"
    ]

    report = build_aya_repair_report(
        bindings={name: binding["sha256"] for name, binding in bindings.items()},
        concentration_metrics=concentration_metrics,
        filter_report=filter_report,
        final_records=final_records,
        owner_sample_records=plan["limits"]["owner_sample_records"],
        plan_file_sha256=file_sha256(plan_path),
        selection_metrics=selection.metrics,
        token_metrics=token_metrics,
        annotators=annotators,
    )
    elapsed = time.monotonic() - started
    if elapsed > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("Aya repair preflight exceeded its local CPU runtime bound")
    report["runtime_seconds"] = round(elapsed, 3)
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    validate_report_seal(report)
    output_path = (root / plan["output"]["report_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("Aya repair report escaped the evidence directory")
    _write_json_atomic(output_path, report)
    print(
        "PASS: Aya Bengali local repair preflight "
        f"input={len(wrapped_rows)} "
        f"repaired={selection.metrics['terminal_pipe_artifacts_repaired']} "
        f"candidate={len(final_records)} supervised_tokens={token_metrics['supervised_tokens']} "
        f"materialized=False production_eligible=False report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
