#!/usr/bin/env python3
"""Diagnose OpenR1 PII-pattern rejections without retaining raw text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_openr1 import select_verified_openr1_records
from lumi.posttraining_openr1_pii import (
    build_openr1_pii_report,
    diagnose_openr1_pii_patterns,
    file_sha256,
    load_openr1_pii_plan,
    sha256_value,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, binding: dict[str, Any]) -> Path:
    return (root / str(binding["path"])).resolve()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 PII evidence: {output}")

    plan = load_openr1_pii_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    evidence_path = _resolve(root, bindings["verified_trace_audit_evidence"])
    evidence = _load_json(evidence_path)
    evidence_without_seal = {
        key: value for key, value in evidence.items() if key != "report_sha256"
    }
    if sha256_value(evidence_without_seal) != evidence.get("report_sha256"):
        raise ValueError("OpenR1 verified-trace evidence seal changed")
    if evidence.get("selection_metrics", {}).get("selected_records") != plan["limits"][
        "expected_selected_records"
    ]:
        raise ValueError("OpenR1 selected-record boundary changed")
    if evidence.get("filter_metrics", {}).get("rejection_counts") != {
        "pii_pattern": plan["limits"]["expected_source_filter_pii_rejections"]
    }:
        raise ValueError("OpenR1 PII rejection boundary changed")

    raw_payload = _load_json(_resolve(root, bindings["raw_rows"]))
    wrapped_rows = raw_payload.get("rows")
    if not isinstance(wrapped_rows, list) or len(wrapped_rows) > plan["limits"][
        "maximum_raw_rows"
    ]:
        raise ValueError("OpenR1 raw row boundary changed")
    records, selection_metrics = select_verified_openr1_records(
        wrapped_rows,
        revision=plan["source"]["revision"],
        maximum_selected_records=plan["limits"]["expected_selected_records"],
    )
    if selection_metrics["selected_records"] != plan["limits"][
        "expected_selected_records"
    ]:
        raise ValueError("OpenR1 verified selection no longer reproduces")

    metrics = diagnose_openr1_pii_patterns(records)
    if metrics["detector_record_counts"].get("phone") != plan["limits"][
        "expected_source_filter_pii_rejections"
    ]:
        raise ValueError("OpenR1 phone-pattern diagnostic does not reconcile")
    report = build_openr1_pii_report(
        diagnostic_metrics=metrics,
        observed_at=plan["observed_at"],
        plan_sha256=file_sha256(plan_path),
        raw_rows_sha256=bindings["raw_rows"]["sha256"],
        source_filter_sha256=bindings["source_filter_implementation"]["sha256"],
        verified_trace_evidence_file_sha256=bindings[
            "verified_trace_audit_evidence"
        ]["sha256"],
        verified_trace_evidence_internal_sha256=evidence["report_sha256"],
    )
    _write_json_atomic(output, report)
    print(
        "PASS: OpenR1 raw-text-free PII diagnostic completed; "
        f"likely_math_numeric={metrics['likely_math_numeric_false_positive_records']} "
        f"unresolved={metrics['unresolved_phone_like_records']} "
        f"filter_modified={report['global_pii_filter_modified']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
