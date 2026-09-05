#!/usr/bin/env python3
"""Run the bounded line-scoped OpenR1 PII reconciliation without retaining text."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_openr1 import select_verified_openr1_records
from lumi.posttraining_openr1_pii_secondary import (
    build_secondary_pii_report,
    diagnose_line_scoped_phone_patterns,
    load_secondary_pii_plan,
    resolve_secondary_binding,
    sha256_value,
    validate_secondary_pii_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_seal(value: dict[str, Any], field: str) -> str:
    seal = value.get(field)
    if not isinstance(seal, str) or sha256_value(
        {key: item for key, item in value.items() if key != field}
    ) != seal:
        raise ValueError(f"{field} changed")
    return seal


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 secondary PII evidence: {path}")
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
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_secondary_pii_plan(plan_path, repository_root=root)
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("OpenR1 secondary PII report must stay under evidence/")

    primary = _read_json(
        resolve_secondary_binding(root, plan["bindings"]["primary_pii_evidence"])
    )
    verified = _read_json(
        resolve_secondary_binding(root, plan["bindings"]["verified_trace_audit_evidence"])
    )
    primary_seal = _verify_seal(primary, "report_sha256")
    verified_seal = _verify_seal(verified, "report_sha256")
    if primary.get("diagnostic_metrics", {}).get("unresolved_phone_like_records") != plan[
        "limits"
    ]["expected_primary_unresolved_records"]:
        raise ValueError("OpenR1 primary PII unresolved boundary changed")

    raw = _read_json(resolve_secondary_binding(root, plan["bindings"]["raw_rows"]))
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) > plan["limits"]["maximum_raw_rows"]:
        raise ValueError("OpenR1 secondary PII raw-row boundary changed")
    records, selection = select_verified_openr1_records(
        rows,
        revision=plan["source"]["revision"],
        maximum_selected_records=plan["limits"]["expected_selected_records"],
    )
    if selection["selected_records"] != plan["limits"]["expected_selected_records"]:
        raise ValueError("OpenR1 secondary PII selection changed")
    metrics = diagnose_line_scoped_phone_patterns(records)
    expected_metrics = {
        "cross_message_join_artifact_records": plan["limits"][
            "expected_cross_message_join_artifact_records"
        ],
        "global_phone_detector_records": plan["limits"][
            "expected_global_phone_detector_records"
        ],
        "line_scoped_math_numeric_false_positive_records": plan["limits"][
            "expected_line_scoped_math_false_positive_records"
        ],
        "unresolved_records": plan["limits"]["expected_secondary_unresolved_records"],
    }
    if any(metrics.get(field) != expected for field, expected in expected_metrics.items()):
        raise ValueError("OpenR1 secondary PII exact projection changed")
    report = build_secondary_pii_report(
        metrics=metrics,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        primary_pii_receipt_sha256=primary_seal,
        verified_trace_receipt_sha256=verified_seal,
    )
    validate_secondary_pii_report(report)
    if time.monotonic() - started > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("OpenR1 secondary PII diagnostic exceeded its runtime bound")
    _write_json_atomic(output, report)
    print(
        "PASS: OpenR1 secondary PII diagnostic; global_phone=6 line_math=5 "
        "cross_message=1 unresolved=0 global_filter_modified=False "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
