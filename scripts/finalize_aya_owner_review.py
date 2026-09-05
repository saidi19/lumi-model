#!/usr/bin/env python3
"""Validate Aya Bengali owner decisions and write a raw-text-free receipt."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.posttraining_aya_owner_review import (
    load_exact_owner_sample,
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
    validate_owner_review_receipt,
)
from lumi.posttraining_first_party import file_sha256


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya owner receipt: {path}")
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
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    decisions_path = args.decisions.resolve()
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    if not receipt_path.resolve().is_relative_to((root / "evidence").resolve()):
        raise ValueError("Aya owner-review receipt must stay under evidence/")
    plan = load_owner_review_plan(plan_path, repository_root=root)
    materialization_report = _read_json(
        resolve_plan_binding(root, plan["bindings"]["materialization_report"])
    )
    sample = load_exact_owner_sample(
        resolve_plan_binding(root, plan["bindings"]["accepted_records"]),
        materialization_report=materialization_report,
    )
    reviewed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    receipt = validate_owner_decisions(
        _read_json(decisions_path),
        sample_rows=sample,
        decision_file_sha256=file_sha256(decisions_path),
        plan_file_sha256=file_sha256(plan_path),
        reviewed_at=reviewed_at,
    )
    validate_owner_review_receipt(receipt)
    _write_json_atomic(receipt_path, receipt)
    print(
        "PASS: Aya owner-review receipt "
        f"accepted={receipt['accepted_records']} rejected={receipt['rejected_records']} "
        f"production_eligible=False receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
