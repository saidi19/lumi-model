#!/usr/bin/env python3
"""Validate corrected OpenR1 decisions and seal a raw-text-free receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_openr1_owner_review import (
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
    validate_owner_review_receipt,
    validate_owner_sample,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 owner receipt: {path}")
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
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    decisions_path = args.decisions.resolve()
    plan = load_owner_review_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    sample = _read_jsonl(resolve_plan_binding(root, bindings["owner_sample"]))
    report = _read_json(resolve_plan_binding(root, bindings["visible_final_report"]))
    verification = _read_json(
        resolve_plan_binding(root, bindings["visible_final_verification"])
    )
    validate_owner_sample(sample, report=report, verification=verification)
    receipt = validate_owner_decisions(
        _read_json(decisions_path),
        sample_rows=sample,
        decision_file_sha256=file_sha256(decisions_path),
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_owner_review_receipt(receipt)

    repository_receipt = (root / plan["receipt_path"]).resolve()
    private_receipt = (
        root / plan["private_output_root"] / "owner-review-receipt.json"
    ).resolve()
    _write_json_atomic(private_receipt, receipt)
    _write_json_atomic(repository_receipt, receipt)
    print(
        "PASS: OpenR1 visible-final owner-review receipt "
        f"accepted={receipt['accepted_records']} "
        f"rejected={receipt['rejected_records']} production_eligible=False "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

