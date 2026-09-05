#!/usr/bin/env python3
"""Validate OASST1 owner decisions and write a raw-text-free receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_oasst1_owner_review import (
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_decisions,
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
        raise FileExistsError(f"refusing to overwrite OASST1 owner receipt: {path}")
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
    decisions = args.decisions.resolve()
    receipt = args.receipt if args.receipt.is_absolute() else root / args.receipt
    if not receipt.resolve().is_relative_to((root / "evidence").resolve()):
        raise ValueError("OASST1 owner-review receipt must stay under evidence/")
    plan = load_owner_review_plan(plan_path, repository_root=root)
    sample = _read_jsonl(resolve_plan_binding(root, plan["bindings"]["owner_sample"]))
    result = validate_owner_decisions(
        _read_json(decisions),
        sample_rows=sample,
        decision_file_sha256=file_sha256(decisions),
        plan_file_sha256=file_sha256(plan_path),
    )
    _write_json_atomic(receipt, result)
    print(
        "PASS: OASST1 owner-review receipt "
        f"accepted={result['accepted_records']} rejected={result['rejected_records']} "
        f"production_eligible=False receipt={result['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

