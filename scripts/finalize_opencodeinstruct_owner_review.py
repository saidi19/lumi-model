#!/usr/bin/env python3
"""Finalize OpenCodeInstruct owner review without promoting the candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_opencodeinstruct_curation import file_sha256
from lumi.posttraining_opencodeinstruct_owner_review import (
    load_owner_review_plan,
    read_json,
    resolve_binding,
    validate_bound_candidate,
    validate_owner_decisions,
    validate_owner_review_receipt,
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenCodeInstruct receipt: {path}")
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

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_owner_review_plan(plan_path, repository_root=root)
    sample, _report, _verification = validate_bound_candidate(
        plan, repository_root=root
    )
    decisions_path = resolve_binding(
        root, plan["bindings"]["decisions"], name="decisions"
    )
    receipt = validate_owner_decisions(
        read_json(decisions_path),
        sample_rows=sample,
        plan=plan,
        decision_file_sha256=file_sha256(decisions_path),
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_owner_review_receipt(receipt)

    repository_receipt = (root / plan["output"]["repository_receipt_path"]).resolve()
    private_receipt = (root / plan["output"]["private_receipt_path"]).resolve()
    if not repository_receipt.is_relative_to((root / "evidence").resolve()):
        raise ValueError("OpenCodeInstruct repository receipt must stay under evidence/")
    if not private_receipt.is_relative_to(root.parent.resolve()):
        raise ValueError("OpenCodeInstruct private receipt must stay in workspace")
    if repository_receipt.exists() or private_receipt.exists():
        raise FileExistsError("refusing to overwrite OpenCodeInstruct owner receipt")
    _write_json_atomic(private_receipt, receipt)
    _write_json_atomic(repository_receipt, receipt)
    print(
        "PASS: OpenCodeInstruct owner-review receipt "
        f"accepted={receipt['accepted_records']} rejected={receipt['rejected_records']} "
        f"production_eligible=False receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
