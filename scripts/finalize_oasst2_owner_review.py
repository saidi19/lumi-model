#!/usr/bin/env python3
"""Finalize OASST2 owner review without promoting the candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_oasst2_curation import (
    file_sha256,
    write_json_atomic,
)
from lumi.posttraining_oasst2_owner_review import (
    load_owner_review_plan,
    read_json,
    resolve_binding,
    validate_bound_candidate,
    validate_owner_decisions,
    validate_owner_review_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_owner_review_plan(plan_path, repository_root=root)
    sample, _report, _verification = validate_bound_candidate(plan, repository_root=root)
    decisions_path = resolve_binding(root, plan["bindings"]["decisions"], name="decisions")
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
        raise ValueError("OASST2 repository receipt must stay under evidence/")
    if not private_receipt.is_relative_to(root.parent.resolve()):
        raise ValueError("OASST2 private receipt must stay in workspace")
    if repository_receipt.exists() or private_receipt.exists():
        raise FileExistsError("refusing to overwrite OASST2 owner receipt")
    write_json_atomic(private_receipt, receipt)
    write_json_atomic(repository_receipt, receipt)
    print(
        "PASS: OASST2 owner-review receipt "
        f"accepted={receipt['accepted_records']} "
        f"rejected={receipt['rejected_records']} "
        f"production_eligible=False receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
