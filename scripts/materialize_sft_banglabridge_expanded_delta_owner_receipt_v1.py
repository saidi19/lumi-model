#!/usr/bin/env python3
"""Materialize only the BanglaBridge expanded-delta owner receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_banglabridge_expanded_delta_owner_review_v1 import (
    _write_json_atomic,
    build_receipt,
    load_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path, root)
    receipt = build_receipt(
        plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    receipt_path = root / plan["output"]["receipt_path"]
    verification_path = root / plan["output"]["verification_path"]
    if verification_path.exists():
        raise FileExistsError("owner verification already exists")
    _write_json_atomic(receipt_path, receipt)
    print(
        json.dumps(
            {
                "accepted_records": receipt["accepted_records"],
                "receipt_sha256": receipt["receipt_sha256"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
