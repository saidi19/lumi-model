#!/usr/bin/env python3
"""Independently reconstruct the BanglaBridge expanded-delta owner receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_banglabridge_expanded_delta_owner_review_v1 import (
    _write_json_atomic,
    build_receipt,
    build_verification,
    load_plan,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    plan = load_plan(plan_path, root)
    expected = build_receipt(
        plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    actual = json.loads(receipt_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("BanglaBridge owner-review receipt is not reproducible")
    verification = build_verification(
        actual, receipt_file_sha256=file_sha256(receipt_path)
    )
    verification_path = root / plan["output"]["verification_path"]
    _write_json_atomic(verification_path, verification)
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
