#!/usr/bin/env python3
"""Verify the materialized ProsocialDialog owner-review receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_prosocial_dialog_safety_curation_v1 import file_sha256
from lumi.posttraining_prosocial_dialog_safety_owner_review_v1 import (
    build_receipt,
    build_verification,
    load_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan, root)
    expected = build_receipt(
        plan, plan_file_sha256=file_sha256(args.plan), repository_root=root
    )
    actual = json.loads(args.receipt.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("ProsocialDialog owner-review receipt is not reproducible")
    verification = build_verification(
        actual, receipt_file_sha256=file_sha256(args.receipt)
    )
    output = root / plan["output"]["verification_path"]
    existing = json.loads(output.read_text(encoding="utf-8"))
    if existing != verification:
        raise ValueError("ProsocialDialog owner-review verification changed")
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
