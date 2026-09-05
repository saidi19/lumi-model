#!/usr/bin/env python3
"""Verify the six-row Aya MIT owner-review receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_owner_review_v1 import (
    build_receipt,
    build_verification,
    canonical_json_bytes,
    file_sha256,
    load_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    receipt_path = Path(args.receipt).resolve()
    plan = load_plan(plan_path, root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = build_receipt(plan, file_sha256(plan_path), root)
    if receipt != expected:
        raise ValueError("owner-review receipt is not reproducible")
    verification = build_verification(receipt, file_sha256(receipt_path))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(verification))


if __name__ == "__main__":
    main()

