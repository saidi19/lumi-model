#!/usr/bin/env python3
"""Independently verify the ProsocialDialog registry-v0.19 promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_prosocial_dialog_safety_curation_v1 import file_sha256
from lumi.posttraining_prosocial_dialog_source_promotion_v19 import (
    build_reaudit_draft,
    build_receipt,
    build_verification,
    json_file_bytes,
    load_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan, root)
    receipt_path = root / plan["output"]["receipt_path"]
    verification_path = root / plan["output"]["verification_path"]
    draft_path = root / plan["output"]["reaudit_draft_path"]
    actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    actual_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    actual_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    expected_receipt, expected_registry = build_receipt(
        plan, plan_file_sha256=file_sha256(args.plan), repository_root=root
    )
    if actual_receipt != expected_receipt:
        raise ValueError("ProsocialDialog promotion receipt is not reproducible")
    registry_path = root / plan["output"]["successor_registry_path"]
    if registry_path.read_bytes() != json_file_bytes(expected_registry):
        raise ValueError("ProsocialDialog registry v0.19 is not reproducible")
    expected_verification = build_verification(
        plan=plan,
        receipt=actual_receipt,
        receipt_file_sha256=file_sha256(receipt_path),
        repository_root=root,
    )
    if actual_verification != expected_verification:
        raise ValueError("ProsocialDialog promotion verification changed")
    expected_draft = build_reaudit_draft(
        plan=plan,
        receipt_file_sha256=file_sha256(receipt_path),
        verification_file_sha256=file_sha256(verification_path),
    )
    if actual_draft != expected_draft:
        raise ValueError("ProsocialDialog re-audit draft changed")
    print(json.dumps(actual_verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
