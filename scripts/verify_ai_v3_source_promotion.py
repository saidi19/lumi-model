#!/usr/bin/env python3
"""Independently replay the AI-assisted v0.3 registry materialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_ai_v3_source_promotion import (
    build_source_promotion_receipt,
    build_source_promotion_verification,
    file_sha256,
    load_ai_v3_source_promotion_plan,
    read_object,
    reconstruct_authorized_successor,
    verify_source_promotion_receipt,
    verify_source_promotion_verification,
)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite AI v0.3 verification: {path}")
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
    plan = load_ai_v3_source_promotion_plan(plan_path, repository_root=root)
    predecessor, expected_successor, preflight = reconstruct_authorized_successor(
        plan,
        repository_root=root,
    )
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    actual_successor = read_object(successor_path)
    if actual_successor != expected_successor:
        raise ValueError("AI v0.3 successor registry reconstruction changed")

    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    receipt = read_object(receipt_path)
    verify_source_promotion_receipt(receipt)
    expected_receipt = build_source_promotion_receipt(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=expected_successor,
        preflight_report=preflight,
    )
    if receipt != expected_receipt:
        raise ValueError("AI v0.3 source-promotion receipt reconstruction changed")

    verification = build_source_promotion_verification(
        plan=plan,
        receipt_file_sha256=file_sha256(receipt_path),
        receipt_internal_sha256=receipt["receipt_sha256"],
    )
    verify_source_promotion_verification(verification)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("AI v0.3 verification must stay under evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: independently verified AI v0.3 candidate registry promotion; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
