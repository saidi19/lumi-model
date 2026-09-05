#!/usr/bin/env python3
"""Materialize the exact authorized AI-assisted v0.3 registry successor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_ai_v3_source_promotion import (
    build_source_promotion_receipt,
    file_sha256,
    json_file_bytes,
    load_ai_v3_source_promotion_plan,
    reconstruct_authorized_successor,
    verify_source_promotion_receipt,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite AI v0.3 promotion output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
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
    predecessor, successor, preflight = reconstruct_authorized_successor(
        plan,
        repository_root=root,
    )
    receipt = build_source_promotion_receipt(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
        preflight_report=preflight,
    )
    verify_source_promotion_receipt(receipt)

    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    if not successor_path.is_relative_to(root / "configs"):
        raise ValueError("AI v0.3 successor registry must stay under configs/")
    if not receipt_path.is_relative_to(root / "evidence"):
        raise ValueError("AI v0.3 source-promotion receipt must stay under evidence/")
    if successor_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite AI v0.3 promotion outputs")
    _write_bytes_atomic(successor_path, json_file_bytes(successor))
    _write_bytes_atomic(receipt_path, json_file_bytes(receipt))
    print(
        "PASS: AI v0.3 registered in immutable candidate successor; "
        f"successor={file_sha256(successor_path)}; "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
