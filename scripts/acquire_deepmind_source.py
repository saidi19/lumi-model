#!/usr/bin/env python3
"""Acquire the exact pinned DeepMind Mathematics source boundary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_deepmind_source_acquisition import (
    acquire_exact_source,
    build_acquisition_receipt,
    file_sha256,
    json_file_bytes,
    load_acquisition_plan,
    validate_acquisition_receipt,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acquisition receipt: {path}")
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
    plan = load_acquisition_plan(plan_path, repository_root=root)
    acquisition = acquire_exact_source(target_root=plan["output"]["source_root"])
    receipt = build_acquisition_receipt(
        acquisition=acquisition,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_acquisition_receipt(receipt)
    output = (root / plan["output"]["receipt_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind acquisition receipt must stay under evidence/")
    _write_bytes_atomic(output, json_file_bytes(receipt))
    print(
        "PASS: exact DeepMind source acquired; "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
