#!/usr/bin/env python3
"""Install and seal the exact DeepMind dependency overlay."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lumi.posttraining_deepmind_dependency_overlay import (
    build_overlay_receipt,
    file_sha256,
    install_overlay,
    json_file_bytes,
    load_overlay_plan,
    overlay_inventory,
    resolve_binding,
    run_import_smoke,
    validate_overlay_receipt,
    validate_wheelhouse,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite overlay receipt: {path}")
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
    plan = load_overlay_plan(plan_path, repository_root=root)
    wheelhouse = Path(plan["output"]["wheelhouse_path"]).resolve()
    overlay = Path(plan["output"]["overlay_path"]).resolve()
    lock_path = resolve_binding(
        root,
        plan["bindings"]["dependency_lock"],
        name="dependency_lock",
    )
    wheel_inventory = validate_wheelhouse(wheelhouse)
    command = install_overlay(
        lock_path=lock_path,
        overlay_path=overlay,
        python_executable=sys.executable,
        wheelhouse_path=wheelhouse,
    )
    inventory = overlay_inventory(overlay)
    smoke = run_import_smoke(
        overlay_path=overlay,
        python_executable=sys.executable,
    )
    receipt = build_overlay_receipt(
        command=command,
        inventory=inventory,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        smoke=smoke,
        wheel_inventory=wheel_inventory,
    )
    validate_overlay_receipt(receipt)
    output = (root / plan["output"]["receipt_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind overlay receipt must stay under evidence/")
    _write_bytes_atomic(output, json_file_bytes(receipt))
    print(
        "PASS: exact DeepMind dependency overlay installed; "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
