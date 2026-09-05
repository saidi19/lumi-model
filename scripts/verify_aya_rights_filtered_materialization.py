#!/usr/bin/env python3
"""Independently replay the approved Aya rights-filtered materialization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_aya_rights_filtered_materialization import (
    canonical_json_bytes,
    load_plan,
    verify_materialized_outputs,
)
from lumi.posttraining_aya_rights_filtered_preflight import file_sha256


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite Aya materialization verification")
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
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path, repository_root=root)
    verification = verify_materialized_outputs(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("Aya materialization verification escaped evidence")
    _write_once(output, canonical_json_bytes(verification))
    print(
        "PASS: independently verified Aya rights-filtered materialization; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
