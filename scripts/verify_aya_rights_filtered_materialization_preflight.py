#!/usr/bin/env python3
"""Independently verify the non-mutating Aya filtered projection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_aya_rights_filtered_preflight import (
    build_report,
    build_verification,
    canonical_json_bytes,
    file_sha256,
    load_plan,
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya filtered verification: {path}")
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
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = build_report(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if report != expected or report_path.read_bytes() != canonical_json_bytes(expected):
        raise ValueError("Aya filtered-materialization preflight does not reproduce")
    verification = build_verification(report, report_file_sha256=file_sha256(report_path))
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("Aya filtered verification escaped evidence/")
    _write_once(output, canonical_json_bytes(verification))
    print(
        "PASS: Aya filtered-materialization preflight independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
