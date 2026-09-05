#!/usr/bin/env python3
"""Run the bounded non-mutating Lumi SFT registry-v0.14 projection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_hygiene_registry_preflight_v2 import (
    json_file_bytes,
    load_hygiene_registry_v14_preflight_plan,
    rebuild_hygiene_registry_v14_preflight,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registry-v0.14 preflight report: {path}")
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
    plan = load_hygiene_registry_v14_preflight_plan(plan_path, repository_root=root)
    report = rebuild_hygiene_registry_v14_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("registry-v0.14 preflight report escaped evidence/")
    _write_atomic(output, json_file_bytes(report))
    successor = report["projected_successor"]
    print(
        "PASS: registry-v0.14 two-source projection built without mutation; "
        f"sources={successor['source_count']} bytes={successor['bytes']} "
        f"sha256={successor['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
