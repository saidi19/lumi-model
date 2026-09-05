#!/usr/bin/env python3
"""Build the non-mutating registry-v0.13 candidate hygiene projection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_hygiene_preflight_v2 import (
    json_file_bytes,
    load_hygiene_preflight_v2_plan,
    rebuild_hygiene_preflight_v2,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite hygiene v0.2 report: {path}")
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
    plan = load_hygiene_preflight_v2_plan(plan_path, repository_root=root)
    report = rebuild_hygiene_preflight_v2(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("candidate hygiene v0.2 report escaped evidence/")
    _write_atomic(output, json_file_bytes(report))
    pool = report["projected_pool"]
    print(
        "PASS: registry-v0.13 hygiene projection built; "
        f"records={pool['records']} removals={pool['removed_records']} "
        f"train_tokens={pool['train_supervised_tokens']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
