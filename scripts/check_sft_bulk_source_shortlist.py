#!/usr/bin/env python3
"""Create the raw-text-free Lumi bulk-source shortlist report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_bulk_source_shortlist import (
    file_sha256,
    json_file_bytes,
    load_bulk_source_shortlist_plan,
    rebuild_bulk_source_shortlist,
    validate_bulk_source_shortlist_report,
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shortlist report: {path}")
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
    plan = load_bulk_source_shortlist_plan(plan_path, repository_root=root)
    report = rebuild_bulk_source_shortlist(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    validate_bulk_source_shortlist_report(report)
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("shortlist report must remain under evidence/")
    _write_once(output, json_file_bytes(report))
    print(
        "PASS: bulk-source shortlist sealed; "
        f"math_next={report['next_gate']['source_id']} "
        f"coding_second=nvidia_opencodeinstruct report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

