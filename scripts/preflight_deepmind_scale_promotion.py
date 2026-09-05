#!/usr/bin/env python3
"""Build the raw-text-free DeepMind scale promotion preflight report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_deepmind_scale_generation import file_sha256
from lumi.posttraining_deepmind_scale_promotion_preflight import (
    build_preflight_report,
    json_file_bytes,
    load_preflight_plan,
    validate_preflight_report,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind scale preflight: {path}")
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
    plan = load_preflight_plan(plan_path, repository_root=root)
    report = build_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    validate_preflight_report(report)
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind scale preflight report escaped evidence/")
    _write_atomic(output, json_file_bytes(report))
    print(
        "PASS: DeepMind scale promotion preflight verified without promotion; "
        f"successor={report['proposed_successor']['file_sha256']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
