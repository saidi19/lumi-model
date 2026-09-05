#!/usr/bin/env python3
"""Run the authorized CPU-only local Aya capacity scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_aya_capacity_scan import (
    build_report,
    file_sha256,
    load_plan,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path.resolve(), repository_root=root)

    def progress(rows: int) -> None:
        print(f"PROGRESS: Aya rows scanned {rows}/1200429", flush=True)

    report = build_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
        progress=progress,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("Aya capacity report escaped evidence/")
    if output.exists():
        raise FileExistsError("refusing to overwrite Aya capacity report")
    write_json_atomic(output, report)
    if output.stat().st_size > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("Aya capacity report exceeded private-write ceiling")
    decision = report["capacity_decision"]
    print(
        "PASS: Aya capacity-only scan complete "
        f"wiki_lower_bound={decision['conditional_wikisplit_train_supervised_tokens_lower_bound']} "
        f"joke_rights_clear={decision['immediately_rights_clear_joke_train_supervised_tokens']} "
        f"report={report['report_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
