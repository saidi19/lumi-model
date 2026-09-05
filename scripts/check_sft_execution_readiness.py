#!/usr/bin/env python3
"""Write a raw-text-free report of every blocker before Lumi SFT execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_sft_plan import (
    build_sft_execution_readiness_report,
    load_sft_execution_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite SFT readiness report: {args.output}")

    plan = load_sft_execution_plan(args.plan, repository_root=args.repository_root)
    report = build_sft_execution_readiness_report(
        plan,
        repository_root=args.repository_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(
        f"BLOCKED: ready_for_gpu={report['ready_for_gpu']} "
        f"blockers={len(report['blockers'])} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
