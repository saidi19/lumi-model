#!/usr/bin/env python3
"""Write the raw-text-free gpt-oss 3.2M closure-generation preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_closure_generation_preflight_v1 import (
    build_preflight_report,
    load_preflight_plan,
)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite closure preflight: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
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
    plan = load_preflight_plan(plan_path.resolve(), repository_root=root)
    report = build_preflight_report(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("closure preflight report escaped evidence/")
    _write_atomic(output, report)
    print(
        "PASS: 3.2M closure generation projected without remote action; "
        f"rows={report['projection']['expected_accepted_train_rows']} "
        f"calls={report['projection']['expected_inference_calls']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
