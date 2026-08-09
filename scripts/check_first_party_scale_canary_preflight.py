#!/usr/bin/env python3
"""Build the no-call provider-bound Lumi first-party scale-canary preflight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_first_party_scale_canary_preflight import (
    build_first_party_scale_canary_preflight_report,
    json_file_bytes,
    load_first_party_scale_canary_preflight_plan,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scale-canary preflight: {path}")
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
    plan = load_first_party_scale_canary_preflight_plan(
        plan_path, repository_root=root
    )
    report = build_first_party_scale_canary_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("scale-canary preflight report escaped evidence/")
    _write_atomic(output, json_file_bytes(report))
    print(
        "PASS: local Qwen3 provider bound for the 1,000-row scale-canary preflight; "
        f"generated=0 api_calls=0 report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
