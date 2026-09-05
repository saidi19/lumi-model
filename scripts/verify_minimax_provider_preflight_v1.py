#!/usr/bin/env python3
"""Independently verify the MiniMax provider preflight report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_minimax_provider_preflight_v1 import (
    build_minimax_provider_preflight_report_v1,
    build_minimax_provider_preflight_verification_v1,
    json_file_bytes,
    load_minimax_provider_preflight_plan_v1,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite MiniMax verification: {path}")
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
    plan = load_minimax_provider_preflight_plan_v1(
        plan_path, repository_root=root
    )
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rebuilt = build_minimax_provider_preflight_report_v1(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if rebuilt != report:
        raise ValueError("MiniMax provider-preflight report does not reproduce")
    verification = build_minimax_provider_preflight_verification_v1(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("MiniMax provider-preflight verification escaped evidence/")
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: MiniMax provider preflight independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
