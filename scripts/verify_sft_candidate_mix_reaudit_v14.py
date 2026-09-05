#!/usr/bin/env python3
"""Independently replay candidate-mix v0.14 and verify the report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v14 import (
    build_verification,
    json_bytes,
    load_plan,
    rebuild_report,
)


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite candidate-mix v0.14 verification: {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan, root)
    report_path = root / plan["output"]["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    replay = rebuild_report(
        plan=plan, plan_file_sha256=file_sha256(args.plan), repository_root=root
    )
    if replay != report:
        raise ValueError("candidate-mix v0.14 full pass did not reproduce")
    verification = build_verification(
        plan=plan, report=report, report_file_sha256=file_sha256(report_path)
    )
    output = root / plan["output"]["verification_path"]
    _write_atomic(output, json_bytes(verification))
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
