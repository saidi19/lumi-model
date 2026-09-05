#!/usr/bin/env python3
"""Independently replay candidate-mix v0.15 and verify the report once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v15 import (
    EXPECTED_LIMITS,
    build_verification,
    json_bytes,
    load_plan,
    rebuild_report,
    validate_report,
    validate_verification,
)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _temporary(path).exists():
        raise FileExistsError(f"refusing to overwrite v0.15 verification: {path}")
    with _temporary(path).open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    _temporary(path).replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan, root)
    report_path = root / plan["output"]["report_path"]
    verification_path = root / plan["output"]["verification_path"]
    if not report_path.is_file():
        raise FileNotFoundError("candidate-mix v0.15 report is missing")
    if verification_path.exists() or _temporary(verification_path).exists():
        raise FileExistsError("candidate-mix v0.15 verification already exists")
    if _temporary(report_path).exists():
        raise ValueError("candidate-mix v0.15 report temporary file exists")

    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    validate_report(report)
    plan_file_sha256 = file_sha256(args.plan)
    if report.get("plan_file_sha256") != plan_file_sha256:
        raise ValueError("candidate-mix v0.15 report is bound to a different plan")
    if json_bytes(report) != report_bytes:
        raise ValueError("candidate-mix v0.15 report is not canonical")
    prospective = build_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    validate_verification(prospective)
    prospective_bytes = json_bytes(prospective)
    if len(report_bytes) + len(prospective_bytes) > EXPECTED_LIMITS[
        "maximum_private_write_bytes"
    ]:
        raise ValueError("candidate-mix v0.15 evidence exceeds private-write cap")

    replay = rebuild_report(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        repository_root=root,
    )
    if replay != report:
        raise ValueError("candidate-mix v0.15 independent full pass did not reproduce")
    verification = build_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    if verification != prospective:
        raise ValueError("candidate-mix v0.15 prospective verification changed")
    _write_atomic(verification_path, json_bytes(verification))
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
