#!/usr/bin/env python3
"""Independently run pass two of the registry-v0.24 candidate-mix audit."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import lumi.posttraining_candidate_mix_reaudit_v18 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("candidate-mix v0.18 verifier runtime cap reached")


def _read_report_once(path: Path) -> bytes:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= contract.EXPECTED_LIMITS["maximum_private_write_bytes"]:
            raise ValueError("candidate-mix v0.18 report metadata changed")
        payload = b""
        while len(payload) < info.st_size:
            chunk = os.read(descriptor, min(65_536, info.st_size - len(payload)))
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("candidate-mix v0.18 report changed during read")
    return payload


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    report_arg = args.report if args.report.is_absolute() else root / args.report
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"])
    try:
        plan = contract.load_plan(plan_path, root)
        report_path, verification_path = contract.preflight_verifier(plan, root)
        if report_arg.resolve() != report_path:
            raise ValueError("candidate-mix v0.18 report argument changed")
        marker_identity = contract.start_marker(
            verification_path, contract.EXPECTED_VERIFICATION_MARKER
        )
        marker_started = True
        report_payload = _read_report_once(report_path)
        actual = contract.strict_json_bytes(report_payload)
        if not isinstance(actual, Mapping) or contract.json_bytes(actual) != report_payload:
            raise ValueError("candidate-mix v0.18 report encoding changed")
        contract.validate_report(actual)
        replay = contract.rebuild_report(
            plan=plan,
            plan_file_sha256=contract.file_sha256(plan_path),
            repository_root=root,
            started=started,
        )
        if replay != actual:
            raise ValueError("candidate-mix v0.18 independent pass did not reproduce")
        verification = contract.build_verification(
            plan=plan,
            report=actual,
            report_file_sha256=contract.sha256_bytes(report_payload),
        )
        payload = contract.json_bytes(verification)
        if contract.prospective_private_write_bytes(
            plan=plan, root=root, report_bytes=report_payload,
            verification_bytes=payload,
        ) > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]:
            raise ValueError("candidate-mix v0.18 private-write cap exceeded")
        contract.finish_marker(
            verification_path, contract.EXPECTED_VERIFICATION_MARKER,
            payload, marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(json.dumps({
            "candidate_row_reads": verification["resource_accounting"]["candidate_row_reads"],
            "production_eligible": False,
            "status": verification["status"],
            "verification_file_sha256": contract.sha256_bytes(payload),
            "verification_internal_sha256": verification["verification_sha256"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(json.dumps({
            "exception_type": type(error).__name__,
            "marker_started": marker_started,
            "status": "FAIL_CLOSED",
        }, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
