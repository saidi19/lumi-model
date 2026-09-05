#!/usr/bin/env python3
"""Run pass two of the registry-v0.24 private SFT encoding preflight."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

import lumi.posttraining_private_sft_encoding_preflight_v1 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("private SFT encoding preflight pass-two runtime cap reached")


def _read_report_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or not 1 <= info.st_size <= contract.EXPECTED_LIMITS["maximum_private_write_bytes"]
        ):
            raise ValueError("private SFT encoding preflight report metadata changed")
        payload = bytearray()
        while len(payload) < info.st_size:
            chunk = os.read(descriptor, min(65_536, info.st_size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("private SFT encoding preflight report changed during read")
    return bytes(payload)


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
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"])
    try:
        plan_path = Path(os.path.abspath(plan_path))
        expected_plan = Path(os.path.abspath(root / contract.EXPECTED_PLAN_PATH))
        if plan_path != expected_plan:
            raise ValueError("private SFT encoding preflight plan path changed")
        plan_info = plan_path.lstat()
        if stat.S_ISLNK(plan_info.st_mode) or not stat.S_ISREG(plan_info.st_mode):
            raise ValueError("private SFT encoding preflight plan is unsafe")
        expected_report, verification_path = contract.output_paths(
            {"output": contract.EXPECTED_OUTPUT}, root
        )
        report_arg = Path(os.path.abspath(report_arg))
        if report_arg != expected_report:
            raise ValueError("private SFT encoding preflight report path changed")
        report_info = expected_report.lstat()
        if stat.S_ISLNK(report_info.st_mode) or not stat.S_ISREG(report_info.st_mode):
            raise ValueError("private SFT encoding preflight report is unsafe")
        if os.path.lexists(verification_path):
            raise FileExistsError("private SFT encoding verification already started")
        marker_identity = contract.start_marker(
            verification_path, contract.EXPECTED_VERIFICATION_MARKER
        )
        marker_started = True
        plan = contract.load_plan(plan_path, repository_root=root)
        report_path, bound_verification = contract.output_paths(plan, root)
        if report_path != expected_report or bound_verification != verification_path:
            raise ValueError("private SFT encoding verification output changed")
        report_payload = _read_report_once(report_path)
        report = contract.strict_json_bytes(report_payload)
        if (
            not isinstance(report, dict)
            or contract.json_file_bytes(report) != report_payload
            or report.get("plan_file_sha256") != contract.file_sha256(plan_path)
        ):
            raise ValueError("private SFT encoding preflight report encoding changed")
        contract.validate_report(report)
        replay = contract.build_report(
            plan=plan,
            plan_file_sha256=contract.file_sha256(plan_path),
            repository_root=root,
        )
        if replay != report:
            raise ValueError("private SFT encoding preflight replay changed")
        verification = contract.build_verification(
            report=report,
            report_file_sha256=contract.sha256_bytes(report_payload),
        )
        payload = contract.json_file_bytes(verification)
        if (
            len(contract.EXPECTED_REPORT_MARKER)
            + len(report_payload)
            + len(contract.EXPECTED_VERIFICATION_MARKER)
            + len(payload)
            > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]
            or time.monotonic() - started
            > contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"]
        ):
            raise ValueError("private SFT encoding preflight total cap exceeded")
        contract.finish_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
            payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "encoded_shards_created": False,
                    "private_jsonl_row_reads": 0,
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": contract.sha256_bytes(payload),
                    "verification_internal_sha256": verification["verification_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "exception_type": type(error).__name__,
                    "marker_started": marker_started,
                    "status": "STOP",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
