#!/usr/bin/env python3
"""Run pass one of the registry-v0.24 private SFT encoding preflight."""

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
    raise TimeoutError("private SFT encoding preflight pass-one runtime cap reached")


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
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
        early_report, early_verification = contract.output_paths(
            {"output": contract.EXPECTED_OUTPUT}, root
        )
        if os.path.lexists(early_report) or os.path.lexists(early_verification):
            raise FileExistsError("private SFT encoding preflight already started")
        marker_identity = contract.start_marker(early_report, contract.EXPECTED_REPORT_MARKER)
        marker_started = True
        plan = contract.load_plan(plan_path, repository_root=root)
        report_path, verification_path = contract.output_paths(plan, root)
        if report_path != early_report or verification_path != early_verification:
            raise ValueError("private SFT encoding preflight output changed")
        report = contract.build_report(
            plan=plan,
            plan_file_sha256=contract.file_sha256(plan_path),
            repository_root=root,
        )
        payload = contract.json_file_bytes(report)
        if (
            len(contract.EXPECTED_REPORT_MARKER) + len(payload)
            > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]
            or time.monotonic() - started
            > contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_pass"]
        ):
            raise ValueError("private SFT encoding preflight pass-one cap exceeded")
        contract.finish_marker(
            report_path,
            contract.EXPECTED_REPORT_MARKER,
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
                    "report_file_sha256": contract.sha256_bytes(payload),
                    "report_internal_sha256": report["report_sha256"],
                    "status": report["status"],
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
