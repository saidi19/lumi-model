#!/usr/bin/env python3
"""Materialize the bounded raw-text-free Be-DSTC2 rights STOP report."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from lumi.posttraining_bedstc2_upstream_rights_review_v1 import (
    EXPECTED_LIMITS,
    EXPECTED_REPORT_MARKER,
    build_report,
    build_verification,
    file_sha256,
    finish_marker,
    json_bytes,
    load_plan,
    preflight_runner,
    prospective_private_write_bytes,
    sha256_bytes,
    start_marker,
)


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("upstream-rights runner runtime cap reached")


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
        signal.alarm(EXPECTED_LIMITS["runtime_seconds_per_process"])
    try:
        plan = load_plan(plan_path, root)
        report_path, _ = preflight_runner(plan, root)
        identity = start_marker(report_path, EXPECTED_REPORT_MARKER)
        marker_started = True
        plan_hash = file_sha256(plan_path)
        report = build_report(repository_root=root, plan_file_sha256=plan_hash)
        report_payload = json_bytes(report)
        prospective = build_verification(
            report,
            report_file_sha256=sha256_bytes(report_payload),
            report_file_bytes=len(report_payload),
            plan_file_sha256=plan_hash,
        )
        verification_payload = json_bytes(prospective)
        if (
            prospective_private_write_bytes(
                plan=plan,
                root=root,
                report_bytes=report_payload,
                verification_bytes=verification_payload,
            )
            > EXPECTED_LIMITS["private_write_bytes"]
        ):
            raise ValueError("upstream-rights private-write cap exceeded")
        if time.monotonic() - started > EXPECTED_LIMITS["runtime_seconds_per_process"]:
            raise TimeoutError("upstream-rights runner runtime cap reached")
        finish_marker(report_path, EXPECTED_REPORT_MARKER, report_payload, identity)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "production_eligible": False,
                    "report_file_sha256": sha256_bytes(report_payload),
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
                    "status": "FAIL_CLOSED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
