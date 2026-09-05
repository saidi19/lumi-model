#!/usr/bin/env python3
"""Materialize the bounded MASSIVE bn-BD parser owner-review receipt."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from lumi.posttraining_massive_bn_bd_parser_owner_review_v4 import (
    EXPECTED_LIMITS,
    EXPECTED_RECEIPT_MARKER,
    build_receipt,
    build_verification,
    file_sha256,
    finish_marker,
    json_bytes,
    load_plan,
    preflight_finalizer,
    prospective_private_write_bytes,
    sha256_bytes,
    start_marker,
    validate_receipt,
    validate_verification,
)


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("owner-review finalizer runtime cap reached")


def _validate_execution_limits() -> None:
    if (
        EXPECTED_LIMITS.get("passes") != 2
        or EXPECTED_LIMITS.get("runtime_seconds_per_process") != 150
        or EXPECTED_LIMITS.get("network_bytes") != 0
        or EXPECTED_LIMITS.get("gpu_use") != 0
        or EXPECTED_LIMITS.get("inference_calls") != 0
    ):
        raise ValueError("owner-review execution limits changed")


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
        _validate_execution_limits()
        plan = load_plan(plan_path, root)
        receipt_path, _ = preflight_finalizer(plan, root)
        marker_identity = start_marker(receipt_path, EXPECTED_RECEIPT_MARKER)
        marker_started = True
        plan_file_sha256 = file_sha256(plan_path)
        receipt = build_receipt(
            plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
        )
        validate_receipt(receipt)
        receipt_payload = json_bytes(receipt)
        prospective_verification = build_verification(
            receipt,
            receipt_file_sha256=sha256_bytes(receipt_payload),
            receipt_file_bytes=len(receipt_payload),
            plan_file_sha256=plan_file_sha256,
        )
        validate_verification(prospective_verification)
        verification_payload = json_bytes(prospective_verification)
        physical_writes = prospective_private_write_bytes(
            plan=plan,
            root=root,
            receipt_bytes=receipt_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > EXPECTED_LIMITS["private_write_bytes"]:
            raise ValueError("owner-review private-write cap exceeded")
        if time.monotonic() - started > EXPECTED_LIMITS["runtime_seconds_per_process"]:
            raise TimeoutError("owner-review finalizer runtime cap reached")
        finish_marker(
            receipt_path,
            EXPECTED_RECEIPT_MARKER,
            receipt_payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "accepted_records": receipt["owner_review"][
                        "accepted_records"
                    ],
                    "production_eligible": False,
                    "receipt_file_sha256": sha256_bytes(receipt_payload),
                    "receipt_internal_sha256": receipt["receipt_sha256"],
                    "status": receipt["status"],
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
