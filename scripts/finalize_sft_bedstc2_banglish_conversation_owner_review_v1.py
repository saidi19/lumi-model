#!/usr/bin/env python3
"""Materialize the bounded raw-text-free Be-DSTC2 owner-review receipt."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from lumi.posttraining_bedstc2_banglish_conversation_owner_review_v1 import (
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
)


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("owner-review finalizer runtime cap reached")


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
        receipt_path, _ = preflight_finalizer(plan, root)
        marker_identity = start_marker(receipt_path, EXPECTED_RECEIPT_MARKER)
        marker_started = True
        receipt = build_receipt(
            plan,
            plan_file_sha256=file_sha256(plan_path),
            repository_root=root,
        )
        receipt_payload = json_bytes(receipt)
        prospective = build_verification(
            receipt,
            receipt_file_sha256=sha256_bytes(receipt_payload),
            receipt_file_bytes=len(receipt_payload),
            plan_file_sha256=file_sha256(plan_path),
        )
        verification_payload = json_bytes(prospective)
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
                    "accepted_records": receipt["owner_review"]["accepted_records"],
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
