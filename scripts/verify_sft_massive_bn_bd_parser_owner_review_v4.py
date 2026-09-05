#!/usr/bin/env python3
"""Reconstruct and verify the MASSIVE bn-BD parser owner-review receipt."""

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

import lumi.posttraining_massive_bn_bd_parser_owner_review_v4 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("owner-review verifier runtime cap reached")


def _validate_execution_limits() -> None:
    if (
        contract.EXPECTED_LIMITS.get("passes") != 2
        or contract.EXPECTED_LIMITS.get("runtime_seconds_per_process") != 150
        or contract.EXPECTED_LIMITS.get("network_bytes") != 0
        or contract.EXPECTED_LIMITS.get("gpu_use") != 0
        or contract.EXPECTED_LIMITS.get("inference_calls") != 0
    ):
        raise ValueError("owner-review execution limits changed")


def _read_receipt_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        maximum_bytes = int(contract.EXPECTED_LIMITS["private_write_bytes"])
        if (
            not stat.S_ISREG(info.st_mode)
            or not 1 <= info.st_size <= maximum_bytes
        ):
            raise ValueError("receipt file metadata changed")
        remaining = info.st_size + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("receipt file size changed during read")
    return payload


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    receipt_path_arg = args.receipt if args.receipt.is_absolute() else root / args.receipt
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["runtime_seconds_per_process"])
    try:
        _validate_execution_limits()
        plan = contract.load_plan(plan_path, root)
        receipt_path, verification_path = contract.preflight_verifier(plan, root)
        if receipt_path_arg.resolve() != receipt_path:
            raise ValueError("receipt argument path changed")
        marker_identity = contract.start_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
        )
        marker_started = True
        receipt_payload = _read_receipt_once(receipt_path)
        actual_receipt = contract.strict_json_bytes(receipt_payload)
        if (
            not isinstance(actual_receipt, Mapping)
            or contract.json_bytes(actual_receipt) != receipt_payload
        ):
            raise ValueError("receipt canonical file encoding changed")
        plan_file_sha256 = contract.file_sha256(plan_path)
        expected_receipt = contract.build_receipt(
            plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
        )
        contract.validate_receipt(expected_receipt)
        if actual_receipt != expected_receipt:
            raise ValueError("owner-review receipt is not reproducible")
        contract.validate_receipt(actual_receipt)
        verification = contract.build_verification(
            actual_receipt,
            receipt_file_sha256=contract.sha256_bytes(receipt_payload),
            receipt_file_bytes=len(receipt_payload),
            plan_file_sha256=plan_file_sha256,
        )
        contract.validate_verification(verification)
        verification_payload = contract.json_bytes(verification)
        physical_writes = contract.prospective_private_write_bytes(
            plan=plan,
            root=root,
            receipt_bytes=receipt_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > contract.EXPECTED_LIMITS["private_write_bytes"]:
            raise ValueError("owner-review private-write cap exceeded")
        if (
            time.monotonic() - started
            > contract.EXPECTED_LIMITS["runtime_seconds_per_process"]
        ):
            raise TimeoutError("owner-review verifier runtime cap reached")
        contract.finish_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
            verification_payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": contract.sha256_bytes(
                        verification_payload
                    ),
                    "verification_internal_sha256": verification[
                        "verification_sha256"
                    ],
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
