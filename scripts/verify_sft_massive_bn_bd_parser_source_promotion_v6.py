#!/usr/bin/env python3
"""Independently reconstruct and verify the MASSIVE registry-v0.22 promotion."""

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

import lumi.posttraining_massive_bn_bd_parser_source_promotion_v6 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("MASSIVE registry-v0.22 verifier runtime cap reached")


def _validate_execution_limits() -> None:
    limits = contract.EXPECTED_LIMITS
    if (
        limits.get("passes") != 2
        or limits.get("maximum_candidate_row_reads") != 0
        or limits.get("maximum_source_line_reads") != 0
        or limits.get("maximum_test_object_reads") != 0
        or limits.get("maximum_predecessor_registry_writes") != 0
        or limits.get("maximum_successor_registry_writes") != 1
        or limits.get("maximum_network_bytes") != 0
        or limits.get("maximum_gpu_use") != 0
        or limits.get("maximum_inference_calls") != 0
        or limits.get("maximum_training_actions") != 0
        or limits.get("maximum_retries_after_started_pass") != 0
    ):
        raise ValueError("MASSIVE registry-v0.22 execution limits changed")


def _read_receipt_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        maximum_bytes = int(contract.EXPECTED_LIMITS["maximum_private_write_bytes"])
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum_bytes:
            raise ValueError("MASSIVE source-promotion receipt metadata changed")
        chunks: list[bytes] = []
        remaining = info.st_size + 1
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
        raise ValueError("MASSIVE source-promotion receipt changed during read")
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
    receipt_arg = args.receipt if args.receipt.is_absolute() else root / args.receipt
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"])
    try:
        _validate_execution_limits()
        plan = contract.load_plan(plan_path, root)
        successor_path, receipt_path, verification_path = contract.preflight_verifier(
            plan,
            root,
        )
        if receipt_arg.resolve() != receipt_path:
            raise ValueError("MASSIVE source-promotion receipt argument changed")
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
            raise ValueError("MASSIVE source-promotion receipt encoding changed")
        plan_file_sha256 = contract.file_sha256(plan_path)
        expected_receipt, expected_successor = contract.build_receipt(
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
        )
        contract.validate_receipt(expected_receipt)
        if actual_receipt != expected_receipt:
            raise ValueError("MASSIVE source-promotion receipt is not reproducible")
        contract.validate_receipt(actual_receipt)
        contract.validate_materialized_successor(
            successor_path,
            repository_root=root,
        )
        successor_payload = contract.json_bytes(expected_successor)
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
            successor_bytes=successor_payload,
            receipt_bytes=receipt_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]:
            raise ValueError("MASSIVE registry-v0.22 private-write cap exceeded")
        if contract.runtime_cap_reached(started):
            raise TimeoutError("MASSIVE registry-v0.22 verifier runtime cap reached")
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
                    "registry_v0_22_file_sha256": contract.EXPECTED_SUCCESSOR[
                        "file_sha256"
                    ],
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
