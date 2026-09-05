#!/usr/bin/env python3
"""Materialize the exact preflight-approved MASSIVE registry-v0.22 successor."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import lumi.posttraining_massive_bn_bd_parser_source_promotion_v6 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("MASSIVE registry-v0.22 promoter runtime cap reached")


def _validate_execution_limits() -> None:
    if contract.EXPECTED_LIMITS != {
        "passes": 2,
        "maximum_candidate_row_reads": 0,
        "maximum_source_line_reads": 0,
        "maximum_test_object_reads": 0,
        "maximum_predecessor_registry_writes": 0,
        "maximum_successor_registry_writes": 1,
        "maximum_private_write_bytes": 2_097_152,
        "maximum_runtime_seconds_per_process": 300,
        "maximum_runtime_seconds_total": 600,
        "maximum_network_bytes": 0,
        "maximum_gpu_use": 0,
        "maximum_inference_calls": 0,
        "maximum_training_actions": 0,
        "maximum_retries_after_started_pass": 0,
    }:
        raise ValueError("MASSIVE registry-v0.22 execution limits changed")


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    marker_started = False
    registry_written = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"])
    try:
        _validate_execution_limits()
        plan = contract.load_plan(plan_path, root)
        successor_path, receipt_path, _ = contract.preflight_promoter(plan, root)
        marker_identity = contract.start_marker(
            receipt_path,
            contract.EXPECTED_RECEIPT_MARKER,
        )
        marker_started = True
        plan_file_sha256 = contract.file_sha256(plan_path)
        receipt, successor = contract.build_receipt(
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
        )
        contract.validate_receipt(receipt)
        successor_payload = contract.json_bytes(successor)
        receipt_payload = contract.json_bytes(receipt)
        prospective_verification = contract.build_verification(
            receipt,
            receipt_file_sha256=contract.sha256_bytes(receipt_payload),
            receipt_file_bytes=len(receipt_payload),
            plan_file_sha256=plan_file_sha256,
        )
        contract.validate_verification(prospective_verification)
        verification_payload = contract.json_bytes(prospective_verification)
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
            raise TimeoutError("MASSIVE registry-v0.22 promoter runtime cap reached")
        contract.write_new_file(successor_path, successor_payload)
        registry_written = True
        contract.finish_marker(
            receipt_path,
            contract.EXPECTED_RECEIPT_MARKER,
            receipt_payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "production_eligible": False,
                    "receipt_file_sha256": contract.sha256_bytes(receipt_payload),
                    "receipt_internal_sha256": receipt["receipt_sha256"],
                    "registry_v0_22_file_sha256": contract.sha256_bytes(
                        successor_payload
                    ),
                    "registry_v0_22_sources": len(successor["sources"]),
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
                    "registry_written": registry_written,
                    "status": "FAIL_CLOSED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
