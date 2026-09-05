#!/usr/bin/env python3
"""Independently replay and verify candidate-mix v0.16."""

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

import lumi.posttraining_candidate_mix_reaudit_v16 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("candidate-mix v0.16 verifier runtime cap reached")


def _validate_execution_limits() -> None:
    limits = contract.EXPECTED_LIMITS
    if (
        limits.get("maximum_candidate_records_per_pass") != 314_780
        or limits.get("maximum_full_audit_passes") != 2
        or limits.get("maximum_total_candidate_row_reads") != 629_560
        or limits.get("maximum_source_line_reads") != 0
        or limits.get("maximum_test_object_reads") != 0
        or limits.get("maximum_registry_writes") != 0
        or limits.get("maximum_network_bytes") != 0
        or limits.get("maximum_gpu_use") != 0
        or limits.get("maximum_inference_calls") != 0
        or limits.get("maximum_training_actions") != 0
        or limits.get("maximum_retries_after_started_pass") != 0
    ):
        raise ValueError("candidate-mix v0.16 execution limits changed")


def _read_report_once(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        maximum_bytes = int(contract.EXPECTED_LIMITS["maximum_private_write_bytes"])
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum_bytes:
            raise ValueError("candidate-mix v0.16 report metadata changed")
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
        raise ValueError("candidate-mix v0.16 report changed during read")
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
        _validate_execution_limits()
        plan = contract.load_plan(plan_path, root)
        report_path, verification_path = contract.preflight_verifier(plan, root)
        if report_arg.resolve() != report_path:
            raise ValueError("candidate-mix v0.16 report argument changed")
        marker_identity = contract.start_marker(
            verification_path,
            contract.EXPECTED_VERIFICATION_MARKER,
        )
        marker_started = True
        report_payload = _read_report_once(report_path)
        actual_report = contract.strict_json_bytes(report_payload)
        if (
            not isinstance(actual_report, Mapping)
            or contract.json_bytes(actual_report) != report_payload
        ):
            raise ValueError("candidate-mix v0.16 report encoding changed")
        contract.validate_report(actual_report)
        plan_file_sha256 = contract.file_sha256(plan_path)
        if actual_report.get("plan_file_sha256") != plan_file_sha256:
            raise ValueError("candidate-mix v0.16 report plan binding changed")
        prospective = contract.build_verification(
            plan=plan,
            report=actual_report,
            report_file_sha256=contract.sha256_bytes(report_payload),
        )
        replay = contract.rebuild_report(
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
            started=started,
        )
        if replay != actual_report:
            raise ValueError("candidate-mix v0.16 independent full pass did not reproduce")
        verification = contract.build_verification(
            plan=plan,
            report=actual_report,
            report_file_sha256=contract.sha256_bytes(report_payload),
        )
        if verification != prospective:
            raise ValueError("candidate-mix v0.16 verification drifted")
        contract.validate_verification(verification)
        verification_payload = contract.json_bytes(verification)
        physical_writes = contract.prospective_private_write_bytes(
            plan=plan,
            root=root,
            report_bytes=report_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]:
            raise ValueError("candidate-mix v0.16 private-write cap exceeded")
        if (
            time.monotonic() - started
            > contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"]
        ):
            raise TimeoutError("candidate-mix v0.16 verifier runtime cap reached")
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
                    "candidate_row_reads": verification["resource_accounting"][
                        "candidate_row_reads"
                    ],
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
