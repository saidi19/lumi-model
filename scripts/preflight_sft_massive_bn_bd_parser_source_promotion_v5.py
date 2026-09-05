#!/usr/bin/env python3
"""Run the bounded MASSIVE bn-BD source-promotion preflight."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import lumi.posttraining_massive_bn_bd_parser_source_promotion_preflight_v5 as contract


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("MASSIVE source-promotion preflight runtime cap reached")


def _validate_execution_limits() -> None:
    if contract.EXPECTED_LIMITS != {
        "passes": 2,
        "maximum_baseline_candidate_row_reads": 629_432,
        "maximum_massive_candidate_row_reads": 128,
        "maximum_total_candidate_row_reads": 629_560,
        "maximum_source_line_reads": 0,
        "maximum_test_object_reads": 0,
        "maximum_registry_writes": 0,
        "maximum_private_write_bytes": 4_194_304,
        "maximum_runtime_seconds_per_process": 1_800,
        "maximum_runtime_seconds_total": 3_600,
        "maximum_network_bytes": 0,
        "maximum_gpu_use": 0,
        "maximum_inference_calls": 0,
        "maximum_training_actions": 0,
        "maximum_retries_after_started_pass": 0,
    }:
        raise ValueError("MASSIVE source-promotion execution limits changed")


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
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"])
    try:
        _validate_execution_limits()
        plan = contract.load_plan(plan_path, root)
        report_path, _ = contract.preflight_runner(plan, root)
        marker_identity = contract.start_marker(
            report_path,
            contract.EXPECTED_RECEIPT_MARKER,
        )
        marker_started = True
        plan_file_sha256 = contract.file_sha256(plan_path)
        report = contract.build_report(
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            repository_root=root,
            started=started,
        )
        contract.validate_report(report)
        report_payload = contract.json_bytes(report)
        prospective_verification = contract.build_verification(
            report,
            report_file_sha256=contract.sha256_bytes(report_payload),
            report_file_bytes=len(report_payload),
            plan_file_sha256=plan_file_sha256,
        )
        contract.validate_verification(prospective_verification)
        verification_payload = contract.json_bytes(prospective_verification)
        physical_writes = contract.prospective_private_write_bytes(
            plan=plan,
            root=root,
            report_bytes=report_payload,
            verification_bytes=verification_payload,
        )
        if physical_writes > contract.EXPECTED_LIMITS["maximum_private_write_bytes"]:
            raise ValueError("MASSIVE source-promotion private-write cap exceeded")
        if (
            time.monotonic() - started
            > contract.EXPECTED_LIMITS["maximum_runtime_seconds_per_process"]
        ):
            raise TimeoutError("MASSIVE source-promotion preflight runtime cap reached")
        contract.finish_marker(
            report_path,
            contract.EXPECTED_RECEIPT_MARKER,
            report_payload,
            marker_identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "cross_source_collision_rows": report["cross_source_audit"][
                        "total_collision_rows"
                    ],
                    "production_eligible": False,
                    "registry_successor_materialized": False,
                    "report_file_sha256": contract.sha256_bytes(report_payload),
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
