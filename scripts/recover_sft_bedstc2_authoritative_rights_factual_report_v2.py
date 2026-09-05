#!/usr/bin/env python3
"""Materialize the recovered Be-DSTC2 factual pass-1 report with zero document reads."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_factual_review_recovery_v2 as recovery
import lumi.posttraining_bedstc2_authoritative_rights_factual_review_v1 as review_v1


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("v0.2 factual recovery runtime cap reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(recovery.EXPECTED_LIMITS["maximum_seconds_per_process"])
    try:
        plan = recovery.load_plan(plan_path, root)
        report_path = recovery.resolve_repo_path(root, plan["output"]["report_path"])
        verification_path = recovery.resolve_repo_path(
            root, plan["output"]["verification_path"]
        )
        if report_path.exists() or verification_path.exists():
            raise FileExistsError("v0.2 factual recovery already started")
        with review_v1.network_disabled():
            witness = recovery.validate_completion_witness(root)
        identity = recovery.start_marker(report_path, recovery.REPORT_MARKER)
        marker_started = True
        with review_v1.network_disabled():
            report = recovery.assemble_report(
                recovery.file_sha256(root / recovery.EXPECTED_PLAN_PATH), witness
            )
        payload = recovery.canonical_json_file_bytes(report)
        if len(payload) + len(recovery.REPORT_MARKER) > recovery.EXPECTED_LIMITS[
            "maximum_evidence_write_bytes"
        ]:
            raise ValueError("v0.2 recovery evidence write cap exceeded")
        recovery.finish_marker(
            report_path, recovery.REPORT_MARKER, payload, identity
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "factual_document_passes_completed": 1,
                    "raw_document_reads": 0,
                    "production_eligible": False,
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
                    "retry_authorized": False,
                    "status": "FAIL_CLOSED",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
