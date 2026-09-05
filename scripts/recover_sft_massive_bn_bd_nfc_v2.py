#!/usr/bin/env python3
"""Execute the bounded two-pass local MASSIVE bn-BD NFC recovery."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_massive_bn_bd_nfc_recovery_v2 as recovery

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json"


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("bounded NFC recovery runtime cap reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    marker_started = False
    marker: Path | None = None
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(recovery.EXPECTED_LIMITS["maximum_runtime_seconds"])
    try:
        plan = recovery.load_plan(args.plan, repository_root=root)
        marker, _report_path, _verification_path, _source = recovery.preflight_paths(
            plan, repository_root=root
        )
        report, _written_report = recovery.execute_recovery(
            plan=plan,
            plan_file_sha256=recovery.sha256_file(args.plan),
            repository_root=root,
        )
        marker_started = marker.exists()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "active_tool_api_credit_rows": 0,
                    "candidate_rows_constructed": 0,
                    "nfc_recovery_gate_pass": report["decision"]["nfc_recovery_gate_pass"],
                    "source_row_reads": report["resource_accounting"]["source_row_reads"],
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        marker_started = marker is not None and marker.exists()
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
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
