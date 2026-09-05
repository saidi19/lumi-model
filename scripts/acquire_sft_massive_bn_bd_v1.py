#!/usr/bin/env python3
"""Execute the single bounded MASSIVE bn-BD train acquisition and audit."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_massive_bn_bd_acquisition_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json"


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("MASSIVE bounded acquisition runtime cap reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    marker_started = False
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(contract.EXPECTED_LIMITS["maximum_runtime_seconds"])
    try:
        plan = contract.load_plan(args.plan, repository_root=root)
        private_root, _report, _verification = contract.preflight_paths(plan, repository_root=root)
        marker = private_root / ".acquisition-started-v0.1"
        contract.start_marker(marker)
        marker_started = True
        report = contract.run_acquisition(plan, repository_root=root)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "candidate_rows_constructed": 0,
                    "production_eligible": False,
                    "source_rows_read": report["aggregate_audit"]["source_rows_read"],
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
