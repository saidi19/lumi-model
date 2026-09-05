#!/usr/bin/env python3
"""Run the single bounded Be-DSTC2 public rights-document acquisition."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as contract


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("authoritative-rights acquisition runtime cap reached")


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
        signal.alarm(contract.EXPECTED_LIMITS["maximum_seconds"])
    try:
        plan = contract.load_plan(plan_path, root)
        _private, _manifest, _receipt, marker_path = contract.preflight_acquisition(
            plan, root
        )
        contract.start_marker(marker_path, contract.ACQUISITION_MARKER)
        marker_started = True
        receipt = contract.run_acquisition(plan, root)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "documents": receipt["accounting"]["documents"],
                    "inbound_bytes": receipt["accounting"]["inbound_bytes"],
                    "production_eligible": False,
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
