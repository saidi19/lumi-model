#!/usr/bin/env python3
"""Zero-network integrity verification of acquired Be-DSTC2 rights documents."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as contract


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("authoritative-rights acquisition verifier cap reached")


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
        verification_path = contract.resolve_repo_path(
            root, plan["output"]["verification_path"]
        )
        receipt_path = contract.resolve_repo_path(root, plan["output"]["receipt_path"])
        if not receipt_path.is_file() or verification_path.exists():
            raise FileExistsError("acquisition verification preflight changed")
        identity = contract.start_marker(
            verification_path, contract.VERIFICATION_MARKER
        )
        marker_started = True
        verification = contract.build_acquisition_verification(plan, root)
        payload = contract.canonical_json_file_bytes(verification)
        contract.finish_marker(
            verification_path,
            contract.VERIFICATION_MARKER,
            payload,
            identity,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": contract.sha256_bytes(payload),
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
