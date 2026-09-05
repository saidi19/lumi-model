#!/usr/bin/env python3
"""Execute the one-pass collision-aware MASSIVE bn-BD parser canary."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import lumi.posttraining_massive_bn_bd_parser_canary_v3 as canary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json"


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("bounded parser-canary runtime cap reached")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    marker: Path | None = None
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(canary.EXPECTED_LIMITS["maximum_runtime_seconds"])
    try:
        plan = canary.load_plan(args.plan, repository_root=root)
        if args.preflight_only:
            canary.load_runtime_contracts(plan, repository_root=root)
            private, marker, _report, _verification, _source = canary.preflight_paths(
                plan, repository_root=root
            )
            if private.exists() or marker.exists():
                raise FileExistsError("parser-canary preflight is not fresh")
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            print(
                json.dumps(
                    {
                        "candidate_rows_constructed": 0,
                        "source_row_reads": 0,
                        "status": "PASS_zero_row_preflight_fresh",
                    },
                    sort_keys=True,
                )
            )
            return 0
        private = canary._resolve_private_root(root, plan["output"]["private_root"])
        marker = private / plan["output"]["marker_name"]
        report, _report_path = canary.execute_canary(
            plan=plan,
            plan_file_sha256=canary.sha256_file(args.plan),
            repository_root=root,
        )
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "active_tool_api_credit_rows": 0,
                    "candidate_rows_constructed": report["decision"][
                        "candidate_rows_constructed"
                    ],
                    "owner_review_status": report["decision"]["owner_review_status"],
                    "source_row_reads": report["resource_accounting"]["source_row_reads"],
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        print(
            json.dumps(
                {
                    "marker_started": marker is not None and marker.exists(),
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

