#!/usr/bin/env python3
"""Run the zero-row-read production source-policy preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_production_source_policy_preflight_v1 import (
    _write_exclusive,
    build_report,
    file_sha256,
    json_file_bytes,
    load_plan,
    preflight_output_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    try:
        plan = load_plan(plan_path, repository_root=root)
        report_path, _ = preflight_output_paths(plan, root)
        report = build_report(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            repository_root=root,
        )
        payload = json_file_bytes(report)
        if len(payload) > plan["limits"]["maximum_private_write_bytes"]:
            raise ValueError("policy-preflight report exceeds write ceiling")
        _write_exclusive(report_path, payload)
        print(
            json.dumps(
                {
                    "candidate_row_reads": 0,
                    "production_eligible": False,
                    "report_file_sha256": file_sha256(report_path),
                    "report_internal_sha256": report["report_sha256"],
                    "status": report["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        print(json.dumps({"error_type": type(exc).__name__, "status": "STOP"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
