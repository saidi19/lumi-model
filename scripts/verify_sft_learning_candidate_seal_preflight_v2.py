#!/usr/bin/env python3
"""Independently verify the registry-v0.24 seal preflight evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_learning_candidate_seal_preflight_v2 import (
    EXPECTED_OUTPUT,
    _load_object,
    _reserve_output,
    build_report,
    build_verification,
    file_sha256,
    json_file_bytes,
    load_plan,
    output_paths,
    validate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    report_path = Path(args.report).resolve()
    try:
        expected_plan_path = (
            root / "configs/lumi-4b-sft-learning-candidate-seal-preflight-v0.2.json"
        ).resolve()
        expected_report_path = (root / EXPECTED_OUTPUT["report_path"]).resolve()
        verification_path = (root / EXPECTED_OUTPUT["verification_path"]).resolve()
        if plan_path != expected_plan_path:
            raise ValueError("registry-v0.24 seal preflight plan path changed")
        if report_path != expected_report_path or not report_path.is_file():
            raise ValueError("registry-v0.24 seal preflight report path changed")
        with _reserve_output(verification_path) as output_handle:
            plan = load_plan(plan_path, repository_root=root)
            bound_report_path, bound_verification_path = output_paths(plan, root)
            if bound_report_path != report_path or bound_verification_path != verification_path:
                raise ValueError("registry-v0.24 seal preflight output path changed")
            report = _load_object(report_path)
            validate_report(report)
            replay = build_report(
                plan=plan,
                plan_file_sha256=file_sha256(plan_path),
                repository_root=root,
            )
            if replay != report:
                raise ValueError("registry-v0.24 seal preflight independent replay changed")
            verification = build_verification(
                report=report,
                report_file_sha256=file_sha256(report_path),
            )
            payload = json_file_bytes(verification)
            total_bytes = report_path.stat().st_size + len(payload)
            if total_bytes > plan["limits"]["maximum_private_write_bytes"]:
                raise ValueError("registry-v0.24 seal preflight evidence exceeds write ceiling")
            output_handle.write(payload)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        print(
            json.dumps(
                {
                    "candidate_row_reads": 0,
                    "dataset_sealed": False,
                    "production_eligible": False,
                    "status": verification["status"],
                    "verification_file_sha256": file_sha256(verification_path),
                    "verification_internal_sha256": verification["verification_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {"error_type": type(exc).__name__, "status": "STOP"},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
