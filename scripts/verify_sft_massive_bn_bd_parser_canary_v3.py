#!/usr/bin/env python3
"""Run zero-source-row structural verification for the MASSIVE parser canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lumi.posttraining_massive_bn_bd_parser_canary_v3 as canary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json"
DEFAULT_REPORT = ROOT / "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json"
DEFAULT_VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-verification-v0.3.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args()

    plan = canary.load_plan(args.plan, repository_root=ROOT)
    expected_report = (ROOT / plan["output"]["report_path"]).resolve()
    expected_verification = (ROOT / plan["output"]["verification_path"]).resolve()
    if args.report.resolve() != expected_report:
        raise ValueError("parser-canary report path changed")
    if args.verification.resolve() != expected_verification:
        raise ValueError("parser-canary verification path changed")
    if args.verification.exists() or args.verification.is_symlink():
        raise FileExistsError("refusing to overwrite parser-canary verification")
    report = canary._read_object(args.report)
    verification = canary.verify_execution(
        plan=plan,
        report=report,
        plan_file_sha256=canary.sha256_file(args.plan),
        report_file_sha256=canary.sha256_file(args.report),
        repository_root=ROOT,
    )
    canary._write_fresh(args.verification, canary._json_bytes(verification), mode=0o644)
    print(
        json.dumps(
            {
                "candidate_audit_reads": verification["resource_accounting"][
                    "candidate_audit_reads"
                ],
                "source_row_reads": 0,
                "status": verification["status"],
                "verification_path": str(args.verification.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

