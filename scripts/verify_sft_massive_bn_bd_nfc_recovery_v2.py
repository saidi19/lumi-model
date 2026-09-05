#!/usr/bin/env python3
"""Run zero-source-row structural verification for MASSIVE NFC recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lumi.posttraining_massive_bn_bd_nfc_recovery_v2 as recovery

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json"
DEFAULT_REPORT = ROOT / "evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json"
DEFAULT_VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-verification-v0.2.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args()

    plan = recovery.load_plan(args.plan, repository_root=ROOT)
    expected_report = (ROOT / plan["output"]["report_path"]).resolve()
    expected_verification = (ROOT / plan["output"]["verification_path"]).resolve()
    if args.report.resolve() != expected_report:
        raise ValueError("recovery report path changed")
    if args.verification.resolve() != expected_verification:
        raise ValueError("recovery verification path changed")
    report = recovery._read_object(args.report)
    verification = recovery.build_verification(
        plan=plan,
        report=report,
        plan_file_sha256=recovery.sha256_file(args.plan),
        report_file_sha256=recovery.sha256_file(args.report),
        repository_root=ROOT,
    )
    recovery.write_json_fresh(args.verification, verification)
    print(
        json.dumps(
            {
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
