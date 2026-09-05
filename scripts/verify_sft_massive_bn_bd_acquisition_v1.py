#!/usr/bin/env python3
"""Run the two sequential zero-network MASSIVE acquisition verification passes."""

from __future__ import annotations

import argparse
from pathlib import Path

import lumi.posttraining_massive_bn_bd_acquisition_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json"
DEFAULT_REPORT = ROOT / "evidence/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json"
DEFAULT_VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-massive-bn-bd-acquisition-verification-v0.1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args()

    plan = contract.load_plan(args.plan, repository_root=ROOT)
    report = contract._read_object(args.report)
    contract.validate_raw_text_free_report(report)
    verification = contract.build_verification(
        plan,
        report,
        repository_root=ROOT,
        report_file_sha256=contract.sha256_file(args.report),
    )
    contract.validate_verification(verification)
    output = plan["output"]["verification_path"]
    expected = contract._resolve_repo_path(ROOT, output)
    if args.verification.resolve() != expected:
        raise ValueError("verification output path changed")
    contract.write_json_fresh(args.verification, verification)
    print(args.verification.resolve())


if __name__ == "__main__":
    main()
