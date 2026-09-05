#!/usr/bin/env python3
"""Write the raw-text-free public gap-source discovery report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_public_gap_source_discovery_v1 import (
    build_report,
    json_file_bytes,
    load_acquisition_preflight,
    load_plan,
    sha256_file,
    validate_report,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-public-gap-source-discovery-v0.1.json"
DEFAULT_ACQUISITION_PREFLIGHT = (
    ROOT / "configs/lumi-4b-sft-public-gap-canary-acquisition-preflight-v0.1.json"
)
DEFAULT_REPORT = ROOT / "evidence/lumi-4b-sft-public-gap-source-discovery-v0.1.json"
MAXIMUM_REPORT_BYTES = 131072


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--acquisition-preflight", type=Path, default=DEFAULT_ACQUISITION_PREFLIGHT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report_path = args.report.resolve()
    evidence_root = (ROOT / "evidence").resolve()
    if evidence_root not in report_path.parents:
        raise ValueError("public-source discovery report escaped evidence/")
    if report_path.exists():
        raise FileExistsError("refusing to overwrite public-source discovery report")

    plan = load_plan(args.plan, repository_root=ROOT)
    acquisition_preflight = load_acquisition_preflight(args.acquisition_preflight)
    report = build_report(
        plan,
        acquisition_preflight,
        plan_file_sha256=sha256_file(args.plan),
        acquisition_preflight_file_sha256=sha256_file(args.acquisition_preflight),
        repository_root=ROOT,
    )
    validate_report(report)
    payload = json_file_bytes(report)
    if len(payload) > MAXIMUM_REPORT_BYTES:
        raise ValueError("public-source discovery report exceeded write limit")
    write_json_atomic(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
