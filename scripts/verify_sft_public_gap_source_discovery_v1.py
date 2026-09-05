#!/usr/bin/env python3
"""Independently verify the public gap-source discovery report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_public_gap_source_discovery_v1 import (
    _read_object,
    build_report,
    build_verification,
    json_file_bytes,
    load_acquisition_preflight,
    load_plan,
    sha256_file,
    validate_verification,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/lumi-4b-sft-public-gap-source-discovery-v0.1.json"
DEFAULT_ACQUISITION_PREFLIGHT = (
    ROOT / "configs/lumi-4b-sft-public-gap-canary-acquisition-preflight-v0.1.json"
)
DEFAULT_REPORT = ROOT / "evidence/lumi-4b-sft-public-gap-source-discovery-v0.1.json"
DEFAULT_VERIFICATION = (
    ROOT / "evidence/lumi-4b-sft-public-gap-source-discovery-verification-v0.1.json"
)
MAXIMUM_VERIFICATION_BYTES = 65536


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--acquisition-preflight", type=Path, default=DEFAULT_ACQUISITION_PREFLIGHT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args()

    verification_path = args.verification.resolve()
    evidence_root = (ROOT / "evidence").resolve()
    if evidence_root not in verification_path.parents:
        raise ValueError("public-source verification escaped evidence/")
    if verification_path.exists():
        raise FileExistsError("refusing to overwrite public-source verification")

    plan = load_plan(args.plan, repository_root=ROOT)
    acquisition_preflight = load_acquisition_preflight(args.acquisition_preflight)
    report = _read_object(args.report)
    expected_report = build_report(
        plan,
        acquisition_preflight,
        plan_file_sha256=sha256_file(args.plan),
        acquisition_preflight_file_sha256=sha256_file(args.acquisition_preflight),
        repository_root=ROOT,
    )
    if report != expected_report:
        raise ValueError("materialized public-source discovery report changed")
    verification = build_verification(
        plan=plan,
        acquisition_preflight=acquisition_preflight,
        report=report,
        report_file_sha256=sha256_file(args.report),
    )
    validate_verification(verification)
    payload = json_file_bytes(verification)
    if len(payload) > MAXIMUM_VERIFICATION_BYTES:
        raise ValueError("public-source verification exceeded write limit")
    write_json_atomic(verification_path, verification)
    print(verification_path)


if __name__ == "__main__":
    main()
