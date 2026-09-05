#!/usr/bin/env python3
"""Verify the local token-consistent candidate salvage preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_preflight_v1 import (
    build_report,
    build_verification,
    canonical_json_bytes,
    file_sha256,
    load_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    report_path = Path(args.report).resolve()
    plan = load_plan(plan_path, root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = build_report(plan, file_sha256(plan_path), root)
    if report != expected:
        raise ValueError("salvage preflight report is not reproducible")
    verification = build_verification(report, file_sha256(report_path))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(verification))


if __name__ == "__main__":
    main()

