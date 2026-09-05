#!/usr/bin/env python3
"""Write the local-only existing-candidate reconciliation report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_existing_candidate_reconciliation_v1 import (
    build_report,
    canonical_json_bytes,
    file_sha256,
    load_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    plan = load_plan(plan_path, root)
    report = build_report(plan, file_sha256(plan_path), root)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))


if __name__ == "__main__":
    main()

