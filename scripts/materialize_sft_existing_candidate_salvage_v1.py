#!/usr/bin/env python3
"""Materialize exact local Lumi SFT salvage subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_existing_candidate_salvage_materialization_v1 import (
    canonical_json_bytes,
    file_sha256,
    load_plan,
    materialize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = Path(args.plan).resolve()
    plan = load_plan(plan_path, root)
    report = materialize(
        plan, file_sha256(plan_path), root, Path(args.output_root).resolve()
    )
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report))


if __name__ == "__main__":
    main()

