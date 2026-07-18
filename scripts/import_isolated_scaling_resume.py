#!/usr/bin/env python3
"""Import a verified Lumi isolated step-100 checkpoint into a paid workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.scaling_run import import_isolated_scaling_resume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = import_isolated_scaling_resume(
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_commit=args.source_commit,
        source_export=args.source_export,
        workspace=args.workspace,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        "PASS: imported "
        f"step={report['authorization']['checkpoint_step']} "
        f"metrics_first_step={report['metrics_first_step']}"
    )


if __name__ == "__main__":
    main()
