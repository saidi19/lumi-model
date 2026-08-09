#!/usr/bin/env python3
"""Run the authorized Mac-local Lumi stronger-teacher paired A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab import run_paired_ab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--continuation-authorization", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_paired_ab(
        plan_path=args.plan.resolve(),
        continuation_authorization_path=args.continuation_authorization.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
