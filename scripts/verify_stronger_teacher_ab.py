#!/usr/bin/env python3
"""Independently verify the Lumi stronger-teacher paired A/B."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab import verify_paired_ab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--continuation-authorization", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_paired_ab(
        plan_path=args.plan.resolve(),
        continuation_authorization_path=args.continuation_authorization.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(
        "PASS paired A/B independently verified "
        f"machine_gate={verification['machine_gate_passed']} "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
