#!/usr/bin/env python3
"""Independently verify a completed v0.9 full-canary resume."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_gpt_oss_teacher_canary_v9 import verify_full_canary_resume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-reuse-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_full_canary_resume(plan_path=args.plan.resolve(), authorization_path=args.authorization.resolve(), runtime_reuse_receipt_path=args.runtime_reuse_receipt.resolve(), repository_root=args.repository_root.resolve(), artifact_root=args.artifact_root.resolve())
    print(f"PASS: v0.9 full-canary resume verified; verification={verification['verification_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
