#!/usr/bin/env python3
"""Verify the v0.11 one-row repair run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_gpt_oss_teacher_dual_field_literal_repair_v11 import verify_dual_field_repair_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-reuse-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_dual_field_repair_run(plan_path=args.plan.resolve(), authorization_path=args.authorization.resolve(), runtime_reuse_receipt_path=args.runtime_reuse_receipt.resolve(), repository_root=args.repository_root.resolve(), artifact_root=args.artifact_root.resolve())
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
