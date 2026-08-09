#!/usr/bin/env python3
"""Independently verify the corrected A10 candidate completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab_gpu_completion import verify_completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--completion-plan", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_completion(
        base_plan_path=args.base_plan.resolve(),
        authorization_path=args.authorization.resolve(),
        completion_plan_path=args.completion_plan.resolve(),
        remote_receipt_path=args.remote_receipt.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
