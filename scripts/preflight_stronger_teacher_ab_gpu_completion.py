#!/usr/bin/env python3
"""Build the exact corrected A10 continuation plan without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab_gpu_completion import build_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_preflight(
        base_plan_path=args.base_plan.resolve(),
        failure_receipt_path=args.failure_receipt.resolve(),
        remote_receipt_path=args.remote_receipt.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
