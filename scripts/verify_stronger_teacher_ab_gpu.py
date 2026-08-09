#!/usr/bin/env python3
"""Independently verify the candidate-only A10 continuation."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab_gpu import verify_gpu_continuation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-continuation-authorization", type=Path, required=True)
    parser.add_argument("--gpu-authorization", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_gpu_continuation(
        base_plan_path=args.base_plan.resolve(),
        continuation_authorization_path=args.base_continuation_authorization.resolve(),
        gpu_authorization_path=args.gpu_authorization.resolve(),
        remote_receipt_path=args.remote_receipt.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(
        "PASS A10 continuation independently verified "
        f"verification={result['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
