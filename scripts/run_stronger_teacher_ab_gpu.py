#!/usr/bin/env python3
"""Run the authorized candidate-only A10 continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab_gpu import run_gpu_continuation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-continuation-authorization", type=Path, required=True)
    parser.add_argument("--gpu-authorization", type=Path, required=True)
    parser.add_argument("--remote-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    args = parser.parse_args()
    report = run_gpu_continuation(
        base_plan_path=args.base_plan.resolve(),
        continuation_authorization_path=args.base_continuation_authorization.resolve(),
        gpu_authorization_path=args.gpu_authorization.resolve(),
        remote_receipt_path=args.remote_receipt.resolve(),
        repository_root=args.repository_root.resolve(),
        ssh_key=args.ssh_key.resolve(),
        known_hosts=args.known_hosts.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
