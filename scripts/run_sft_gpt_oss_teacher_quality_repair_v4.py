#!/usr/bin/env python3
"""Run only an explicitly authorized one-row gpt-oss prompt-quality repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_gpt_oss_teacher_quality_repair_v4 import run_quality_repair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-reuse-receipt", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    args = parser.parse_args()
    report = run_quality_repair(
        plan_path=args.plan.resolve(), authorization_path=args.authorization.resolve(),
        runtime_reuse_receipt_path=args.runtime_reuse_receipt.resolve(), repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(), ssh_key=args.ssh_key.resolve(), known_hosts=args.known_hosts.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
