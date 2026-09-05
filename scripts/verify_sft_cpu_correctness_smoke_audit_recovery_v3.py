#!/usr/bin/env python3
"""Verify the zero-data-read v0.3 CPU-smoke audit recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_sft_cpu_audit_recovery_v3 import verify_audit_recovery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "configs/lumi-4b-sft-cpu-correctness-smoke-audit-recovery-v0.3.json"
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verification = verify_audit_recovery(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
