#!/usr/bin/env python3
"""Run the single bounded v0.2 local distributed CPU preflight attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_distributed_cpu_preflight_v2 import (
    EXPECTED_PLAN_PATH,
    run_distributed_cpu_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=EXPECTED_PLAN_PATH)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-port", type=int, default=29_650)
    args = parser.parse_args()
    report = run_distributed_cpu_preflight(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
        base_port=args.base_port,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
