#!/usr/bin/env python3
"""Verify the existing distributed CPU attempt without running training."""

import argparse
import json
from pathlib import Path

from lumi.posttraining_distributed_cpu_preflight_v2 import EXPECTED_PLAN_PATH
from lumi.posttraining_distributed_cpu_verification_v2 import verify_distributed_cpu_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=EXPECTED_PLAN_PATH)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = verify_distributed_cpu_preflight(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
