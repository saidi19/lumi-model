#!/usr/bin/env python3
"""Independently verify the v0.2 tiny-CPU SFT correctness smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_sft_cpu_correctness_smoke_v2 import (
    verify_cpu_correctness_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/lumi-4b-sft-cpu-correctness-smoke-v0.2.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verification = verify_cpu_correctness_smoke(
        plan_path=args.plan.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
