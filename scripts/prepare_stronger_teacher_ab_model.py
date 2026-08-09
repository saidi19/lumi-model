#!/usr/bin/env python3
"""Import and verify the exact local Qwen3-14B Ollama alias."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab import load_plan, prepare_candidate_alias


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan.resolve(), repository_root=root)
    receipt = prepare_candidate_alias(plan=plan, repository_root=root)
    print(
        "PASS candidate Ollama alias imported and verified "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
