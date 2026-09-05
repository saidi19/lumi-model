#!/usr/bin/env python3
"""Run the local-only deterministic gpt-oss tool-response repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_gpt_oss_deterministic_tool_repair_v6 import (
    materialize_deterministic_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--repository-root", default=Path("."), type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    report = materialize_deterministic_repair(
        plan_path=args.plan,
        repository_root=args.repository_root,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
