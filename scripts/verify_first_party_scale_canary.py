#!/usr/bin/env python3
"""Independently replay the Lumi first-party scale-canary evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_first_party_scale_canary import verify_scale_canary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify_scale_canary(
        plan_path=arguments.plan,
        repository_root=arguments.repository_root,
        artifact_root=arguments.artifact_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
