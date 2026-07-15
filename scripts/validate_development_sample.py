#!/usr/bin/env python3
"""Validate an all-component Lumi development sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_pipeline import validate_sample
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs" / "development-sample-v0.1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    config = compose_development_config(args.plan, repository_root=REPOSITORY_ROOT)
    result = validate_sample(args.sample, expected_config=config)
    print("PASS: all-component development sample is valid")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
