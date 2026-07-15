#!/usr/bin/env python3
"""Run the full-size tokenizer quota preflight without network or output writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_pipeline import preflight_sample_config
from lumi.development_config import compose_development_config

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "configs" / "tokenizer-sample-v0.3.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    config = compose_development_config(args.plan, repository_root=REPOSITORY_ROOT)
    result = preflight_sample_config(config)
    print("PASS: representative tokenizer sample quota preflight is green")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
