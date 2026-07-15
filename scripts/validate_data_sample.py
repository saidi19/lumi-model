#!/usr/bin/env python3
"""Validate a completed Lumi data sample and its provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.data_pipeline import load_sample_config, validate_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--quality-config", type=Path)
    args = parser.parse_args()
    if args.quality_config is not None and args.config is None:
        raise ValueError("--quality-config requires --config")
    expected_config = (
        load_sample_config(args.config, quality_path=args.quality_config)
        if args.config is not None
        else None
    )
    result = validate_sample(args.sample, expected_config=expected_config)
    print("PASS: sample data, safety scan, hashes, and manifest are valid")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
