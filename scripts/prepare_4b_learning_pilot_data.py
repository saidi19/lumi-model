#!/usr/bin/env python3
"""Prepare Lumi's exact final-corpus prefix for the bounded 4B learning pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.distributed_learning_pilot import prepare_learning_pilot_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_learning_pilot_fixture(
        contract_path=args.contract,
        final_manifest_path=args.final_manifest,
        source_shard_path=args.source_shard,
        output=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
