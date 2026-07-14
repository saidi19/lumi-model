#!/usr/bin/env python3
"""Print an analytical parameter breakdown without allocating model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi import LumiConfig, parameter_breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = LumiConfig(**json.load(handle))
    breakdown = parameter_breakdown(config)

    print(f"embeddings : {breakdown.embeddings:>15,}")
    print(f"attention  : {breakdown.attention:>15,}")
    print(f"mlp        : {breakdown.mlp:>15,}")
    print(f"norms      : {breakdown.norms:>15,}")
    print(f"output head: {breakdown.output_head:>15,}")
    print(f"total      : {breakdown.total:>15,} ({breakdown.total / 1e9:.6f}B)")


if __name__ == "__main__":
    main()
