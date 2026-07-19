#!/usr/bin/env python3
"""Print durable progress for a Lumi corpus retention canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.corpus_retention import retention_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(retention_progress(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
