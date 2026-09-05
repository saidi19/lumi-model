#!/usr/bin/env python3
"""Validate and seal Lumi's metadata-only response-language routing matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_language import (
    build_language_routing_contract_audit,
    load_language_routing_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load_language_routing_plan(args.config)
    report = build_language_routing_contract_audit(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
