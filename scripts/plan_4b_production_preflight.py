#!/usr/bin/env python3
"""Build a planning-only Lumi-4B pilot/full-run preflight report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.production_training import (
    build_production_training_preflight_plan,
    validate_production_training_preflight_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_production_training_preflight_plan(
        contract_path=args.contract,
        manifest_path=args.manifest,
        model_config_path=args.model_config,
        source_commit=args.source_commit,
    )
    validate_production_training_preflight_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(
        "PASS: "
        f"fingerprint={plan['plan_fingerprint']} "
        f"tokens={plan['corpus']['tokens']} "
        f"sequences={plan['corpus']['sequences']} "
        f"pilot_world_size={plan['pilot']['fixed']['world_size']}"
    )


if __name__ == "__main__":
    main()
