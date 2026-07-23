#!/usr/bin/env python3
"""Build Lumi-4B's exact planning-only 80B-token production specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.production_pretraining import plan_4b_80b_production


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = plan_4b_80b_production(
        contract_path=args.contract,
        manifest_path=args.manifest,
        config_path=args.config,
        pilot_result_path=args.pilot_result,
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: planning_only "
        f"fingerprint={plan['run_fingerprint']} "
        f"steps={plan['run']['maximum_steps']} "
        f"tokens={plan['corpus']['training_tokens']} "
        f"segments={len(plan['segments'])}"
    )


if __name__ == "__main__":
    main()
