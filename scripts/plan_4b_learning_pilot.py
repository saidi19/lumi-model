#!/usr/bin/env python3
"""Plan one exact 4- or 8-GPU Lumi-4B learning pilot without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.distributed_learning_pilot import plan_4b_learning_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-cuda-planning", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = plan_4b_learning_pilot(
        contract_path=args.contract,
        manifest_path=args.manifest,
        heldout_manifest_path=args.heldout_manifest,
        config_path=args.config,
        source_commit=args.source_commit,
        world_size=args.world_size,
        require_cuda=not args.allow_non_cuda_planning,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: planning_only fingerprint={plan['run_fingerprint']} "
        f"world_size={plan['selected_topology']['world_size']} "
        f"steps={plan['target']['resume_through_step']} "
        f"tokens={plan['target']['tokens_through_resume']}"
    )


if __name__ == "__main__":
    main()
