#!/usr/bin/env python3
"""Validate and optionally cost the pinned Lumi-1B scaling-pilot plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.scaling_pilot import (
    load_scaling_pilot_plan,
    plan_scaling_pilot,
    validate_scaling_pilot_inputs,
    write_scaling_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("configs/scaling-pilot-1b-v0.1.json"))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--heldout-selection-ledger", type=Path)
    parser.add_argument("--token-count-report", type=Path)
    parser.add_argument("--baseline-evaluation-report", type=Path)
    parser.add_argument("--tokens-per-second", type=float)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--fixed-overhead-hours", type=float, default=0.0)
    parser.add_argument("--credit-balance", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_paths = (
        args.source_manifest,
        args.heldout_manifest,
        args.heldout_selection_ledger,
        args.token_count_report,
        args.baseline_evaluation_report,
    )
    if any(path is not None for path in artifact_paths) and not all(
        path is not None for path in artifact_paths
    ):
        raise ValueError("all five local input paths must be supplied together")
    input_validation = None
    if all(path is not None for path in artifact_paths):
        plan = load_scaling_pilot_plan(args.plan)
        input_validation = validate_scaling_pilot_inputs(
            plan,
            source_manifest_path=args.source_manifest,
            heldout_manifest_path=args.heldout_manifest,
            heldout_selection_ledger_path=args.heldout_selection_ledger,
            token_count_report_path=args.token_count_report,
            baseline_evaluation_report_path=args.baseline_evaluation_report,
        )
    report = plan_scaling_pilot(
        plan_path=args.plan,
        tokens_per_second=args.tokens_per_second,
        hourly_rate=args.hourly_rate,
        fixed_overhead_hours=args.fixed_overhead_hours,
        credit_balance=args.credit_balance,
        input_validation=input_validation,
    )
    if args.output is not None:
        write_scaling_report(args.output, report)
    final = report["stages"][-1]
    message = (
        f"PASS: Lumi-1B={report['model']['parameters']:,} parameters; "
        f"final={final['processed_tokens']:,} tokens "
        f"({final['processed_tokens_per_parameter']:.3f} tokens/parameter)"
    )
    if final.get("projection") is not None:
        projection = final["projection"]
        message += (
            f"; projected={projection['compute_hours']:.2f} compute-hours, "
            f"USD {projection['estimated_cost']:.2f}"
        )
    else:
        message += "; cost pending measured 1B throughput and selected hourly rate"
    print(message)


if __name__ == "__main__":
    main()
