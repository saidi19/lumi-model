#!/usr/bin/env python3
"""Build one planning-only, checkpoint-bound Lumi Segment-3 continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.segment3_continuation import (
    build_segment3_continuation_plan,
    owner_approval_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--checkpoint-gate-report", type=Path, required=True)
    parser.add_argument("--segment2-handoff", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance-ip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-request", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_segment3_continuation_plan(
        base_plan_path=args.base_plan,
        checkpoint_gate_report_path=args.checkpoint_gate_report,
        segment2_handoff_path=args.segment2_handoff,
        config_path=args.config,
        corpus_manifest_path=args.corpus_manifest,
        checkpoint_path=args.checkpoint_path,
        source_commit=args.source_commit,
    )
    if args.output.exists() or args.approval_request.exists():
        raise FileExistsError("refusing to overwrite sealed Segment-3 artifacts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    approval = owner_approval_text(plan, instance_ip=args.instance_ip)
    args.approval_request.write_text(approval + "\n", encoding="utf-8")
    print(
        "PASS: planning_only "
        f"plan={plan['plan_sha256']} run={plan['run_fingerprint']} "
        f"steps={plan['segment']['start_step']}->{plan['segment']['stop_step']} "
        f"measured_hours={plan['budget']['estimated_measured_training_hours']:.2f}"
    )


if __name__ == "__main__":
    main()
