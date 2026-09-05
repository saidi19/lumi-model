#!/usr/bin/env python3
"""Audit the exact existing SFT source canary without retaining source text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_source_quality import (
    load_source_quality_audit_plan,
    run_source_quality_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--canary-attempt", type=Path, required=True)
    parser.add_argument("--ai-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--sft-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source audit: {args.output}")

    plan = load_source_quality_audit_plan(args.plan)
    policy = load_sft_policy(args.sft_contract)
    report = run_source_quality_audit(
        plan=plan,
        canary_attempt=args.canary_attempt,
        ai_candidate_jsonl=args.ai_candidate_jsonl,
        target_mix=policy.task_mix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(
        "PASS: representative source audit completed fail-closed; "
        f"promotions={report['production_promotions']} "
        f"findings={report['total_production_blocker_findings']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
