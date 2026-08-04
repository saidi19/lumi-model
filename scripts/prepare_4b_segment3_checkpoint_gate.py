#!/usr/bin/env python3
"""Seal authorization for an already-pinned Segment-2 checkpoint plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from lumi.segment3_checkpoint_verification import (
    build_checkpoint_gate_authorization,
    write_checkpoint_gate_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approved-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approved_at = (
        None
        if args.approved_at is None
        else datetime.fromisoformat(args.approved_at)
    )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    authorization = build_checkpoint_gate_authorization(
        plan=plan,
        approved_at=approved_at,
    )
    write_checkpoint_gate_bundle(
        plan=plan,
        authorization=authorization,
        output=args.output,
    )
    print(
        f"PASS: plan={plan['plan_sha256']} "
        f"authorization={authorization['authorization_sha256']} "
        f"deadline={authorization['deadline_utc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
