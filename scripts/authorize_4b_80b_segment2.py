#!/usr/bin/env python3
"""Write one exact owner-approved Lumi Segment-2 authorization."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.segment2_continuation import write_segment2_owner_authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--exact-owner-approval", type=Path, required=True)
    parser.add_argument("--instance-ip", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = write_segment2_owner_authorization(
        plan_path=args.plan,
        output=args.output,
        exact_owner_approval=args.exact_owner_approval.read_text(
            encoding="utf-8"
        ).strip(),
        instance_ip=args.instance_ip,
        s3_bucket=args.s3_bucket,
    )
    print(
        "PASS: owner_approved "
        f"authorization={authorization['authorization_sha256']} "
        f"through_step={authorization['approved_through_step']} "
        f"token_cap={authorization['approved_token_cap']}"
    )


if __name__ == "__main__":
    main()
