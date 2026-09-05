#!/usr/bin/env python3
"""Finalize the BanglaBridge expanded-delta human owner review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_banglabridge_expanded_delta_owner_review_v1 import finalize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    receipt, verification = finalize(args.plan, args.repository_root)
    print(
        json.dumps(
            {
                "accepted_records": receipt["accepted_records"],
                "receipt_sha256": receipt["receipt_sha256"],
                "status": verification["status"],
                "verification_sha256": verification["verification_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
