#!/usr/bin/env python3
"""Promote the owner-accepted ProsocialDialog source into registry v0.19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_prosocial_dialog_source_promotion_v19 import promote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    receipt, verification, draft = promote(args.plan, args.repository_root)
    print(
        json.dumps(
            {
                "candidate_records": receipt["candidate"]["records"],
                "draft_input_records": draft["expected"]["input_records"],
                "receipt_sha256": receipt["receipt_sha256"],
                "registry_sha256": receipt["registry"]["successor_sha256"],
                "status": verification["status"],
                "verification_sha256": verification["verification_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
