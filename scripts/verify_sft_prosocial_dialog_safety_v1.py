#!/usr/bin/env python3
"""Independently verify the ProsocialDialog safety-helpfulness candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_prosocial_dialog_safety_curation_v1 import verify_curation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_curation(
        plan_path=args.plan,
        report_path=args.report,
        repository_root=args.repository_root,
    )
    print(
        json.dumps(
            {
                "candidate_records": verification["candidate_records"],
                "status": verification["status"],
                "train_supervised_tokens": verification["train_supervised_tokens"],
                "validation_supervised_tokens": verification[
                    "validation_supervised_tokens"
                ],
                "verification_sha256": verification["verification_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
