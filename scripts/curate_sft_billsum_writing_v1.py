#!/usr/bin/env python3
"""Run the bounded BillSum writing-closure curation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_billsum_writing_curation_v1 import run_curation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    report, verification = run_curation(
        plan_path=args.plan, repository_root=args.repository_root
    )
    print(
        json.dumps(
            {
                "candidate_records": report["tokens"]["accepted_records"],
                "owner_review_packet_sha256": report["owner_review"]["packet_sha256"],
                "report_sha256": report["report_sha256"],
                "status": verification["status"],
                "train_supervised_tokens": report["tokens"]["train_supervised_tokens"],
                "validation_supervised_tokens": report["tokens"][
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
