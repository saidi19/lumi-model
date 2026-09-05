#!/usr/bin/env python3
"""Run pass two of the hardened registry-v0.24 private SFT encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lumi.posttraining_token_consistent_shards_v2 as contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    started = time.monotonic()
    marker_path: Path | None = None
    try:
        plan = contract.load_token_consistent_shards_plan(
            plan_path,
            repository_root=root,
        )
        marker_path = contract.output_paths(plan, root)["verification_path"]
        verification = contract.run_verification_pass(
            plan=plan,
            plan_file_sha256=plan.file_sha256,
            repository_root=root,
            started=started,
        )
        print(
            json.dumps(
                {
                    "encoded_index_row_reads": verification["resource_accounting"][
                        "encoded_index_row_reads"
                    ],
                    "input_jsonl_row_reads": verification["resource_accounting"][
                        "input_jsonl_row_reads"
                    ],
                    "production_eligible": verification["production_eligible"],
                    "records": verification["fused_audit"]["records"],
                    "status": verification["status"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        # Never echo exception text: audit failures must remain raw-text-free.
        marker_present = marker_path is not None and os.path.lexists(marker_path)
        print(
            json.dumps(
                {
                    "exception_type": type(error).__name__,
                    "no_retry_marker_present": marker_present,
                    "status": "STOP",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
