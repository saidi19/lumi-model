#!/usr/bin/env python3
"""Run the registry-v0.19 full candidate-mix v0.13 re-audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v13 import (
    json_bytes,
    load_plan,
    rebuild_report,
)


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate-mix v0.13 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = load_plan(args.plan, root)
    report = rebuild_report(
        plan=plan, plan_file_sha256=file_sha256(args.plan), repository_root=root
    )
    output = root / plan["output"]["report_path"]
    _write_atomic(output, json_bytes(report))
    print(
        json.dumps(
            {
                "accepted_records": report["candidate_pool"][
                    "accepted_records_after_generic_quarantine"
                ],
                "numeric_gap_to_10m": report["candidate_pool"]["numeric_gap_to_10m"],
                "report_sha256": report["report_sha256"],
                "status": report["status"],
                "train_supervised_tokens": report["candidate_pool"][
                    "accepted_train_supervised_tokens"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
