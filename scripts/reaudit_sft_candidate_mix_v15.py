#!/usr/bin/env python3
"""Run the registry-v0.21 full candidate-mix v0.15 re-audit once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v15 import (
    EXPECTED_LIMITS,
    json_bytes,
    load_plan,
    rebuild_report,
)


def _ensure_fresh(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to start v0.15 pass with existing output: {path}")


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_fresh(path)
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
    report_path = root / plan["output"]["report_path"]
    verification_path = root / plan["output"]["verification_path"]
    _ensure_fresh(report_path)
    _ensure_fresh(verification_path)
    report = rebuild_report(
        plan=plan,
        plan_file_sha256=file_sha256(args.plan),
        repository_root=root,
    )
    payload = json_bytes(report)
    if len(payload) > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("candidate-mix v0.15 report exceeds private-write cap")
    _write_atomic(report_path, payload)
    print(
        json.dumps(
            {
                "accepted_records": report["candidate_pool"][
                    "accepted_records_after_generic_quarantine"
                ],
                "candidate_row_reads": report["resource_accounting"][
                    "candidate_row_reads"
                ],
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
