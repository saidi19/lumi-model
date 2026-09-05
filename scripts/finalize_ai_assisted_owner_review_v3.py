#!/usr/bin/env python3
"""Validate owner decisions and create a raw-text-free v0.3 review receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_owner_review import (
    read_json,
    read_jsonl,
    validate_owner_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
    manifest = read_json(args.manifest)
    sample = read_jsonl(args.sample)
    if args.audit is None:
        sample_sha256 = str(manifest["owner_sample_sha256"])
    else:
        audit = read_json(args.audit)
        if audit.get("records") != len(sample):
            raise ValueError("full owner-review row count does not match the audit")
        sample_sha256 = str(audit["candidate_source_revision"])
    receipt = validate_owner_decisions(
        read_json(args.decisions),
        sample_rows=sample,
        sample_sha256=sample_sha256,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.receipt)
    print(
        "PASS: "
        f"accepted={receipt['accepted_records']} "
        f"rejected={receipt['rejected_records']} "
        f"status={receipt['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
