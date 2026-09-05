#!/usr/bin/env python3
"""Build a private self-contained owner-review page for the v0.3 sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_owner_review import build_owner_review_html, read_json, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    manifest = read_json(args.manifest)
    sample = read_jsonl(args.sample)
    if args.audit is None:
        sample_sha256 = str(manifest["owner_sample_sha256"])
    else:
        audit = read_json(args.audit)
        if audit.get("records") != len(sample):
            raise ValueError("full owner-review row count does not match the audit")
        sample_sha256 = str(audit["candidate_source_revision"])
    html = build_owner_review_html(
        sample,
        sample_sha256=sample_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(html)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    scope = "full_candidate" if args.audit is not None else "spot_check"
    print(f"PASS: owner_review_rows={len(sample)} scope={scope} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
