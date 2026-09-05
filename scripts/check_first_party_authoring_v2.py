#!/usr/bin/env python3
"""Audit private message-native human SFT rows without printing raw text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_first_party import canonical_json_bytes
from lumi.posttraining_first_party_v2 import (
    audit_first_party_authoring_rows_v2,
    load_first_party_authoring_v2_plan,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"authoring row {line_number} must be an object")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan, coverage = load_first_party_authoring_v2_plan(
        args.config,
        repository_root=args.repository_root,
    )
    report = audit_first_party_authoring_rows_v2(
        _load_jsonl(args.rows),
        plan=plan,
        coverage=coverage,
    )
    payload = canonical_json_bytes(report) + b"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
        args.output.write_bytes(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
