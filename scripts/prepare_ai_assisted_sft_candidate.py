#!/usr/bin/env python3
"""Validate private AI-assisted rows and materialize a non-production SFT candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_ai_assisted import (
    audit_ai_assisted_candidate_rows,
    build_ai_assisted_manifest,
    load_ai_assisted_pilot_plan,
    select_owner_review_sample,
    to_sft_candidate_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}")
            rows.append(value)
    return rows


def _write_json_atomic(path: Path, value: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


MappingLike = dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--owner-sample", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.audit, args.manifest, args.owner_sample, args.sft_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite outputs: " + ",".join(existing))

    plan, coverage = load_ai_assisted_pilot_plan(
        args.plan,
        repository_root=args.repository_root,
    )
    rows = _read_jsonl(args.input)
    audit = audit_ai_assisted_candidate_rows(
        rows,
        coverage_plan=coverage,
        plan=plan,
    )
    owner_sample = select_owner_review_sample(rows)
    candidates = to_sft_candidate_records(
        rows,
        plan=plan,
        source_revision=audit["candidate_source_revision"],
    )
    manifest = build_ai_assisted_manifest(
        audit=audit,
        candidate_records=candidates,
        owner_sample=owner_sample,
    )
    _write_json_atomic(args.audit, audit)
    _write_jsonl_atomic(args.owner_sample, owner_sample)
    _write_jsonl_atomic(args.sft_output, candidates)
    _write_json_atomic(args.manifest, manifest)
    print(
        "PASS: "
        f"records={audit['records']} automated={audit['automated_review_passed_records']} "
        f"owner_sample={len(owner_sample)} production_eligible={manifest['production_eligible']} "
        f"source_revision={audit['candidate_source_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
