#!/usr/bin/env python3
"""Validate private v0.3 rows and materialize an owner-review-pending candidate."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_ai_assisted_v3 import (
    audit_ai_assisted_v3_rows,
    build_ai_assisted_v3_manifest,
    load_ai_assisted_v3_plan,
    select_owner_review_sample_v3,
    to_sft_candidate_records_v3,
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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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


def _require_private_artifact(path: Path, repository_root: Path) -> Path:
    root = repository_root.resolve()
    artifacts = (root / "artifacts").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(artifacts):
        raise ValueError(f"raw candidate data must remain under ignored artifacts/: {path}")
    return resolved


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
    root = args.repository_root.resolve()
    input_path = _require_private_artifact(args.input, root)
    outputs = tuple(
        _require_private_artifact(path, root)
        for path in (args.audit, args.manifest, args.owner_sample, args.sft_output)
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite outputs: " + ",".join(existing))

    plan, coverage = load_ai_assisted_v3_plan(args.plan, repository_root=root)
    rows = _read_jsonl(input_path)
    audit = audit_ai_assisted_v3_rows(rows, coverage_plan=coverage, plan=plan)
    owner_sample = select_owner_review_sample_v3(rows)
    candidates = to_sft_candidate_records_v3(
        rows,
        plan=plan,
        source_revision=audit["candidate_source_revision"],
    )
    manifest = build_ai_assisted_v3_manifest(
        audit=audit,
        candidate_records=candidates,
        owner_sample=owner_sample,
    )
    _write_json_atomic(outputs[0], audit)
    _write_json_atomic(outputs[1], manifest)
    _write_jsonl_atomic(outputs[2], owner_sample)
    _write_jsonl_atomic(outputs[3], candidates)
    print(
        "PASS: "
        f"records={audit['records']} "
        f"leakage={audit['quality_gate_results']} "
        f"owner_sample={len(owner_sample)} "
        f"production_eligible={manifest['production_eligible']} "
        f"source_revision={audit['candidate_source_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
