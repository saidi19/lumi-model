#!/usr/bin/env python3
"""Validate and seal the private Lumi safety/tool SFT lane candidate."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_safety_tool import (
    audit_safety_tool_rows,
    build_evidence,
    build_manifest,
    load_safety_tool_plan,
    select_owner_sample,
    to_candidate_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _private(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    artifacts = (root / "artifacts").resolve()
    if not resolved.is_relative_to(artifacts):
        raise ValueError(f"private safety/tool artifact must stay under artifacts/: {path}")
    return resolved


def _evidence_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    evidence = (root / "evidence").resolve()
    if not resolved.is_relative_to(evidence):
        raise ValueError(f"raw-free evidence must stay under evidence/: {path}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--owner-sample", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    input_path = _private(args.input, root)
    audit_path = _private(args.audit, root)
    manifest_path = _private(args.manifest, root)
    sample_path = _private(args.owner_sample, root)
    candidate_path = _private(args.candidate, root)
    evidence_path = _evidence_path(args.evidence, root)
    outputs = (audit_path, manifest_path, sample_path, candidate_path, evidence_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite outputs: " + ",".join(existing))

    plan = load_safety_tool_plan(plan_path, repository_root=root)
    rows = _read_jsonl(input_path)
    audit = audit_safety_tool_rows(rows, plan=plan)
    owner_sample = select_owner_sample(rows)
    candidates = to_candidate_records(
        rows,
        source_revision=audit["candidate_source_revision"],
    )
    manifest = build_manifest(
        audit=audit,
        candidate_records=candidates,
        owner_sample=owner_sample,
    )
    _write_json_atomic(audit_path, audit)
    _write_json_atomic(manifest_path, manifest)
    _write_jsonl_atomic(sample_path, owner_sample)
    _write_jsonl_atomic(candidate_path, candidates)
    evidence = build_evidence(
        audit=audit,
        audit_file_sha256=file_sha256(audit_path),
        candidate_file_sha256=file_sha256(candidate_path),
        manifest=manifest,
        manifest_file_sha256=file_sha256(manifest_path),
        owner_sample_file_sha256=file_sha256(sample_path),
        plan_file_sha256=file_sha256(plan_path),
        source_file_sha256=file_sha256(input_path),
    )
    _write_json_atomic(evidence_path, evidence)
    print(
        "PASS: safety/tool candidate "
        f"records={audit['records']} splits={audit['split_counts']} "
        f"owner_sample={len(owner_sample)} production_eligible=False "
        f"source_revision={audit['candidate_source_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
