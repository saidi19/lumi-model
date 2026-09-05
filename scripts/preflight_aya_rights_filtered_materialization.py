#!/usr/bin/env python3
"""Materialize the non-mutating Aya rights-filtered preflight report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_aya_rights_filtered_preflight import (
    build_report,
    canonical_json_bytes,
    file_sha256,
    load_plan,
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya filtered preflight: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path, repository_root=root)
    report = build_report(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("Aya filtered preflight escaped evidence/")
    payload = canonical_json_bytes(report)
    if len(payload) > plan["limits"]["maximum_written_evidence_bytes"]:
        raise ValueError("Aya filtered preflight evidence exceeded write limit")
    _write_once(output, payload)
    print(
        "PASS: projected rights-filtered Aya successor; "
        f"records={report['projection']['output_records']} "
        f"train_tokens={report['projection']['retained']['splits']['train']['supervised_tokens']} "
        f"candidate={report['projection']['output_candidate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
