#!/usr/bin/env python3
"""Project an isolated Aya validation split without materializing raw rows."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from lumi.posttraining_aya_validation import (
    build_split_projection,
    build_validation_preflight_report,
    load_aya_validation_plan,
    resolve_validation_binding,
    validate_validation_preflight_report,
    verify_validation_upstream,
)
from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_oasst1_full_run import load_runtime_components


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya validation report: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
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

    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_aya_validation_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    verify_validation_upstream(
        materialization_report=_read_json(
            resolve_validation_binding(root, bindings["materialization_report"])
        ),
        owner_review_receipt=_read_json(
            resolve_validation_binding(root, bindings["owner_review_receipt"])
        ),
        source_promotion_receipt=_read_json(
            resolve_validation_binding(root, bindings["source_promotion_receipt"])
        ),
    )
    rows = _read_jsonl(resolve_validation_binding(root, bindings["candidate_records"]))
    attribution = _read_jsonl(resolve_validation_binding(root, bindings["attribution_manifest"]))
    _policy, _registry, _benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    projection = build_split_projection(
        rows,
        attribution,
        plan=plan,
        template=template,
        tokenizer=tokenizer,
    )
    if time.monotonic() - started > plan["limits"]["maximum_runtime_minutes"] * 60:
        raise TimeoutError("Aya validation preflight exceeded its runtime bound")
    report = build_validation_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        projection=projection,
    )
    validate_validation_preflight_report(report)
    report_path = (root / plan["output"]["report_path"]).resolve()
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("Aya validation report must remain under evidence/")
    _write_json_atomic(report_path, report)
    print(
        "PASS: Aya validation split preflight; "
        f"train={projection['train']['records']}/"
        f"{projection['train']['supervised_tokens']}; "
        f"validation={projection['validation']['records']}/"
        f"{projection['validation']['supervised_tokens']}; "
        f"receipt={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
