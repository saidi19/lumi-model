#!/usr/bin/env python3
"""Build the private 32-row OASST1 owner-review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_oasst1_owner_review import (
    READY_SCHEMA_VERSION,
    build_oasst1_owner_review_html,
    load_owner_review_plan,
    resolve_plan_binding,
    sha256_value,
    validate_owner_sample,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OASST1 owner-review page: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    ready = args.ready_receipt if args.ready_receipt.is_absolute() else root / args.ready_receipt
    plan = load_owner_review_plan(plan_path, repository_root=root)
    private_root = (root / plan["private_output_root"]).resolve()
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("OASST1 owner-review page must stay under its private root")
    if not ready.resolve().is_relative_to((root / "evidence").resolve()):
        raise ValueError("OASST1 owner-review ready receipt must stay under evidence/")
    bindings = plan["bindings"]
    sample = _read_jsonl(resolve_plan_binding(root, bindings["owner_sample"]))
    report = _read_json(resolve_plan_binding(root, bindings["curation_report"]))
    verification = _read_json(resolve_plan_binding(root, bindings["curation_verification"]))
    coverage = validate_owner_sample(
        sample,
        curation_report=report,
        verification=verification,
        plan=plan,
    )
    html, payload_sha256 = build_oasst1_owner_review_html(sample)
    if payload_sha256 != coverage["payload_sha256"]:
        raise ValueError("OASST1 owner-review display payload changed")
    _write_text_atomic(output, html)
    receipt = {
        "authorization": plan["authorization"],
        "html_file_sha256": file_sha256(output),
        "observed_at": plan["observed_at"],
        "owner_review_complete": False,
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "review_payload_sha256": payload_sha256,
        "sample_coverage": coverage,
        "schema_version": READY_SCHEMA_VERSION,
        "status": "private_human_owner_review_page_ready_no_promotion",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    _write_json_atomic(ready, receipt)
    print(
        "PASS: OASST1 owner-review page "
        f"records={coverage['records']} split_task_pairs=8 "
        f"owner_review_complete=False output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

