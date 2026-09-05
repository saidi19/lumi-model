#!/usr/bin/env python3
"""Validate the local OASST1 full-curation plan and seal its preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_oasst1_curation import (
    build_preflight_report,
    file_sha256,
    json_file_bytes,
    load_oasst1_curation_plan,
    resolve_curation_binding,
    verify_preflight_report,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    plan = load_oasst1_curation_plan(plan_path, repository_root=root)
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("OASST1 preflight output must stay under evidence/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OASST1 preflight: {output}")

    bindings = plan["bindings"]
    bound_values = {
        name: _load_json(resolve_curation_binding(root, bindings[name], name=name))
        for name in (
            "pinned_metadata",
            "prior_reaudit_plan",
            "prior_reaudit_receipt",
            "raw_sample",
        )
    }
    report = build_preflight_report(
        metadata=bound_values["pinned_metadata"],
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        prior_plan=bound_values["prior_reaudit_plan"],
        prior_receipt=bound_values["prior_reaudit_receipt"],
        raw_sample=bound_values["raw_sample"],
    )
    verify_preflight_report(report)
    _write_bytes_atomic(output, json_file_bytes(report))
    print(
        "PASS: OASST1 full-curation local preflight; "
        f"sample_accepted={report['preflight_sample']['accepted_pairs']} "
        f"acquisition_authorized={report['acquisition_authorized']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
