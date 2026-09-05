#!/usr/bin/env python3
"""Build the private 40-row Lumi safety/tool owner-review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_safety_tool_owner_review import (
    build_safety_tool_owner_review_html,
    load_owner_review_plan,
    resolve_plan_binding,
    validate_owner_sample,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}")
            values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    artifacts_root = (root / "artifacts").resolve()
    if not output.resolve().is_relative_to(artifacts_root):
        raise ValueError("safety/tool owner-review page must stay under artifacts/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite owner-review page: {output}")

    plan = load_owner_review_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    sample = _read_jsonl(resolve_plan_binding(root, bindings["owner_sample"]))
    manifest = _read_json(resolve_plan_binding(root, bindings["pilot_manifest"]))
    evidence = _read_json(resolve_plan_binding(root, bindings["candidate_evidence"]))
    coverage = validate_owner_sample(
        sample,
        manifest=manifest,
        candidate_evidence=evidence,
        plan=plan,
    )
    html = build_safety_tool_owner_review_html(
        sample,
        sample_sha256=plan["candidate"]["sample_internal_sha256"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(html)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(
        "PASS: safety/tool owner-review page "
        f"records={coverage['records']} skills={coverage['skills']} "
        f"language_modes={coverage['language_modes']} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

