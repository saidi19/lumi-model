#!/usr/bin/env python3
"""Project and verify the reviewed AI v0.3 candidate-registry successor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_ai_v3_source_promotion_preflight import (
    build_ai_v3_promotion_preflight_report,
    load_ai_v3_promotion_preflight_plan,
    verify_ai_v3_promotion_preflight_report,
)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite AI v0.3 preflight: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
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
    plan = load_ai_v3_promotion_preflight_plan(plan_path, repository_root=root)
    report = build_ai_v3_promotion_preflight_report(
        plan,
        plan_file_sha256=_file_sha256(plan_path),
        repository_root=root,
    )
    verify_ai_v3_promotion_preflight_report(report)
    output = (root / plan["output"]["preflight_report_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("AI v0.3 preflight report must stay under evidence/")
    _write_json_atomic(output, report)
    print(
        "PASS: AI v0.3 registry successor preflight verified; "
        f"successor={report['projected_successor']['sha256']}; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
