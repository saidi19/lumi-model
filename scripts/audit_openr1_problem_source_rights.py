#!/usr/bin/env python3
"""Materialize a raw-text-free OpenR1 problem-source rights audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumi.posttraining_first_party import file_sha256
from lumi.posttraining_openr1_rights import (
    audit_attributions,
    build_rights_report,
    load_rights_plan,
    resolve_plan_binding,
    validate_rights_report,
    write_json_atomic,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_rights_plan(plan_path, repository_root=root)
    attribution_path = resolve_plan_binding(
        root, plan["bindings"]["attribution_manifest"]
    )
    audit = audit_attributions(_read_jsonl(attribution_path))
    report = build_rights_report(
        plan,
        audit=audit,
        plan_file_sha256=file_sha256(plan_path),
    )
    validate_rights_report(report)
    output = (root / plan["output"]["report_path"]).resolve()
    write_json_atomic(output, report)
    print(
        "PASS: OpenR1 rights-map audit "
        f"records={audit['records']} rights_cleared=0 unresolved={audit['unresolved_records']} "
        f"production_eligible=False report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

