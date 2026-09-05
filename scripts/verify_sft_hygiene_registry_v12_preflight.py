#!/usr/bin/env python3
"""Independently replay the non-mutating Lumi SFT registry-v0.12 projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_hygiene_registry_preflight import (
    build_hygiene_registry_preflight_verification,
    json_file_bytes,
    load_hygiene_registry_preflight_plan,
    rebuild_hygiene_registry_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_hygiene_registry_preflight_plan(plan_path, repository_root=root)
    report_path = root / plan["output"]["preflight_report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    replayed = rebuild_hygiene_registry_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    verification = build_hygiene_registry_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
        replayed_report=replayed,
    )
    output = root / plan["output"]["verification_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json_file_bytes(verification))
    print(output)
    print(verification["verification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

