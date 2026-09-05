#!/usr/bin/env python3
"""Independently replay and verify the Lumi candidate-mix re-audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_reaudit import (
    build_candidate_mix_reaudit_verification,
    file_sha256,
    json_file_bytes,
    load_candidate_mix_reaudit_plan,
    rebuild_candidate_mix_reaudit,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite re-audit verification: {path}")
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
    plan = load_candidate_mix_reaudit_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rebuilt = rebuild_candidate_mix_reaudit(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if rebuilt != report:
        raise ValueError("candidate-mix re-audit report does not reproduce exactly")
    verification = build_candidate_mix_reaudit_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(report_path),
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: candidate-mix re-audit independently reproduced; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
