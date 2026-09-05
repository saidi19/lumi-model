#!/usr/bin/env python3
"""Build the registry-v0.9-bound raw-text-free Lumi SFT mix re-audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit_v4 import (
    json_file_bytes,
    load_candidate_mix_reaudit_v4_plan,
    rebuild_candidate_mix_reaudit_v4,
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate-mix v0.4 report: {path}")
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
    plan = load_candidate_mix_reaudit_v4_plan(plan_path, repository_root=root)
    report = rebuild_candidate_mix_reaudit_v4(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["report_path"]).resolve()
    _write_atomic(output, json_file_bytes(report))
    pool = report["candidate_pool"]
    print(
        "PASS: registry-v0.9 full candidate pool re-audited fail-closed; "
        f"accepted={report['generic_sft_validation']['accepted']} "
        f"train_tokens={pool['accepted_train_supervised_tokens']} "
        f"gap={pool['gap_to_10m_train_supervised_tokens']} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
