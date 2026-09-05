#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_existing_candidate_cross_source_reaudit_v18 import (
    json_bytes,
    load_plan,
    rebuild_report,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    plan_path = root / "configs/lumi-4b-sft-existing-candidate-cross-source-reaudit-v0.18.json"
    output = root / "evidence/lumi-4b-sft-existing-candidate-cross-source-reaudit-v0.18.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite registry-v0.18 re-audit")
    plan = load_plan(plan_path, root)
    report = rebuild_report(
        plan=plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    output.write_bytes(json_bytes(report))


if __name__ == "__main__":
    main()

