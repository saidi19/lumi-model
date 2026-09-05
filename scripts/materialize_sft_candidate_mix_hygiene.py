#!/usr/bin/env python3
"""Materialize exact private OASST1 and Aya hygiene successors."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_candidate_mix_hygiene_materialization import (
    load_hygiene_materialization_plan,
    materialize_hygiene_candidates,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_hygiene_materialization_plan(plan_path, repository_root=root)
    receipt = materialize_hygiene_candidates(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    print(root / plan["output"]["receipt_path"])
    print(receipt["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

