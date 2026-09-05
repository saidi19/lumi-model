#!/usr/bin/env python3
"""Materialize the authorized token-consistent smoke-only SFT shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_token_consistent_shards_v1 import (
    _write_atomic,
    json_file_bytes,
    load_token_consistent_shards_plan,
    materialize_token_consistent_shards,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_token_consistent_shards_plan(plan_path, repository_root=root)
    successor, receipt = materialize_token_consistent_shards(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    successor_path = (root / plan["output"]["successor_seal_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    evidence = (root / "evidence").resolve()
    if not successor_path.is_relative_to(evidence) or not receipt_path.is_relative_to(evidence):
        raise ValueError("token-consistent evidence escaped evidence/")
    _write_atomic(successor_path, json_file_bytes(successor))
    _write_atomic(receipt_path, json_file_bytes(receipt))
    print(
        "PASS: token-consistent smoke-only seal and six shards materialized; "
        f"tokens={successor['tokens']} supervised={successor['supervised_tokens']} "
        f"seal={successor['seal_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
