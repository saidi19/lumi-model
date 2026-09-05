#!/usr/bin/env python3
"""Independently verify token-consistent smoke-only SFT shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_token_consistent_shards_v1 import (
    _write_atomic,
    build_token_consistent_shards_verification,
    json_file_bytes,
    load_token_consistent_shards_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_token_consistent_shards_plan(
        plan_path.resolve(), repository_root=root
    )
    verification = build_token_consistent_shards_verification(
        plan=plan,
        repository_root=root,
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("token-consistent verification escaped evidence/")
    _write_atomic(output, json_file_bytes(verification))
    print(
        "PASS: token-consistent six shards independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
