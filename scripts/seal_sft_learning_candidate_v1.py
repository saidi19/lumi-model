#!/usr/bin/env python3
"""Materialize the authorized registry-v0.17 smoke-only learning seal."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_learning_candidate_seal_v1 import (
    build_learning_candidate_seal,
    json_file_bytes,
    load_learning_candidate_seal_plan,
    _write_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_learning_candidate_seal_plan(plan_path, repository_root=root)
    manifest = build_learning_candidate_seal(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    output = (root / plan["output"]["manifest_path"]).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("learning-candidate manifest escaped evidence/")
    _write_atomic(output, json_file_bytes(manifest))
    splits = manifest["output"]["splits"]
    print(
        "PASS: nonproduction learning candidate sealed; "
        f"train_rows={splits['train']['records']} "
        f"train_tokens={splits['train']['supervised_tokens']} "
        f"validation_rows={splits['validation']['records']} "
        f"validation_tokens={splits['validation']['supervised_tokens']} "
        f"seal={manifest['seal_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
