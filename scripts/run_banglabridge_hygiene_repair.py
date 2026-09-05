#!/usr/bin/env python3
"""Materialize the bounded private BanglaBridge hygiene-repair candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_banglabridge_hygiene_repair import (
    build_repair,
    load_repair_plan,
    write_bytes_atomic,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_repair_plan(plan_path.resolve(), repository_root=root)
    candidate_payload, manifest, report = build_repair(
        plan,
        plan_file_sha256=sha256_file(plan_path),
        repository_root=root,
    )
    candidate_path = (root / plan["output"]["candidate_path"]).resolve()
    manifest_path = (root / plan["output"]["manifest_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    if any(path.exists() for path in (candidate_path, manifest_path, report_path)):
        raise FileExistsError("refusing to overwrite BanglaBridge hygiene-repair output")
    if not candidate_path.is_relative_to(root.parent / "lumi-artifacts"):
        raise ValueError("BanglaBridge repaired candidate escaped lumi-artifacts")
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("BanglaBridge hygiene-repair report escaped evidence")
    write_bytes_atomic(candidate_path, candidate_payload)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(report_path, report)
    written = (
        candidate_path.stat().st_size
        + manifest_path.stat().st_size
        + report_path.stat().st_size
    )
    if written > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("BanglaBridge hygiene repair exceeded write limit")
    print(
        "PASS: repaired BanglaBridge candidate "
        f"records={report['output']['records']} "
        f"train_tokens={report['output']['train_supervised_tokens']} "
        f"candidate={report['output']['candidate_file_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

