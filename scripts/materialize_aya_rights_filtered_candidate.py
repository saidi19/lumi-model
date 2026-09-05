#!/usr/bin/env python3
"""Materialize the exact approved private rights-filtered Aya candidate."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from lumi.posttraining_aya_rights_filtered_materialization import (
    build_materialization,
    canonical_json_bytes,
    load_plan,
)
from lumi.posttraining_aya_rights_filtered_preflight import file_sha256


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    started = time.monotonic()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path, repository_root=root)
    candidate, manifest, report = build_materialization(
        plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )

    candidate_path = (root / plan["output"]["candidate_path"]).resolve()
    manifest_path = (root / plan["output"]["manifest_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    attempt_root = manifest_path.parent
    if any(path.exists() for path in (attempt_root, report_path)):
        raise FileExistsError("refusing to overwrite Aya rights-filtered materialization")
    if not candidate_path.is_relative_to(root.parent / "lumi-artifacts"):
        raise ValueError("Aya rights-filtered candidate escaped lumi-artifacts")
    if not report_path.is_relative_to(root / "evidence"):
        raise ValueError("Aya rights-filtered report escaped evidence")

    staging = attempt_root.parent / f".{attempt_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError("Aya rights-filtered staging path already exists")
    try:
        _write_bytes(
            staging / candidate_path.relative_to(attempt_root),
            candidate,
        )
        _write_bytes(
            staging / manifest_path.relative_to(attempt_root),
            canonical_json_bytes(manifest),
        )
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staging.replace(attempt_root)
        _write_bytes(report_path, canonical_json_bytes(report))
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    elapsed = time.monotonic() - started
    if elapsed > int(plan["limits"]["maximum_runtime_minutes"]) * 60:
        raise TimeoutError("Aya rights-filtered materialization exceeded runtime limit")
    written = (
        candidate_path.stat().st_size + manifest_path.stat().st_size + report_path.stat().st_size
    )
    if written > int(plan["limits"]["maximum_private_write_bytes"]):
        raise ValueError("Aya rights-filtered materialization exceeded write limit")
    print(
        "PASS: byte-exact Aya rights-filtered candidate materialized; "
        f"records={report['output']['records']} "
        f"train_tokens={report['output']['train_supervised_tokens']} "
        f"candidate={report['output']['candidate_file_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
