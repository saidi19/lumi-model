#!/usr/bin/env python3
"""Materialize the exact registry-v0.13 OASST1/OASST2 hygiene successors."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_hygiene_materialization_v2 import (
    build_hygiene_materialization_v2_receipt,
    json_file_bytes,
    load_hygiene_materialization_v2_plan,
    project_hygiene_candidates_v2,
    resolve_materialization_output,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite materialization output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = plan_path.resolve()
    plan = load_hygiene_materialization_v2_plan(plan_path, repository_root=root)
    candidates, manifest = project_hygiene_candidates_v2(plan=plan, repository_root=root)
    manifest_payload = json_file_bytes(manifest)
    receipt = build_hygiene_materialization_v2_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        manifest=manifest,
        manifest_file_sha256=_sha256_bytes(manifest_payload),
    )
    receipt_payload = json_file_bytes(receipt)
    payloads = {
        "oasst1_candidate_path": candidates["openassistant_oasst1"],
        "oasst2_candidate_path": candidates["openassistant_oasst2"],
        "manifest_path": manifest_payload,
        "receipt_path": receipt_payload,
    }
    outputs = {
        name: resolve_materialization_output(root, plan["output"][name], name=name)
        for name in payloads
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("one or more hygiene materialization outputs already exist")
    total_write_bytes = sum(len(payload) for payload in payloads.values())
    if total_write_bytes > plan["limits"]["maximum_total_write_bytes"]:
        raise ValueError("hygiene materialization v0.2 write ceiling exceeded")
    for name in (
        "oasst1_candidate_path",
        "oasst2_candidate_path",
        "manifest_path",
        "receipt_path",
    ):
        _write_atomic(outputs[name], payloads[name])
        if file_sha256(outputs[name]) != _sha256_bytes(payloads[name]):
            raise ValueError(f"materialized output hash changed: {name}")
    print(
        "PASS: exact private OASST1/OASST2 hygiene successors materialized; "
        f"records={sum(item['records'] for item in manifest['outputs'].values())} "
        f"removals={manifest['removals']['records']} "
        f"manifest={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
