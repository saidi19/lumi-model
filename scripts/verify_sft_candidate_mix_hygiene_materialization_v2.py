#!/usr/bin/env python3
"""Independently verify the registry-v0.13 hygiene materialization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_hygiene_materialization_v2 import (
    EXPECTED_INPUTS,
    build_hygiene_materialization_v2_verification,
    json_file_bytes,
    load_hygiene_materialization_v2_plan,
    project_hygiene_candidates_v2,
    resolve_materialization_output,
    validate_hygiene_materialization_v2_manifest,
    validate_hygiene_materialization_v2_receipt,
)
from lumi.posttraining_candidate_mix_preflight import (
    _load_json,
    file_sha256,
)
from lumi.posttraining_candidate_mix_reaudit import resolve_binding


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {path}")
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
    plan_path = plan_path.resolve()
    plan = load_hygiene_materialization_v2_plan(plan_path, repository_root=root)
    projected_bytes, projected_manifest = project_hygiene_candidates_v2(
        plan=plan, repository_root=root
    )
    outputs = {
        name: resolve_materialization_output(root, relative, name=name)
        for name, relative in plan["output"].items()
        if name.endswith("_path")
    }
    for source_id, name in (
        ("openassistant_oasst1", "oasst1_candidate_path"),
        ("openassistant_oasst2", "oasst2_candidate_path"),
    ):
        observed = outputs[name].read_bytes()
        if observed != projected_bytes[source_id]:
            raise ValueError(f"materialized successor bytes changed: {source_id}")
    manifest = _load_json(outputs["manifest_path"])
    receipt = _load_json(outputs["receipt_path"])
    validate_hygiene_materialization_v2_manifest(manifest)
    validate_hygiene_materialization_v2_receipt(receipt)
    if manifest != projected_manifest:
        raise ValueError("materialized hygiene manifest does not reproduce")
    for source_id, expected in EXPECTED_INPUTS.items():
        path = resolve_binding(root, plan["bindings"][str(expected["binding"])], name=source_id)
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"materialization input changed: {source_id}")
    attempt_root = outputs["manifest_path"].parent
    inventory = sorted(path.name for path in attempt_root.iterdir() if path.is_file())
    verification = build_hygiene_materialization_v2_verification(
        plan=plan,
        manifest=manifest,
        manifest_file_sha256=file_sha256(outputs["manifest_path"]),
        receipt=receipt,
        receipt_file_sha256=file_sha256(outputs["receipt_path"]),
        inventory_before_verification=inventory,
    )
    payload = json_file_bytes(verification)
    existing_bytes = sum(path.stat().st_size for path in attempt_root.iterdir())
    if existing_bytes + len(payload) > plan["limits"]["maximum_total_write_bytes"]:
        raise ValueError("hygiene verification write ceiling exceeded")
    _write_atomic(outputs["verification_path"], payload)
    final_inventory = sorted(path.name for path in attempt_root.iterdir() if path.is_file())
    if final_inventory != [
        "materialization-manifest.json",
        "materialization-receipt.json",
        "materialization-verification.json",
        "oasst1-clean-candidate.jsonl",
        "oasst2-clean-candidate.jsonl",
    ]:
        raise ValueError("hygiene materialization final inventory changed")
    print(
        "PASS: registry-v0.13 hygiene materialization independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
