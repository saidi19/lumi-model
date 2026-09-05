#!/usr/bin/env python3
"""Create the immutable reviewed-Aya candidate registry successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from lumi.posttraining_aya_source_promotion import (
    build_source_promotion_receipt,
    build_successor_registry,
    json_file_bytes,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    resolve_promotion_binding,
    validate_successor_registry,
    verify_source_promotion_prerequisites,
    verify_source_promotion_receipt,
)
from lumi.posttraining_first_party import file_sha256


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Aya output: {path}")
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
    plan = load_source_promotion_plan(plan_path, repository_root=root)
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    if not successor_path.is_relative_to(root / "configs"):
        raise ValueError("Aya successor registry must stay under configs/")
    if not receipt_path.is_relative_to(root / "evidence"):
        raise ValueError("Aya promotion receipt must stay under evidence/")
    for output in (successor_path, receipt_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Aya output: {output}")

    bindings = plan["bindings"]
    materialization = _read_json(
        resolve_promotion_binding(root, bindings["materialization_report"])
    )
    owner_receipt = _read_json(resolve_promotion_binding(root, bindings["owner_review_receipt"]))
    attribution = _read_jsonl(resolve_promotion_binding(root, bindings["attribution_manifest"]))
    verify_source_promotion_prerequisites(
        attribution_rows=attribution,
        materialization_report=materialization,
        owner_review_receipt=owner_receipt,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    successor_value = build_successor_registry(predecessor_value, plan=plan)
    successor_payload = json_file_bytes(successor_value)
    successor_sha256 = hashlib.sha256(successor_payload).hexdigest()
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=successor_sha256,
    )
    receipt = build_source_promotion_receipt(
        owner_review_receipt=owner_receipt,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
    )
    verify_source_promotion_receipt(receipt)
    _write_bytes_atomic(successor_path, successor_payload)
    _write_bytes_atomic(receipt_path, json_file_bytes(receipt))
    print(
        "PASS: reviewed Aya registered in immutable candidate successor; "
        f"successor={successor_sha256}; receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
