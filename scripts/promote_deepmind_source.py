#!/usr/bin/env python3
"""Create the immutable DeepMind Mathematics candidate-registry successor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_canary_run import file_sha256, json_file_bytes
from lumi.posttraining_deepmind_source_promotion import (
    build_source_promotion_receipt,
    build_successor_registry,
    load_policy_and_predecessor,
    load_source_promotion_plan,
    successor_payload_sha256,
    validate_records_against_successor,
    validate_successor_registry,
    verify_promotion_prerequisites,
    verify_source_promotion_receipt,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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
        raise ValueError("DeepMind successor registry escaped configs/")
    if not receipt_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind promotion receipt escaped evidence/")
    for output in (successor_path, receipt_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite DeepMind output: {output}")

    records, candidate_audit = verify_promotion_prerequisites(
        plan=plan,
        repository_root=root,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    successor_value = build_successor_registry(predecessor_value, plan=plan)
    successor_payload = json_file_bytes(successor_value)
    successor_sha256 = successor_payload_sha256(successor_value)
    successor = validate_successor_registry(
        policy=policy,
        predecessor_value=predecessor_value,
        successor_value=successor_value,
        successor_file_sha256=successor_sha256,
    )
    validate_records_against_successor(records, successor=successor)
    receipt = build_source_promotion_receipt(
        candidate_audit=candidate_audit,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        successor=successor,
    )
    verify_source_promotion_receipt(receipt)

    predecessor_path = root / plan["bindings"]["predecessor_registry"]["path"]
    data_contract_path = root / plan["bindings"]["data_contract"]["path"]
    predecessor_sha256 = file_sha256(predecessor_path)
    data_contract_sha256 = file_sha256(data_contract_path)
    _write_bytes_atomic(successor_path, successor_payload)
    _write_bytes_atomic(receipt_path, json_file_bytes(receipt))
    if file_sha256(predecessor_path) != predecessor_sha256:
        raise ValueError("DeepMind promotion mutated predecessor registry")
    if file_sha256(data_contract_path) != data_contract_sha256:
        raise ValueError("DeepMind promotion mutated active data contract")
    if _read_json(successor_path) != successor_value:
        raise ValueError("DeepMind successor write verification failed")
    print(
        "PASS: DeepMind Mathematics registered in immutable candidate successor; "
        f"successor={successor_sha256}; receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
