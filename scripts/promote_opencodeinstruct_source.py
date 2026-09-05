#!/usr/bin/env python3
"""Create immutable source registry v0.11 with OpenCodeInstruct appended."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.data_quality import sha256_file
from lumi.posttraining_opencodeinstruct_source_promotion import (
    build_and_validate_successor,
    build_promotion_receipt,
    json_file_bytes,
    load_promotion_plan,
    resolve_binding,
    validate_promotion_receipt,
    verify_promotion_prerequisites,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OpenCodeInstruct output: {path}")
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
    plan = load_promotion_plan(plan_path, repository_root=root)
    _, preflight_report, predecessor = verify_promotion_prerequisites(
        plan=plan,
        repository_root=root,
    )
    policy = _load(
        resolve_binding(root, plan["bindings"]["data_contract"], name="data_contract")
    )
    successor = build_and_validate_successor(
        plan=plan,
        predecessor=predecessor,
        policy=policy,
    )
    receipt = build_promotion_receipt(
        plan=plan,
        plan_file_sha256=sha256_file(plan_path),
        preflight_report=preflight_report,
    )
    validate_promotion_receipt(receipt)

    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    if not successor_path.is_relative_to((root / "configs").resolve()):
        raise ValueError("OpenCodeInstruct successor escaped configs/")
    if not receipt_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("OpenCodeInstruct receipt escaped evidence/")
    if successor_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing partial OpenCodeInstruct promotion overwrite")

    predecessor_path = resolve_binding(
        root,
        plan["bindings"]["predecessor_registry"],
        name="predecessor_registry",
    )
    contract_path = resolve_binding(
        root,
        plan["bindings"]["data_contract"],
        name="data_contract",
    )
    predecessor_sha256 = sha256_file(predecessor_path)
    contract_sha256 = sha256_file(contract_path)
    _write_atomic(successor_path, json_file_bytes(successor))
    _write_atomic(receipt_path, json_file_bytes(receipt))
    if sha256_file(predecessor_path) != predecessor_sha256:
        raise ValueError("OpenCodeInstruct promotion mutated predecessor registry")
    if sha256_file(contract_path) != contract_sha256:
        raise ValueError("OpenCodeInstruct promotion mutated active data contract")
    print(
        "PASS: immutable OpenCodeInstruct registry v0.11 created; "
        f"successor={receipt['registry']['successor_sha256']} "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
