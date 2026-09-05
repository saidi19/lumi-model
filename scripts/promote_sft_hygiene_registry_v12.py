#!/usr/bin/env python3
"""Materialize the exact immutable source registry v0.12 successor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit import resolve_binding
from lumi.posttraining_hygiene_registry_preflight import json_file_bytes
from lumi.posttraining_hygiene_registry_promotion import (
    build_and_validate_successor,
    build_promotion_receipt,
    load_hygiene_registry_promotion_plan,
    verify_promotion_prerequisites,
)


def _write_atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite hygiene registry output: {path}")
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
    plan = load_hygiene_registry_promotion_plan(plan_path, repository_root=root)
    preflight_report, _, predecessor = verify_promotion_prerequisites(
        plan=plan,
        repository_root=root,
    )
    data_contract_path = resolve_binding(
        root, plan["bindings"]["data_contract"], name="data_contract"
    )
    successor, changed_sources = build_and_validate_successor(
        predecessor=predecessor,
        data_contract_path=data_contract_path,
    )
    receipt = build_promotion_receipt(
        plan_file_sha256=file_sha256(plan_path),
        preflight_report=preflight_report,
        changed_sources=changed_sources,
    )

    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    configs_root = (root / "configs").resolve()
    evidence_root = (root / "evidence").resolve()
    if not successor_path.is_relative_to(configs_root):
        raise ValueError("hygiene registry successor escaped configs/")
    if not receipt_path.is_relative_to(evidence_root):
        raise ValueError("hygiene registry receipt escaped evidence/")
    if successor_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing partial hygiene registry promotion overwrite")

    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    predecessor_sha256 = file_sha256(predecessor_path)
    data_contract_sha256 = file_sha256(data_contract_path)
    _write_atomic_new(successor_path, json_file_bytes(successor))
    _write_atomic_new(receipt_path, json_file_bytes(receipt))
    if file_sha256(predecessor_path) != predecessor_sha256:
        raise ValueError("hygiene registry promotion mutated v0.11")
    if file_sha256(data_contract_path) != data_contract_sha256:
        raise ValueError("hygiene registry promotion mutated data contract")
    print(
        "PASS: immutable source registry v0.12 materialized; "
        f"successor={receipt['registry']['successor_sha256']} "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
