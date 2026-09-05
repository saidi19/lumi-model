#!/usr/bin/env python3
"""Independently reconstruct and verify source registry v0.14 promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_candidate_mix_reaudit import resolve_binding
from lumi.posttraining_hygiene_registry_preflight_v2 import json_file_bytes
from lumi.posttraining_hygiene_registry_promotion_v2 import (
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_SHA256,
    build_and_validate_registry_v14_successor,
    build_registry_v14_promotion_receipt,
    build_registry_v14_promotion_verification,
    load_hygiene_registry_v14_promotion_plan,
    validate_registry_v14_promotion_receipt,
    verify_v14_promotion_prerequisites,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite registry-v0.14 verification: {path}")
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
    plan = load_hygiene_registry_v14_promotion_plan(plan_path, repository_root=root)
    preflight_report, _, predecessor = verify_v14_promotion_prerequisites(
        plan=plan,
        repository_root=root,
    )
    data_contract_path = resolve_binding(
        root, plan["bindings"]["data_contract"], name="data_contract"
    )
    expected_successor, changed_sources = build_and_validate_registry_v14_successor(
        predecessor=predecessor,
        data_contract_path=data_contract_path,
    )
    successor_path = (root / plan["output"]["successor_registry_path"]).resolve()
    if _load(successor_path) != expected_successor:
        raise ValueError("registry-v0.14 does not reproduce")
    if (
        file_sha256(successor_path) != SUCCESSOR_REGISTRY_SHA256
        or successor_path.stat().st_size != SUCCESSOR_REGISTRY_BYTES
    ):
        raise ValueError("registry-v0.14 file identity changed")

    receipt_path = (root / plan["output"]["receipt_path"]).resolve()
    observed_receipt = _load(receipt_path)
    validate_registry_v14_promotion_receipt(observed_receipt)
    expected_receipt = build_registry_v14_promotion_receipt(
        plan_file_sha256=file_sha256(plan_path),
        preflight_report=preflight_report,
        changed_sources=changed_sources,
    )
    if observed_receipt != expected_receipt:
        raise ValueError("registry-v0.14 promotion receipt does not reproduce")

    predecessor_path = resolve_binding(
        root, plan["bindings"]["predecessor_registry"], name="predecessor_registry"
    )
    verification = build_registry_v14_promotion_verification(
        predecessor_file_sha256=file_sha256(predecessor_path),
        receipt_file_sha256=file_sha256(receipt_path),
        receipt_internal_sha256=observed_receipt["receipt_sha256"],
        successor_file_sha256=file_sha256(successor_path),
        successor_file_bytes=successor_path.stat().st_size,
    )
    output_path = (root / plan["output"]["verification_path"]).resolve()
    if not output_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("registry-v0.14 verification escaped evidence/")
    _write_atomic_new(output_path, json_file_bytes(verification))
    print(
        "PASS: source registry v0.14 independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
