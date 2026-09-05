#!/usr/bin/env python3
"""Materialize the exact reviewed teacher-canary candidate and registry v0.15."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lumi.posttraining_teacher_canary_source_promotion import (
    CANONICAL_CANDIDATE_BYTES,
    CANONICAL_CANDIDATE_PATH,
    CANONICAL_CANDIDATE_SHA256,
    RECEIPT_PATH,
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_PATH,
    SUCCESSOR_REGISTRY_SHA256,
    VERIFICATION_PATH,
    build_promotion_receipt,
    file_sha256,
    json_file_bytes,
    jsonl_file_bytes,
    load_materialization_plan,
    reconstruct_materialized_values,
    resolve_binding,
    validate_canonical_rows,
    validate_promotion_receipt,
)


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale teacher-canary staging file exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _materialize_transaction(outputs: list[tuple[Path, bytes]]) -> None:
    if any(path.exists() for path, _ in outputs):
        present = [str(path) for path, _ in outputs if path.exists()]
        raise FileExistsError(f"refusing teacher-canary output overwrite: {present}")
    staged: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for path, payload in outputs:
            staged.append((path, _stage(path, payload)))
        for path, temporary in staged:
            temporary.replace(path)
            installed.append(path)
        for parent in {path.parent for path, _ in outputs}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        for _, temporary in staged:
            if temporary.exists():
                temporary.unlink()
        for path in reversed(installed):
            if path.exists():
                path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_materialization_plan(plan_path, repository_root=root)
    canonical, successor, report = reconstruct_materialized_values(
        plan=plan,
        repository_root=root,
    )
    validate_canonical_rows(canonical)
    receipt = build_promotion_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        report=report,
    )
    validate_promotion_receipt(receipt)

    candidate_path = (root / CANONICAL_CANDIDATE_PATH).resolve()
    registry_path = (root / SUCCESSOR_REGISTRY_PATH).resolve()
    receipt_path = (root / RECEIPT_PATH).resolve()
    verification_path = (root / VERIFICATION_PATH).resolve()
    expected_candidate_parent = (
        root.parent
        / "lumi-artifacts/4b-sft-first-party-teacher-canary-v0.1/"
        "e80b7aa7936b/attempt-018/private"
    ).resolve()
    if candidate_path.parent != expected_candidate_parent:
        raise ValueError("teacher-canary canonical candidate escaped exact private path")
    if not registry_path.is_relative_to((root / "configs").resolve()):
        raise ValueError("teacher-canary successor registry escaped configs/")
    if not receipt_path.is_relative_to((root / "evidence").resolve()):
        raise ValueError("teacher-canary receipt escaped evidence/")
    if verification_path.exists():
        raise FileExistsError("teacher-canary verification already exists")

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
    predecessor_sha256 = file_sha256(predecessor_path)
    contract_sha256 = file_sha256(contract_path)
    candidate_bytes = jsonl_file_bytes(canonical)
    registry_bytes = json_file_bytes(successor)
    receipt_bytes = json_file_bytes(receipt)
    if len(candidate_bytes) != CANONICAL_CANDIDATE_BYTES:
        raise ValueError("teacher-canary canonical candidate byte limit changed")
    if len(registry_bytes) != SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("teacher-canary successor registry byte limit changed")

    _materialize_transaction(
        [
            (candidate_path, candidate_bytes),
            (registry_path, registry_bytes),
            (receipt_path, receipt_bytes),
        ]
    )
    if (
        file_sha256(candidate_path) != CANONICAL_CANDIDATE_SHA256
        or candidate_path.stat().st_size != CANONICAL_CANDIDATE_BYTES
        or file_sha256(registry_path) != SUCCESSOR_REGISTRY_SHA256
        or registry_path.stat().st_size != SUCCESSOR_REGISTRY_BYTES
    ):
        raise ValueError("teacher-canary materialized output identity changed")
    if file_sha256(predecessor_path) != predecessor_sha256:
        raise ValueError("teacher-canary materialization mutated registry v0.14")
    if file_sha256(contract_path) != contract_sha256:
        raise ValueError("teacher-canary materialization mutated the data contract")
    print(
        "PASS: teacher-canary canonical candidate and registry v0.15 materialized; "
        f"candidate={CANONICAL_CANDIDATE_SHA256}; "
        f"registry={SUCCESSOR_REGISTRY_SHA256}; "
        f"receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
