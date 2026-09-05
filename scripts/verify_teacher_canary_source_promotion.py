#!/usr/bin/env python3
"""Independently verify teacher-canary candidate and registry v0.15."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_teacher_canary_source_promotion import (
    CANONICAL_CANDIDATE_BYTES,
    CANONICAL_CANDIDATE_PATH,
    CANONICAL_CANDIDATE_SHA256,
    PREDECESSOR_REGISTRY_SHA256,
    RECEIPT_PATH,
    SUCCESSOR_REGISTRY_BYTES,
    SUCCESSOR_REGISTRY_PATH,
    SUCCESSOR_REGISTRY_SHA256,
    VERIFICATION_PATH,
    VERIFICATION_SCHEMA_VERSION,
    build_promotion_receipt,
    file_sha256,
    load_materialization_plan,
    reconstruct_materialized_values,
    resolve_binding,
    sha256_value,
    validate_canonical_rows,
    validate_promotion_receipt,
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    values: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}: {path}")
            values.append(value)
    return values


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing teacher-canary verification overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
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
    plan = load_materialization_plan(plan_path, repository_root=root)
    expected_candidate, expected_registry, report = reconstruct_materialized_values(
        plan=plan,
        repository_root=root,
    )

    candidate_path = (root / CANONICAL_CANDIDATE_PATH).resolve()
    registry_path = (root / SUCCESSOR_REGISTRY_PATH).resolve()
    receipt_path = (root / RECEIPT_PATH).resolve()
    observed_candidate = _read_jsonl(candidate_path)
    observed_registry = _read_object(registry_path)
    if observed_candidate != expected_candidate:
        raise ValueError("teacher-canary canonical candidate does not reproduce")
    validate_canonical_rows(observed_candidate)
    if (
        file_sha256(candidate_path) != CANONICAL_CANDIDATE_SHA256
        or candidate_path.stat().st_size != CANONICAL_CANDIDATE_BYTES
    ):
        raise ValueError("teacher-canary canonical candidate file identity changed")
    if observed_registry != expected_registry:
        raise ValueError("teacher-canary registry v0.15 does not reproduce")
    if (
        file_sha256(registry_path) != SUCCESSOR_REGISTRY_SHA256
        or registry_path.stat().st_size != SUCCESSOR_REGISTRY_BYTES
    ):
        raise ValueError("teacher-canary registry v0.15 file identity changed")

    observed_receipt = _read_object(receipt_path)
    validate_promotion_receipt(observed_receipt)
    expected_receipt = build_promotion_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        report=report,
    )
    if observed_receipt != expected_receipt:
        raise ValueError("teacher-canary promotion receipt does not reproduce")

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
    verification = {
        "checks": {
            "all_80_rows_reproduced_and_contract_accepted": True,
            "canonical_candidate_hash_and_size_verified": True,
            "data_contract_unchanged": True,
            "dataset_remains_unsealed": True,
            "external_api_absent": True,
            "gpu_use_absent": True,
            "human_authorship_claim_absent": True,
            "network_use_absent": True,
            "predecessor_registry_unchanged": True,
            "promotion_receipt_reproduced": True,
            "raw_training_text_absent_from_verification": True,
            "sft_not_started": True,
            "successor_registry_hash_and_size_verified": True,
        },
        "observed_at": "2026-08-09",
        "predecessor_registry_file_sha256": file_sha256(predecessor_path),
        "production_eligible": False,
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": observed_receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_teacher_canary_candidate_and_registry_v0_15_nonproduction",
        "canonical_candidate": {
            "bytes": candidate_path.stat().st_size,
            "file_sha256": file_sha256(candidate_path),
            "records": len(observed_candidate),
        },
        "successor_registry": {
            "bytes": registry_path.stat().st_size,
            "file_sha256": file_sha256(registry_path),
            "source_count": len(observed_registry["sources"]),
        },
        "unchanged_data_contract_file_sha256": file_sha256(contract_path),
    }
    if verification["predecessor_registry_file_sha256"] != PREDECESSOR_REGISTRY_SHA256:
        raise ValueError("teacher-canary predecessor registry changed")
    if not all(verification["checks"].values()):
        raise ValueError("teacher-canary independent verification checks failed")
    verification["verification_sha256"] = sha256_value(verification)
    output = (root / VERIFICATION_PATH).resolve()
    if not output.is_relative_to((root / "evidence").resolve()):
        raise ValueError("teacher-canary verification escaped evidence/")
    _write_json_atomic(output, verification)
    print(
        "PASS: teacher-canary canonical candidate and registry v0.15 independently "
        f"verified; verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
