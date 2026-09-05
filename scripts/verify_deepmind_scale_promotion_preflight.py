#!/usr/bin/env python3
"""Independently rebuild and verify the DeepMind scale promotion preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_deepmind_scale_generation import file_sha256, sha256_value
from lumi.posttraining_deepmind_scale_promotion_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    build_preflight_report,
    json_file_bytes,
    load_preflight_plan,
    validate_preflight_report,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind verification: {path}")
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
    plan = load_preflight_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    observed = _load(report_path)
    validate_preflight_report(observed)
    rebuilt = build_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if rebuilt != observed:
        raise ValueError("DeepMind scale preflight report does not reproduce")
    receipt = {
        "checks": {
            "bound_artifacts_reverified": True,
            "predecessor_registry_unchanged": True,
            "projection_reproduced": True,
            "proposed_successor_hash_reproduced": True,
            "raw_training_text_retained": False,
            "registry_promotions": 0,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "proposed_successor_file_sha256": observed["proposed_successor"][
            "file_sha256"
        ],
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": observed["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_deepmind_scale_promotion_preflight_nonproduction",
    }
    receipt["verification_sha256"] = sha256_value(receipt)
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind scale preflight verification escaped evidence/")
    _write_atomic(output, json_file_bytes(receipt))
    print(
        "PASS: DeepMind scale promotion preflight independently reproduced; "
        f"verification={receipt['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
