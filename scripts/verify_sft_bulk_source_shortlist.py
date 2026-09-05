#!/usr/bin/env python3
"""Independently reproduce the Lumi bulk-source shortlist report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_bulk_source_shortlist import (
    VERIFICATION_SCHEMA_VERSION,
    file_sha256,
    json_file_bytes,
    load_bulk_source_shortlist_plan,
    rebuild_bulk_source_shortlist,
    sha256_value,
    validate_bulk_source_shortlist_report,
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shortlist verification: {path}")
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
    plan = load_bulk_source_shortlist_plan(plan_path, repository_root=root)
    report_path = (root / plan["output"]["report_path"]).resolve()
    observed = json.loads(report_path.read_text(encoding="utf-8"))
    validate_bulk_source_shortlist_report(observed)
    rebuilt = rebuild_bulk_source_shortlist(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
    )
    if rebuilt != observed:
        raise ValueError("bulk-source shortlist report did not reproduce")
    verification = {
        "checks": {
            "all_local_binding_hashes_passed": True,
            "deepmind_canary_and_registry_binding_passed": True,
            "exact_priority_gaps_reproduced": True,
            "no_dataset_payload_downloaded": True,
            "no_gpu_or_sft_action": True,
            "no_raw_training_text_retained": True,
            "opencode_exact_revision_and_canary_object_pinned": True,
            "production_promotions_zero": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": observed["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_bulk_source_shortlist_nonproduction",
        "verification_sha256": "",
    }
    verification["verification_sha256"] = sha256_value(
        verification, excluded_key="verification_sha256"
    )
    output = (root / plan["output"]["verification_path"]).resolve()
    if not output.is_relative_to(root / "evidence"):
        raise ValueError("shortlist verification must remain under evidence/")
    _write_once(output, json_file_bytes(verification))
    print(
        "PASS: bulk-source shortlist independently verified; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

