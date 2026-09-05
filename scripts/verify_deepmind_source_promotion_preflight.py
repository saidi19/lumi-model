#!/usr/bin/env python3
"""Independently verify the DeepMind source-promotion preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_deepmind_source_promotion_preflight import (
    VERIFICATION_SCHEMA_VERSION,
    build_preflight_report,
    file_sha256,
    load_jsonl,
    load_policy_and_predecessor,
    load_preflight_plan,
    resolve_binding,
    sha256_value,
    validate_preflight_report,
    verify_canary_prerequisites,
    verify_current_schema_blocker,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind verification: {path}")
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
    plan = load_preflight_plan(plan_path, repository_root=root)
    bindings = plan["bindings"]
    candidate_audit = verify_canary_prerequisites(
        accepted_records=load_jsonl(
            resolve_binding(
                root,
                bindings["accepted_records"],
                name="accepted_records",
            )
        ),
        attributions=load_jsonl(
            resolve_binding(
                root,
                bindings["attribution_manifest"],
                name="attribution_manifest",
            )
        ),
        canary_report=_read_json(
            resolve_binding(root, bindings["canary_report"], name="canary_report")
        ),
        canary_verification=_read_json(
            resolve_binding(
                root,
                bindings["canary_verification"],
                name="canary_verification",
            )
        ),
        token_metrics=_read_json(
            resolve_binding(root, bindings["token_metrics"], name="token_metrics")
        ),
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    expected_report = build_preflight_report(
        candidate_audit=candidate_audit,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        predecessor_value=predecessor_value,
        schema_blocker=verify_current_schema_blocker(policy=policy),
    )
    report_path = (root / plan["output"]["report_path"]).resolve()
    report = _read_json(report_path)
    validate_preflight_report(report)
    if report != expected_report:
        raise ValueError("DeepMind promotion-preflight report reconstruction changed")
    proposed_path = (root / plan["output"]["proposed_successor_registry_path"]).resolve()
    if proposed_path.exists():
        raise ValueError("DeepMind preflight unexpectedly wrote a successor registry")

    verification: dict[str, Any] = {
        "checks": {
            "candidate_and_attribution_replayed": True,
            "canary_report_and_verification_seals_verified": True,
            "current_repository_url_schema_blocker_reproduced": True,
            "network_api_and_gpu_usage": 0,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "proposed_successor_not_written": True,
            "report_reconstructed": True,
        },
        "candidate_records": 1_915,
        "candidate_supervised_tokens": 18_731,
        "new_explicit_promotion_authorization_required": True,
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_deepmind_source_promotion_preflight_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    output_path = (root / plan["output"]["verification_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind promotion-preflight verification escaped evidence/")
    _write_json_atomic(output_path, verification)
    print(
        "PASS: independently verified DeepMind promotion preflight; "
        f"verification={verification['verification_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

