#!/usr/bin/env python3
"""Run the non-mutating DeepMind candidate-registry promotion preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lumi.posttraining_deepmind_source_promotion_preflight import (
    build_preflight_report,
    file_sha256,
    json_file_bytes,
    load_jsonl,
    load_policy_and_predecessor,
    load_preflight_plan,
    resolve_binding,
    validate_preflight_report,
    verify_canary_prerequisites,
    verify_current_schema_blocker,
)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DeepMind preflight output: {path}")
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
    bindings = plan["bindings"]
    accepted_records = load_jsonl(
        resolve_binding(root, bindings["accepted_records"], name="accepted_records")
    )
    attributions = load_jsonl(
        resolve_binding(
            root,
            bindings["attribution_manifest"],
            name="attribution_manifest",
        )
    )
    canary_report = json.loads(
        resolve_binding(root, bindings["canary_report"], name="canary_report")
        .read_text(encoding="utf-8")
    )
    canary_verification = json.loads(
        resolve_binding(
            root,
            bindings["canary_verification"],
            name="canary_verification",
        ).read_text(encoding="utf-8")
    )
    token_metrics = json.loads(
        resolve_binding(root, bindings["token_metrics"], name="token_metrics")
        .read_text(encoding="utf-8")
    )
    candidate_audit = verify_canary_prerequisites(
        accepted_records=accepted_records,
        attributions=attributions,
        canary_report=canary_report,
        canary_verification=canary_verification,
        token_metrics=token_metrics,
    )
    policy, predecessor, predecessor_value = load_policy_and_predecessor(
        plan=plan,
        repository_root=root,
    )
    schema_blocker = verify_current_schema_blocker(policy=policy)
    report = build_preflight_report(
        candidate_audit=candidate_audit,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        predecessor=predecessor,
        predecessor_value=predecessor_value,
        schema_blocker=schema_blocker,
    )
    validate_preflight_report(report)
    output_path = (root / plan["output"]["report_path"]).resolve()
    if not output_path.is_relative_to(root / "evidence"):
        raise ValueError("DeepMind promotion-preflight report must stay under evidence/")
    _write_bytes_atomic(output_path, json_file_bytes(report))
    print(
        "PASS: DeepMind source-promotion preflight verified without registry mutation; "
        f"candidate={plan['candidate']['accepted_records']}; "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
