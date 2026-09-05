#!/usr/bin/env python3
"""Re-audit the repaired OASST1 adapter against an existing pinned raw slice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_canary import adapt_oasst1_rows
from lumi.posttraining_data import filter_sft_records, load_sft_policy
from lumi.posttraining_source_quality import (
    build_oasst1_repair_audit_report,
    file_sha256,
)
from lumi.posttraining_sources import load_sft_source_registry


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(repository_root: Path, relative_path: str) -> Path:
    return (repository_root / relative_path).resolve()


def _verify_binding(repository_root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(repository_root, str(binding["path"]))
    if file_sha256(path) != binding["sha256"]:
        raise ValueError(f"OASST1 repair binding changed: {binding['path']}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OASST1 repair audit: {output}")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "lumi.sft-oasst1-repair-reaudit-plan.v0.1":
        raise ValueError("unsupported OASST1 repair re-audit plan")
    expected_authorization = {
        "bulk_data_download_authorized": False,
        "dataset_promotion_authorized": False,
        "external_api_spend_authorized": False,
        "network_access_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
        "purpose": "existing_pinned_oasst1_slice_structural_reaudit_only",
    }
    if plan.get("authorization") != expected_authorization:
        raise ValueError("OASST1 repair authorization boundary changed")

    bindings = plan["bindings"]
    paths = {
        name: _verify_binding(root, binding)
        for name, binding in bindings.items()
    }
    metadata_pre = _load_json(paths["metadata_pre"])
    metadata_post = _load_json(paths["metadata_post"])
    source = plan["source"]
    for metadata in (metadata_pre, metadata_post):
        if metadata.get("id") != source["dataset_id"]:
            raise ValueError("OASST1 metadata dataset identity changed")
        if metadata.get("sha") != source["revision"]:
            raise ValueError("OASST1 metadata source revision changed")
    if metadata_pre != metadata_post:
        raise ValueError("OASST1 source head changed during original retrieval")

    raw_payload = _load_json(paths["raw_rows"])
    rows = raw_payload.get("rows")
    if not isinstance(rows, list) or len(rows) > plan["limits"]["maximum_fetched_rows"]:
        raise ValueError("OASST1 raw row boundary changed")
    candidates = adapt_oasst1_rows(
        rows,
        revision=source["revision"],
        target_records=plan["limits"]["target_records"],
    )

    policy = load_sft_policy(paths["sft_contract"])
    registry = load_sft_source_registry(
        paths["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    accepted, filter_report = filter_sft_records(
        candidates,
        policy=policy,
        benchmark_registry=BenchmarkRegistry(
            policy.benchmark_decontamination,
            root,
        ),
        source_registry=registry,
    )
    report = build_oasst1_repair_audit_report(
        accepted=accepted,
        adapter_implementation_sha256=bindings["adapter_implementation"]["sha256"],
        filter_report=filter_report,
        metadata_sha256=bindings["metadata_pre"]["sha256"],
        observed_at=plan["observed_at"],
        raw_payload=raw_payload,
        raw_rows_sha256=bindings["raw_rows"]["sha256"],
        source_revision=source["revision"],
        target_records=plan["limits"]["target_records"],
    )
    report["bindings"]["plan_file_sha256"] = file_sha256(plan_path)
    report["bindings"]["sft_contract_sha256"] = bindings["sft_contract"]["sha256"]
    report["bindings"]["source_registry_sha256"] = bindings["source_registry"][
        "sha256"
    ]
    report_without_seal = {key: value for key, value in report.items() if key != "report_sha256"}
    from lumi.posttraining_source_quality import sha256_value

    report["report_sha256"] = sha256_value(report_without_seal)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(
        "PASS: OASST1 structural repair re-audit; "
        f"accepted={report['accepted_candidate_records']} "
        f"remaining_blockers={len(report['remaining_production_blockers'])} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
