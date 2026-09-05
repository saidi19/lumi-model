#!/usr/bin/env python3
"""Reconcile an existing Aya audit against official count statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lumi.posttraining_aya_reconciliation import (
    REPORT_SCHEMA_VERSION,
    build_aya_audit_reconciliation_report,
)
from lumi.posttraining_source_quality import file_sha256


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Aya reconciliation: {output}")
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "lumi.sft-aya-audit-reconciliation-plan.v0.1":
        raise ValueError("unsupported Aya reconciliation plan")
    if plan.get("authorization") != {
        "dataset_promotion_authorized": False,
        "external_api_spend_authorized": False,
        "network_access_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
        "purpose": "existing_aya_audit_retrieval_reconciliation_only",
    }:
        raise ValueError("Aya reconciliation authorization changed")
    if plan.get("output") != {
        "production_promotion_permitted": False,
        "raw_text_retained": False,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "row_level_human_review_claimed": False,
    }:
        raise ValueError("Aya reconciliation output boundary changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "implementation",
        "original_evidence",
        "runner",
    }:
        raise ValueError("Aya reconciliation repository bindings changed")
    resolved_bindings: dict[str, str] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise ValueError(f"Aya reconciliation binding is invalid: {name}")
        path = (root / binding["path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"Aya reconciliation binding escapes repository: {name}")
        digest = file_sha256(path)
        if digest != binding.get("sha256"):
            raise ValueError(f"Aya reconciliation binding changed: {name}")
        resolved_bindings[f"{name}_file_sha256"] = digest

    attempt = args.artifact_attempt.resolve()
    if root == attempt or root in attempt.parents:
        raise ValueError("Aya reconciliation raw artifact must remain outside Git")
    artifact_bindings = plan.get("artifact_bindings")
    if not isinstance(artifact_bindings, dict):
        raise ValueError("Aya reconciliation artifact bindings are missing")
    inventory_path = attempt / "inventory.json"
    original_artifact_report_path = attempt / "audit-report.json"
    statistics_path = attempt / "aya-statistics.json"
    artifact_files = {
        "artifact_inventory": inventory_path,
        "artifact_original_report": original_artifact_report_path,
        "artifact_statistics": statistics_path,
    }
    for name, path in artifact_files.items():
        digest = file_sha256(path)
        if digest != artifact_bindings.get(f"{name}_sha256"):
            raise ValueError(f"Aya reconciliation artifact changed: {name}")
        resolved_bindings[f"{name}_sha256"] = digest

    original_report = _load_json(root / bindings["original_evidence"]["path"])
    if original_report != _load_json(original_artifact_report_path):
        raise ValueError("Aya repository and artifact reports differ")
    if original_report.get("report_sha256") != plan.get("original_report_sha256"):
        raise ValueError("Aya original internal report binding changed")
    inventory = _load_json(inventory_path)
    page_inventory = inventory.get("page_inventory")
    if (
        not isinstance(page_inventory, list)
        or len(page_inventory) != plan.get("expected_page_count")
        or inventory.get("rows") != plan.get("expected_fetched_rows")
    ):
        raise ValueError("Aya reconciliation page inventory boundary changed")

    wrapped_rows: list[dict[str, Any]] = []
    raw_directory = attempt / "raw"
    for page in page_inventory:
        if not isinstance(page, dict) or not isinstance(page.get("filename"), str):
            raise ValueError("Aya reconciliation page binding is invalid")
        page_path = raw_directory / page["filename"]
        if file_sha256(page_path) != page.get("sha256"):
            raise ValueError("Aya reconciliation page hash changed")
        if page_path.stat().st_size != page.get("size_bytes"):
            raise ValueError("Aya reconciliation page size changed")
        payload = _load_json(page_path)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != page.get("rows"):
            raise ValueError("Aya reconciliation page row count changed")
        wrapped_rows.extend(rows)
    if len(wrapped_rows) != plan.get("expected_fetched_rows"):
        raise ValueError("Aya reconciliation fetched row count changed")

    resolved_bindings["plan_file_sha256"] = file_sha256(plan_path)
    report = build_aya_audit_reconciliation_report(
        bindings=resolved_bindings,
        observed_at=plan["observed_at"],
        original_report=original_report,
        statistics=_load_json(statistics_path),
        wrapped_rows=wrapped_rows,
    )
    _write_json_atomic(output, report)
    print(
        "PASS: Aya audit retrieval reconciliation sealed; "
        f"rows={report['retrieval_integrity']['fetched_rows']} "
        f"blockers={len(report['remaining_production_blockers'])} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
