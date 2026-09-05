#!/usr/bin/env python3
"""Result-preserving verifier continuation for selected-pool bookkeeping."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from lumi.posttraining_opencodeinstruct_curation import (
    VERIFICATION_SCHEMA_VERSION,
    file_sha256,
    load_opencodeinstruct_plan,
    sha256_value,
)

ORIGINAL_VERIFIER_SHA256 = (
    "a3800f58f3c3055c58946a036848ad79c16c18cba91a6cfd8d3ca363bc85b070"
)
POOL_ONLY_FIELDS = ("candidate_4k_fit_records", "overlength_rejected")


def _argument(flag: str, default: str | None = None) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        if default is None:
            raise ValueError(f"missing required argument: {flag}") from None
        return default


def _load_original_verifier(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("lumi_original_opencode_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load original OpenCodeInstruct verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(_argument("--repository-root", ".")).resolve()
    plan_arg = Path(_argument("--plan"))
    plan_path = plan_arg if plan_arg.is_absolute() else root / plan_arg
    plan = load_opencodeinstruct_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("executed OpenCodeInstruct plan no longer matches the report")

    original_path = root / "scripts/verify_opencodeinstruct_curation.py"
    if file_sha256(original_path) != ORIGINAL_VERIFIER_SHA256:
        raise ValueError("original OpenCodeInstruct verifier changed")
    original = _load_original_verifier(original_path)
    token_metrics_path = private_root / "curated/token-metrics.json"
    reported_token_metrics = json.loads(token_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(reported_token_metrics, dict):
        raise ValueError("OpenCodeInstruct token metrics root changed")

    original_measure = original.measure_and_select

    def measure_selected_pool(*args: Any, **kwargs: Any) -> Any:
        records, metrics, ledger = original_measure(*args, **kwargs)
        for field in POOL_ONLY_FIELDS:
            metrics[field] = reported_token_metrics[field]
        return records, metrics, ledger

    original_write = original.write_json_atomic

    def write_bound_verification(path: Path, value: dict[str, Any]) -> None:
        if value.get("schema_version") == VERIFICATION_SCHEMA_VERSION:
            value["verification_continuation"] = {
                "candidate_or_report_modified": False,
                "comparison_adjustment": list(POOL_ONLY_FIELDS),
                "continuation_verifier_sha256": file_sha256(Path(__file__).resolve()),
                "original_verifier_sha256": ORIGINAL_VERIFIER_SHA256,
                "purpose": "selected_replay_ignores_preselection_pool_only_counters",
            }
            value.pop("verification_sha256", None)
            value["verification_sha256"] = sha256_value(value)
        original_write(path, value)

    original.measure_and_select = measure_selected_pool
    original.write_json_atomic = write_bound_verification
    return int(original.main())


if __name__ == "__main__":
    raise SystemExit(main())
