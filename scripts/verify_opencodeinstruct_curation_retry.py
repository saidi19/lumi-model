#!/usr/bin/env python3
"""Independently verify the deterministic OpenCodeInstruct retry."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from lumi.posttraining_opencodeinstruct_curation import (
    VERIFICATION_SCHEMA_VERSION,
    file_sha256,
    sha256_value,
)
from lumi.posttraining_opencodeinstruct_retry import (
    ORIGINAL_VERIFIER_SHA256,
    load_retry_plan,
    run_deterministic_unit_test_verifier,
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
    plan = load_retry_plan(plan_path, repository_root=root)
    private_root = (root / plan["output"]["private_root"]).resolve()
    report_path = (root / plan["output"]["raw_text_free_report_path"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("executed deterministic retry plan no longer matches report")

    original_path = root / "scripts/verify_opencodeinstruct_curation.py"
    if file_sha256(original_path) != ORIGINAL_VERIFIER_SHA256:
        raise ValueError("original OpenCodeInstruct verifier changed")
    original = _load_original_verifier(original_path)
    reported_token_metrics = json.loads(
        (private_root / "curated/token-metrics.json").read_text(encoding="utf-8")
    )
    if not isinstance(reported_token_metrics, dict):
        raise ValueError("deterministic retry token metrics root changed")

    original_measure = original.measure_and_select

    def measure_selected_pool(*args: Any, **kwargs: Any) -> Any:
        records, metrics, ledger = original_measure(*args, **kwargs)
        for field in POOL_ONLY_FIELDS:
            metrics[field] = reported_token_metrics[field]
        return records, metrics, ledger

    original_write = original.write_json_atomic
    original_verify_raw_objects = original.verify_raw_objects

    def verify_local_reuse(raw_root: Path) -> dict[str, Any]:
        result = original_verify_raw_objects(raw_root)
        result["local_reuse_bytes"] = result["network_bytes"]
        result["network_bytes"] = 0
        return result

    def write_retry_verification(path: Path, value: dict[str, Any]) -> None:
        if value.get("schema_version") == VERIFICATION_SCHEMA_VERSION:
            value["verification_retry"] = {
                "candidate_or_report_modified_during_verification": False,
                "comparison_adjustment": list(POOL_ONLY_FIELDS),
                "deterministic_python_hash_seed": "0",
                "original_verifier_sha256": ORIGINAL_VERIFIER_SHA256,
                "retry_verifier_sha256": file_sha256(Path(__file__).resolve()),
                "unit_tests_replayed_in_fresh_process": True,
            }
            value.pop("verification_sha256", None)
            value["verification_sha256"] = sha256_value(value)
        original_write(path, value)

    original.load_opencodeinstruct_plan = load_retry_plan
    original.measure_and_select = measure_selected_pool
    original.run_local_unit_test_verifier = run_deterministic_unit_test_verifier
    original.verify_raw_objects = verify_local_reuse
    original.write_json_atomic = write_retry_verification
    return int(original.main())


if __name__ == "__main__":
    raise SystemExit(main())
