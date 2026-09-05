from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_deepmind_canary_gate import (
    audit_runtime_dependencies,
    build_blocked_gate_report,
    load_canary_gate_plan,
    validate_blocked_gate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-canary-gate-v0.1.json"
REPORT_PATH = ROOT / "evidence/lumi-4b-sft-deepmind-canary-gate-v0.1.json"


def _find_spec(module: str) -> object | None:
    return None if module in {"absl", "six"} else object()


def _version(distribution: str) -> str:
    return {"numpy": "2.5.1", "sympy": "1.14.0"}[distribution]


def test_dependency_audit_records_missing_modules_without_importing() -> None:
    audit = audit_runtime_dependencies(
        find_spec=_find_spec,
        distribution_version=_version,
        executable="/isolated/python",
        python_version="3.13.7",
    )
    assert audit["all_required_modules_available"] is False
    assert audit["missing_modules"] == ["absl", "six"]
    assert [item["version"] for item in audit["modules"]] == [
        None,
        "2.5.1",
        None,
        "1.14.0",
    ]


def test_blocked_report_proves_zero_network_and_generation() -> None:
    audit = audit_runtime_dependencies(
        find_spec=_find_spec,
        distribution_version=_version,
        executable="/isolated/python",
        python_version="3.13.7",
    )
    plan = {
        "observed_at": "2026-08-06",
        "source": {
            "declared_license_id": "apache-2.0",
            "exact_object_bytes": 234672,
            "exact_object_count": 30,
            "exact_object_inventory_sha256": "a" * 64,
            "repository_url": "https://example.invalid",
            "revision": "b" * 40,
            "source_id": "deepmind_mathematics_dataset",
        },
    }
    report = build_blocked_gate_report(
        dependency_audit=audit,
        plan=plan,
        plan_file_sha256="c" * 64,
    )
    validate_blocked_gate_report(report)
    assert report["execution"]["network_bytes"] == 0
    assert report["execution"]["generated_records"] == 0


def test_blocked_report_rejects_available_runtime() -> None:
    audit = audit_runtime_dependencies(
        find_spec=lambda _module: object(),
        distribution_version=lambda _distribution: "1.0",
    )
    with pytest.raises(ValueError, match="missing dependency"):
        build_blocked_gate_report(
            dependency_audit=audit,
            plan={"observed_at": "2026-08-06", "source": {}},
            plan_file_sha256="d" * 64,
        )


def test_exact_plan_and_report_load_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind canary-gate plan is not materialized")
    plan = load_canary_gate_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["limits"]["maximum_canary_generated_records"] == 2000
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text())
        validate_blocked_gate_report(report)
        assert report["execution"]["source_acquisition_attempted"] is False
