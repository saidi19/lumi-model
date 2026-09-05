from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_hygiene_registry_preflight_v2 import (
    EXPECTED_LIMITS,
    EXPECTED_SUCCESSOR,
    build_hygiene_registry_v14_preflight_verification,
    load_hygiene_registry_v14_preflight_plan,
    rebuild_hygiene_registry_v14_preflight,
    validate_hygiene_registry_v14_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-hygiene-registry-v0.14-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-hygiene-registry-v0.14-preflight-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-hygiene-registry-v0.14-preflight-verification-v0.1.json"
PROJECTED_REGISTRY = ROOT / "configs/lumi-4b-sft-source-registry-v0.14.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("registry-v0.14 preflight plan is not materialized")
    return load_hygiene_registry_v14_preflight_plan(PLAN, repository_root=ROOT)


def test_registry_v14_preflight_plan_is_non_mutating_and_cpu_only() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["source_registry_mutation_authorized"] is False
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_registry_v14_preflight_report_is_exact_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("registry-v0.14 preflight report is not materialized")
    report = _load(REPORT)
    validate_hygiene_registry_v14_preflight_report(report)
    assert report["changed_sources_count"] == 2
    assert report["projected_successor"] == {
        **EXPECTED_SUCCESSOR,
        "materialized": False,
        "path": "configs/lumi-4b-sft-source-registry-v0.14.json",
    }
    assert not PROJECTED_REGISTRY.exists()


def test_registry_v14_preflight_verification_replays_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("registry-v0.14 verification is not materialized")
    report = _load(REPORT)
    replayed = rebuild_hygiene_registry_v14_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    expected = build_hygiene_registry_v14_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
        replayed_report=replayed,
    )
    assert expected == _load(VERIFICATION)
