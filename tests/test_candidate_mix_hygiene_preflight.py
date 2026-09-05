from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_hygiene_preflight import (
    EXPECTED_LIMITS,
    build_hygiene_preflight_verification,
    load_hygiene_preflight_plan,
    rebuild_hygiene_preflight,
    validate_hygiene_preflight_report,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-v0.1.json"
VERIFICATION = (
    ROOT
    / "evidence/lumi-4b-sft-candidate-mix-hygiene-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("candidate hygiene preflight plan is not materialized")
    return load_hygiene_preflight_plan(PLAN, repository_root=ROOT)


def test_hygiene_preflight_plan_is_cpu_only_and_non_mutating() -> None:
    plan = _load_plan()
    assert plan["limits"] == EXPECTED_LIMITS
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert plan["authorization"]["candidate_materialization_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_hygiene_preflight_report_is_exact_and_raw_text_free_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("candidate hygiene preflight report is not materialized")
    report = _load(REPORT)
    validate_hygiene_preflight_report(report)
    assert report["projected_pool"] == {
        "by_split": {
            "train": {"records": 2_962, "supervised_tokens": 966_069},
            "validation": {"records": 238, "supervised_tokens": 79_056},
        },
        "records": 3_200,
        "removed_records": 23,
    }
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        assert forbidden not in serialized


def test_hygiene_preflight_verification_replays_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("candidate hygiene preflight verification is not materialized")
    report = _load(REPORT)
    replayed = rebuild_hygiene_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    expected = build_hygiene_preflight_verification(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
        replayed_report=replayed,
    )
    assert expected == _load(VERIFICATION)

