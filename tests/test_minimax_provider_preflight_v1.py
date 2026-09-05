from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_minimax_provider_preflight_v1 import (
    build_minimax_provider_preflight_report_v1,
    build_minimax_provider_preflight_verification_v1,
    load_minimax_provider_preflight_plan_v1,
    validate_minimax_provider_preflight_report_v1,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-minimax-provider-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-minimax-provider-preflight-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-minimax-provider-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("MiniMax provider-preflight plan is not materialized")
    return load_minimax_provider_preflight_plan_v1(PLAN, repository_root=ROOT)


def test_plan_authorizes_no_provider_call_or_spend() -> None:
    plan = _load_plan()
    assert plan["authorization"]["external_api_call_authorized"] is False
    assert plan["authorization"]["external_api_spend_authorized"] is False
    assert plan["limits"]["external_api_calls"] == 0
    assert plan["next_gate"]["ollama_cloud_route_permitted"] is False
    assert plan["next_gate"]["maximum_candidate_records"] == 100


def test_report_reproduces_and_fails_ollama_cloud_closed_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("MiniMax provider-preflight report is not materialized")
    observed = _load(REPORT)
    rebuilt = build_minimax_provider_preflight_report_v1(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == observed
    validate_minimax_provider_preflight_report_v1(observed)
    assert observed["decisions"]["ollama_cloud_minimax"].startswith("fail_closed")
    assert observed["decisions"]["minimax_direct_api"].startswith(
        "conditionally_eligible"
    )
    assert observed["local_environment"]["secret_values_logged"] is False


def test_verification_is_exact_and_raw_text_free_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("MiniMax provider-preflight verification is not materialized")
    report = _load(REPORT)
    expected = build_minimax_provider_preflight_verification_v1(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    serialized = REPORT.read_text(encoding="utf-8") + VERIFICATION.read_text(
        encoding="utf-8"
    )
    for forbidden in ('"messages": [', '"raw_candidate"', '"raw_text"'):
        assert forbidden not in serialized
