from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_first_party_scale_canary_preflight import (
    build_first_party_scale_canary_preflight_report,
    build_first_party_scale_canary_preflight_verification,
    load_first_party_scale_canary_preflight_plan,
    validate_first_party_scale_canary_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-first-party-scale-canary-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-first-party-scale-canary-preflight-v0.1.json"
VERIFICATION = ROOT / (
    "evidence/lumi-4b-sft-first-party-scale-canary-preflight-verification-v0.1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("scale-canary preflight plan is not materialized")
    raw = _load(PLAN)
    missing = []
    for name, binding in raw.get("bindings", {}).items():
        path = Path(binding["path"])
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.is_file():
            missing.append(name)
    if missing:
        pytest.skip(f"scale-canary private bindings are unavailable: {missing}")
    return load_first_party_scale_canary_preflight_plan(PLAN, repository_root=ROOT)


def test_preflight_is_no_call_and_nonproduction() -> None:
    plan = _plan()
    assert plan["limits"]["candidate_records_generated"] == 0
    assert plan["limits"]["external_api_calls"] == 0
    assert plan["authorization"]["external_api_authorized"] is False
    assert plan["authorization"]["local_ollama_generation_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False


def test_contract_is_exactly_1000_rows_with_800_200_split() -> None:
    plan = _plan()
    contract = plan["canary_contract"]
    assert contract["candidate_records"] == 1_000
    assert contract["train_candidate_slots"] == 800
    assert contract["validation_candidate_slots"] == 200
    assert contract["maximum_retained_train_supervised_tokens"] == 120_000


def test_report_and_verification_reproduce_when_materialized() -> None:
    plan = _plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("scale-canary preflight evidence is not materialized")
    observed = _load(REPORT)
    rebuilt = build_first_party_scale_canary_preflight_report(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == observed
    validate_first_party_scale_canary_preflight_report(observed)
    expected = build_first_party_scale_canary_preflight_verification(
        plan=plan,
        report=observed,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    serialized = REPORT.read_text(encoding="utf-8") + VERIFICATION.read_text(
        encoding="utf-8"
    )
    for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
        assert forbidden not in serialized
