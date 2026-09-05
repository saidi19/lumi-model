from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_mix_closure_source_selection_v4 import (
    EXPECTED_PROJECTED_FIRST_PARTY_TOKENS,
    EXPECTED_PROJECTED_TOKENS,
    EXPECTED_TARGET_TOKENS,
    RETIRED_CANDIDATE_SHA256,
    build_mix_closure_source_selection_verification_v4,
    load_mix_closure_source_selection_plan_v4,
    rebuild_mix_closure_source_selection_v4,
    validate_mix_closure_source_selection_report_v4,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-10m-mix-closure-source-selection-v0.4.json"
REPORT = ROOT / "evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.4.json"
VERIFICATION = (
    ROOT
    / "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.4.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("v0.4 closure plan is not materialized")
    return load_mix_closure_source_selection_plan_v4(PLAN, repository_root=ROOT)


def test_v0_17_plan_is_local_only_and_provider_non_authorizing() -> None:
    plan = _load_plan()
    assert plan["limits"]["maximum_network_bytes"] == 0
    assert plan["authorization"]["generation_authorized"] is False
    assert plan["authorization"]["external_api_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert plan["next_gate"]["minimax_cloud_use_authorized"] is False
    assert plan["next_gate"]["maximum_candidate_records"] == 100


def test_v0_17_report_reproduces_and_closes_floors_and_caps_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file():
        pytest.skip("v0.4 closure report is not materialized")
    observed = _load(REPORT)
    rebuilt = rebuild_mix_closure_source_selection_v4(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    assert rebuilt == observed
    validate_mix_closure_source_selection_report_v4(observed)
    projected = observed["projected_mix"]
    assert projected["added_train_supervised_tokens"] == EXPECTED_TARGET_TOKENS
    assert projected["train_supervised_tokens"] == EXPECTED_PROJECTED_TOKENS
    assert (
        projected["source_family_train_supervised_tokens"][
            "lumi_first_party_family"
        ]
        == EXPECTED_PROJECTED_FIRST_PARTY_TOKENS
    )
    assert projected["all_task_floors_met"] is True
    assert projected["all_language_floors_met"] is True
    assert projected["all_source_caps_met"] is True
    assert projected["all_source_family_caps_met"] is True


def test_v0_17_retired_candidate_is_excluded() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("v0.4 closure report is not materialized")
    report = _load(REPORT)
    retirement = report["legacy_closure_retirement"]
    assert retirement["candidate_sha256"] == RETIRED_CANDIDATE_SHA256
    assert retirement["candidate_excluded_from_registry_v0_17"] is True
    assert retirement["local_qwen_model_bytes"] == 0
    assert retirement["minimax_cloud_qualified_or_used"] is False


def test_v0_17_verification_is_exact_and_raw_text_free_when_present() -> None:
    plan = _load_plan()
    if not REPORT.is_file() or not VERIFICATION.is_file():
        pytest.skip("v0.4 closure verification is not materialized")
    report = _load(REPORT)
    expected = build_mix_closure_source_selection_verification_v4(
        plan=plan,
        report=report,
        report_file_sha256=file_sha256(REPORT),
    )
    assert expected == _load(VERIFICATION)
    serialized = REPORT.read_text(encoding="utf-8") + VERIFICATION.read_text(
        encoding="utf-8"
    )
    for forbidden in ('"messages"', '"assistant_response"', '"raw_text"'):
        assert forbidden not in serialized
