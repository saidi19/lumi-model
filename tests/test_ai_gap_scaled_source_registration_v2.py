from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lumi.posttraining_ai_gap_scaled_source_registration_v2 as contract


ROOT = Path(__file__).resolve().parents[1]


def test_authorization_binding_is_exact() -> None:
    assert len(contract.AUTHORIZATION_TEXT.encode()) == contract.AUTHORIZATION_BYTES
    assert hashlib.sha256(contract.AUTHORIZATION_TEXT.encode()).hexdigest() == contract.AUTHORIZATION_SHA256


def test_successor_appends_one_source_without_prefix_change() -> None:
    predecessor = json.loads((ROOT / "configs/lumi-4b-sft-source-registry-v0.23.json").read_text())
    successor = contract.build_successor(
        predecessor, source_after=contract.EXPECTED_SOURCE, observed_at="2026-09-02"
    )
    assert successor["sources"][:-1] == predecessor["sources"]
    assert successor["sources"][-1] == contract.EXPECTED_SOURCE
    assert len(successor["sources"]) == 27


def test_source_is_truthful_candidate_only_scaled_lane() -> None:
    source = contract.EXPECTED_SOURCE
    assert source["source_id"] == contract.SOURCE_ID
    assert source["status"] == "candidate_eligible"
    assert source["production_eligible"] is False
    assert source["origin"] == "synthetic"
    assert source["access"] == "first_party"
    assert source["license_id"] == "lumi-owned"
    assert source["production_blockers"]


def test_scaled_evidence_reconstructs_without_candidate_rows() -> None:
    report = json.loads((ROOT / "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json").read_text())
    verification = json.loads((ROOT / "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-verification-v0.2.json").read_text())
    summary = contract.validate_scaled_evidence(report, verification)
    assert summary["accepted_records"] == 13_420
    assert summary["train_records"] == 10_736
    assert summary["validation_records"] == 2_684
    assert summary["train_supervised_tokens"] == 1_157_905
    assert summary["source_revision"] == contract.SOURCE_REVISION


def test_successor_rejects_duplicate_source() -> None:
    predecessor = json.loads((ROOT / "configs/lumi-4b-sft-source-registry-v0.23.json").read_text())
    predecessor["sources"][-1] = dict(contract.EXPECTED_SOURCE)
    with pytest.raises(ValueError, match="already exists"):
        contract.build_successor(
            predecessor, source_after=contract.EXPECTED_SOURCE, observed_at="2026-09-02"
        )


def test_raw_evidence_guard() -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        contract._assert_raw_text_free({"messages": []})
    contract._assert_raw_text_free({"accepted_records": 13_420, "seal": contract.canonical_sha256({})})
