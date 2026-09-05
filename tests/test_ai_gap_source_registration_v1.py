from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_ai_gap_source_registration_v1 import (
    AUTHORIZATION_BYTES,
    AUTHORIZATION_SHA256,
    AUTHORIZATION_TEXT,
    EXPECTED_SOURCE,
    SOURCE_ID,
    _assert_raw_text_free,
    build_successor,
    canonical_sha256,
    validate_canary_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_authorization_binding_is_exact() -> None:
    assert len(AUTHORIZATION_TEXT.encode()) == AUTHORIZATION_BYTES
    assert hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() == AUTHORIZATION_SHA256


def test_successor_appends_exactly_one_source_without_prefix_change() -> None:
    predecessor = json.loads((ROOT / "configs/lumi-4b-sft-source-registry-v0.22.json").read_text())
    successor = build_successor(predecessor, source_after=EXPECTED_SOURCE, observed_at="2026-09-02")
    assert successor["sources"][:-1] == predecessor["sources"]
    assert successor["sources"][-1] == EXPECTED_SOURCE
    assert len(successor["sources"]) == len(predecessor["sources"]) + 1


def test_source_is_candidate_only_and_machine_only_truthful() -> None:
    assert EXPECTED_SOURCE["source_id"] == SOURCE_ID
    assert EXPECTED_SOURCE["status"] == "candidate_eligible"
    assert EXPECTED_SOURCE["production_eligible"] is False
    assert EXPECTED_SOURCE["origin"] == "synthetic"
    assert EXPECTED_SOURCE["access"] == "first_party"
    assert EXPECTED_SOURCE["blockers"] == []
    assert EXPECTED_SOURCE["production_blockers"]


def test_canary_evidence_reconstructs_without_candidate_rows() -> None:
    report = json.loads((ROOT / "evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-v0.1.json").read_text())
    verification = json.loads((ROOT / "evidence/lumi-4b-sft-ai-assisted-gap-closure-canary-verification-v0.1.json").read_text())
    summary = validate_canary_evidence(report, verification)
    assert summary["accepted_records"] == 1_000
    assert summary["train_supervised_tokens"] == 86_712


def test_successor_rejects_duplicate_source() -> None:
    predecessor = json.loads((ROOT / "configs/lumi-4b-sft-source-registry-v0.22.json").read_text())
    predecessor["sources"].append(dict(EXPECTED_SOURCE))
    with pytest.raises(ValueError, match="source count"):
        build_successor(predecessor, source_after=EXPECTED_SOURCE, observed_at="2026-09-02")


def test_evidence_guard_rejects_raw_candidate_keys() -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        _assert_raw_text_free({"messages": []})
    _assert_raw_text_free({"accepted_records": 1000, "seal": canonical_sha256({})})

