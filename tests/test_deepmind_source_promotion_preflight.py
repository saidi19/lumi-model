from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_data import load_sft_policy
from lumi.posttraining_deepmind_source_promotion_preflight import (
    EXPECTED_SOURCE_AFTER,
    build_proposed_successor,
    load_preflight_plan,
    validate_preflight_report,
)
from lumi.posttraining_sources import SFTSource

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json"
REPORT_PATH = ROOT / "evidence/lumi-4b-sft-deepmind-source-promotion-preflight-v0.1.json"


def _predecessor() -> dict:
    return {
        "authorization": {
            "bulk_data_download_authorized": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "metadata_and_source_admissibility_audit_only",
        },
        "observed_at": "2026-08-06",
        "schema_version": "lumi.sft.sources.v0.1",
        "sources": [{"source_id": f"source_{index}"} for index in range(11)],
    }


def test_proposed_successor_appends_only_deepmind() -> None:
    predecessor = _predecessor()
    successor = build_proposed_successor(predecessor)
    assert successor["sources"][:11] == predecessor["sources"]
    assert successor["sources"][11] == EXPECTED_SOURCE_AFTER
    assert len(successor["sources"]) == 12


def test_authorized_github_schema_extension_accepts_exact_source() -> None:
    policy = load_sft_policy(ROOT / "configs/lumi-4b-sft-data-contract-v0.1.json")
    source = SFTSource.from_mapping(
        EXPECTED_SOURCE_AFTER,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    assert source.repository_url == EXPECTED_SOURCE_AFTER["repository_url"]


def test_exact_plan_loads_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind promotion-preflight plan is not materialized")
    with pytest.raises(ValueError, match="source_registry_implementation"):
        load_preflight_plan(PLAN_PATH, repository_root=ROOT)


def test_actual_report_is_raw_free_and_nonmutating_when_available() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("DeepMind promotion-preflight report is not materialized")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    validate_preflight_report(report)
    assert report["execution"]["candidate_registry_promotions"] == 0
    assert report["raw_training_text_present"] is False
