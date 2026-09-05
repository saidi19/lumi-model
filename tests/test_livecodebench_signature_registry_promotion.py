from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lumi.posttraining_bigcodebench_signatures import sha256_value
from lumi.posttraining_livecodebench_signature_promotion import (
    TARGET_QUALITY_SHA256,
    TARGET_REGISTRY_SHA256,
    build_successor_payloads,
    load_promotion_authorization,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-livecodebench-signature-registry-promotion-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-livecodebench-signature-registry-promotion-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("LiveCodeBench registry-promotion authorization is not materialized")
    return load_promotion_authorization(PLAN, repository_root=ROOT)


def test_promotion_payloads_match_preflight_hashes() -> None:
    plan = _load_plan()
    registry, quality, _promotion = build_successor_payloads(plan, repository_root=ROOT)
    assert hashlib.sha256(registry).hexdigest() == TARGET_REGISTRY_SHA256
    assert hashlib.sha256(quality).hexdigest() == TARGET_QUALITY_SHA256


def test_promotion_preserves_research_only_boundaries() -> None:
    plan = _load_plan()
    authorization = plan["authorization"]
    assert authorization["active_contract_activation_authorized"] is False
    assert authorization["sft_training_authorized"] is False
    assert authorization["training_ingestion_authorized"] is False
    assert authorization["rights_clearance_claimed"] is False
    assert authorization["v0_2_mutation_authorized"] is False


def test_materialized_report_is_sealed_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("LiveCodeBench registry-promotion report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    assert sha256_value(unsigned) == report["report_sha256"]
    assert report["production_eligible"] is False
