from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_bigcodebench_signature_promotion import (
    TARGET_QUALITY_SHA256,
    TARGET_REGISTRY_SHA256,
    build_successor_payloads,
    load_promotion_authorization,
    validate_report,
)
from lumi.posttraining_bigcodebench_signatures import sha256_value

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-bigcodebench-signature-registry-promotion-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-bigcodebench-signature-registry-promotion-v0.1.json"


def _load_plan() -> dict:
    if not PLAN.is_file():
        pytest.skip("BigCodeBench registry-promotion authorization is not materialized")
    return load_promotion_authorization(PLAN, repository_root=ROOT)


def test_promotion_payloads_match_preflight_hashes() -> None:
    plan = _load_plan()
    registry, quality, _promotion = build_successor_payloads(plan, repository_root=ROOT)
    import hashlib

    assert hashlib.sha256(registry).hexdigest() == TARGET_REGISTRY_SHA256
    assert hashlib.sha256(quality).hexdigest() == TARGET_QUALITY_SHA256


def test_promotion_does_not_authorize_activation_or_sft() -> None:
    plan = _load_plan()
    assert plan["authorization"]["active_contract_activation_authorized"] is False
    assert plan["authorization"]["sft_training_authorized"] is False
    assert plan["authorization"]["livecodebench_access_authorized"] is False


def test_materialized_report_is_sealed_when_present() -> None:
    _load_plan()
    if not REPORT.is_file():
        pytest.skip("BigCodeBench registry-promotion report is not materialized")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    assert sha256_value(unsigned) == report["report_sha256"]
    assert report["production_eligible"] is False
