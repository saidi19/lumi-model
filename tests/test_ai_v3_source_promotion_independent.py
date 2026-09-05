from __future__ import annotations

from pathlib import Path

from lumi.posttraining_ai_v3_source_promotion import SUCCESSOR_SHA256
from lumi.posttraining_ai_v3_source_promotion_independent import (
    EXPECTED_CHECKS,
    verify_independent_materialization,
    verify_independent_verification_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "lumi-4b-sft-ai-v3-source-promotion-v0.1.json"


def test_materialized_successor_and_receipt_verify_independently() -> None:
    report = verify_independent_materialization(PLAN, repository_root=ROOT)
    verify_independent_verification_report(report)
    assert report["checks"] == EXPECTED_CHECKS
    assert report["successor_registry_sha256"] == SUCCESSOR_SHA256
    assert report["production_eligible"] is False
    assert report["checks"]["candidate_rows_modified"] is False
    assert report["checks"]["production_dataset_promotions"] == 0
