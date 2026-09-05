from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_scale_source_promotion_preflight_v17 import (
    EXPECTED_CANONICAL_FILE_SHA256,
    EXPECTED_PROJECTED_GAP,
    EXPECTED_SUCCESSOR_REGISTRY_SHA256,
    EXPECTED_TOKEN_METRICS,
    build_preflight_report,
    build_reference_projection,
    load_preflight_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-scale-source-promotion-preflight-v0.17.json"


def test_reference_projection_is_exact_and_non_materializing() -> None:
    projection = build_reference_projection(ROOT)["projection"]
    assert projection["canonical_file_sha256"] == EXPECTED_CANONICAL_FILE_SHA256
    assert projection["successor_registry_sha256"] == EXPECTED_SUCCESSOR_REGISTRY_SHA256
    assert projection["token_metrics"] == EXPECTED_TOKEN_METRICS
    assert projection["projected_gap_to_10m"] == EXPECTED_PROJECTED_GAP
    assert projection["filter_accepted"] == 984
    assert projection["filter_rejected"] == 0


def test_plan_and_report_remain_non_authorizing() -> None:
    plan = load_preflight_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    report = build_preflight_report(
        plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT
    )
    assert all(report["checks"].values())
    assert not any(report["production"].values())
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert '"messages"' not in serialized
    assert '"content"' not in serialized
