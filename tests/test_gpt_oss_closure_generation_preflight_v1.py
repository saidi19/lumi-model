from __future__ import annotations

import json
from pathlib import Path

from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_gpt_oss_closure_generation_preflight_v1 import (
    EXPECTED_REMOTE_BOUNDS,
    EXPECTED_SCHEDULE_SPEC_SHA256,
    build_preflight_report,
    load_preflight_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-closure-generation-preflight-v0.1.json"


def test_plan_is_local_only_and_schedule_is_exact() -> None:
    plan = load_preflight_plan(PLAN, repository_root=ROOT)
    assert all(
        value is False
        for key, value in plan["authorization"].items()
        if key.endswith("_authorized")
    )
    assert plan["remote_bounds_if_authorized"] == EXPECTED_REMOTE_BOUNDS


def test_report_is_raw_text_free_and_non_authorizing() -> None:
    plan = load_preflight_plan(PLAN, repository_root=ROOT)
    report = build_preflight_report(
        plan, plan_file_sha256=file_sha256(PLAN), repository_root=ROOT
    )
    assert report["schedule_spec_sha256"] == EXPECTED_SCHEDULE_SPEC_SHA256
    assert report["closure"]["selected_retained_train_token_target"] == 3_200_000
    assert report["closure"]["task_gap_headroom"] == 318_240
    assert all(report["checks"].values())
    assert not any(report["production"].values())
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        assert forbidden not in serialized
