from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_math_replacement_preflight import (
    DEEPMIND_OBJECTS,
    DMATH_OBJECTS,
    EXPECTED_AI_V3_METRICS,
    file_sha256,
    load_math_replacement_plan,
    rebuild_math_replacement_preflight,
    validate_math_replacement_report,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-math-replacement-preflight-v0.1.json"
REPORT = ROOT / "evidence/lumi-4b-sft-math-replacement-preflight-v0.1.json"
VERIFICATION = ROOT / "evidence/lumi-4b-sft-math-replacement-preflight-verification-v0.1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_source_boundaries_are_small_and_no_download() -> None:
    plan = load_math_replacement_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["network_access_authorized"] is False
    assert plan["authorization"]["dataset_acquisition_authorized"] is False
    assert plan["authorization"]["production_sft_authorized"] is False
    assert len(DMATH_OBJECTS) == 3
    assert sum(item["size_bytes"] for item in DMATH_OBJECTS) == 12_206_887
    assert len(DEEPMIND_OBJECTS) == 30
    assert sum(item["size_bytes"] for item in DEEPMIND_OBJECTS) == 234_672
    assert plan["acquisition_policy"]["immediate_source_id"] == "dmath"


def test_math_replacement_report_reproduces_exact_token_gap() -> None:
    plan = load_math_replacement_plan(PLAN, repository_root=ROOT)
    rebuilt = rebuild_math_replacement_preflight(
        plan=plan,
        plan_file_sha256=file_sha256(PLAN),
        repository_root=ROOT,
    )
    observed = _load(REPORT)
    assert rebuilt == observed
    validate_math_replacement_report(observed)
    assert observed["source_token_evidence"]["ai_assisted_v0_3"] == EXPECTED_AI_V3_METRICS
    assert observed["mix_gap"] == {
        "learning_candidate_floor_gap_after_reviewed_draft": 8_997_800,
        "planning_train_supervised_tokens": 1_002_200,
        "reasoning_math_maximum_gap_after_reviewed_draft": 6_599_880,
        "reasoning_math_minimum_gap_after_reviewed_draft": 2_199_880,
        "registered_reviewed_train_supervised_tokens": 992_886,
        "reviewed_unregistered_ai_v3_train_supervised_tokens": 9_314,
    }


def test_report_and_verification_are_raw_text_free_and_nonproduction() -> None:
    report = _load(REPORT)
    verification = _load(VERIFICATION)
    assert report["production_eligible"] is False
    assert report["production_dataset_promotions"] == 0
    assert verification["production_eligible"] is False
    serialized = REPORT.read_text(encoding="utf-8")
    for forbidden in (
        '"messages"',
        '"question_en"',
        '"answer_en"',
        '"solution_code_en"',
        '"assistant_response"',
        '"user_prompt"',
    ):
        assert forbidden not in serialized


def test_plan_rejects_source_decision_drift(tmp_path: Path) -> None:
    changed = copy.deepcopy(_load(PLAN))
    changed["shortlist"][0]["decision"] = "production_eligible"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="shortlist"):
        load_math_replacement_plan(path, repository_root=ROOT)
