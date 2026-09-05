from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from lumi.posttraining_punctuation import (
    CONTENT_CONTEXTS,
    LANGUAGE_MODES,
    PUNCTUATION_OPERATIONS,
    build_punctuation_contract_audit,
    build_punctuation_lanes,
    load_punctuation_quality_plan,
    score_punctuation_outcomes,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-punctuation-quality-gate-v0.1.json"


def _passing_labels(lanes: list) -> dict[str, dict[str, bool]]:
    return {
        lane.case_id: {
            "human_naturalness_approved": True,
            "meaning_preserved": True,
            "protected_span_preserved": True,
            "punctuation_correct": True,
        }
        for lane in lanes
    }


def test_punctuation_plan_builds_exact_raw_text_free_coverage_matrix() -> None:
    plan = load_punctuation_quality_plan(PLAN_PATH)
    lanes = build_punctuation_lanes(plan)
    assert len(lanes) == 72
    assert len({lane.case_id for lane in lanes}) == 72
    assert Counter(lane.language_mode for lane in lanes) == {
        language: 24 for language in LANGUAGE_MODES
    }
    assert Counter(lane.punctuation_operation for lane in lanes) == {
        operation: 12 for operation in PUNCTUATION_OPERATIONS
    }
    assert Counter(lane.content_context for lane in lanes) == {
        context: 18 for context in CONTENT_CONTEXTS
    }
    assert not any(hasattr(lane, "prompt") or hasattr(lane, "response") for lane in lanes)


def test_contract_audit_does_not_claim_model_punctuation_quality() -> None:
    audit = build_punctuation_contract_audit(load_punctuation_quality_plan(PLAN_PATH))
    assert audit["lanes"] == 72
    assert audit["future_model_cases_required"] == 288
    assert audit["oracle_outcomes_meet_thresholds"] is True
    assert audit["model_inference_executed"] is False
    assert audit["model_quality_gate_passed"] is False
    assert audit["model_response_text_accepted"] is False
    assert audit["report_kind"] == "policy_oracle_audit_not_model_quality_evidence"


def test_protected_span_or_meaning_corruption_fails_closed() -> None:
    lanes = build_punctuation_lanes(load_punctuation_quality_plan(PLAN_PATH))
    labels = _passing_labels(lanes)
    protected = next(lane for lane in lanes if lane.content_context == "url_or_path")
    labels[protected.case_id]["protected_span_preserved"] = False
    report = score_punctuation_outcomes(lanes, labels)
    assert report["failed_lanes"] == 1
    assert report["rates"]["protected_span_preservation_rate"] < 1.0
    assert report["quality_thresholds_met"] is False


def test_punctuation_plan_rejects_scope_and_threshold_drift(tmp_path: Path) -> None:
    value = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["authorization"]["model_inference_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_punctuation_quality_plan(path)

    changed = copy.deepcopy(value)
    changed["post_sft_evaluation"]["thresholds"]["protected_span_preservation_rate"] = 0.99
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="quality thresholds"):
        load_punctuation_quality_plan(path)
