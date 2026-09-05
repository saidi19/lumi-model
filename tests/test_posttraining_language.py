from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_language import (
    build_language_routing_cases,
    build_language_routing_contract_audit,
    load_language_routing_plan,
    score_language_routing_modes,
    select_response_language_mode,
)

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs" / "lumi-4b-language-routing-gate-v0.1.json"


def test_language_routing_plan_builds_exact_raw_text_free_matrix() -> None:
    plan = load_language_routing_plan(PLAN_PATH)
    cases = build_language_routing_cases(plan)
    assert len(cases) == 216
    assert len({case.case_id for case in cases}) == 216
    assert {case.expected_response_mode for case in cases} == {
        "english",
        "bengali_script",
        "banglish",
        "code_switching",
        "other_language",
    }
    audit = build_language_routing_contract_audit(plan)
    assert audit["contract_pass_rate"] == 1.0
    assert audit["explicit_override_pass_rate"] == 1.0
    assert audit["model_inference_executed"] is False
    assert audit["model_response_text_accepted"] is False
    assert audit["report_kind"] == (
        "policy_oracle_contract_audit_not_model_quality_evidence"
    )


def test_language_routing_precedence_is_explicit_then_latest_then_context() -> None:
    assert select_response_language_mode(
        explicit_response_mode="english",
        latest_user_mode="bengali_script",
        established_conversation_mode="banglish",
    ) == "english"
    assert select_response_language_mode(
        explicit_response_mode=None,
        latest_user_mode="banglish",
        established_conversation_mode="bengali_script",
    ) == "banglish"
    assert select_response_language_mode(
        explicit_response_mode=None,
        latest_user_mode="undetermined",
        established_conversation_mode="bengali_script",
    ) == "bengali_script"
    assert select_response_language_mode(
        explicit_response_mode=None,
        latest_user_mode="undetermined",
        established_conversation_mode=None,
    ) == "english"


def test_language_routing_scorer_retains_labels_not_model_text() -> None:
    cases = build_language_routing_cases(load_language_routing_plan(PLAN_PATH))
    observed = {case.case_id: case.expected_response_mode for case in cases}
    observed[cases[0].case_id] = (
        "banglish" if cases[0].expected_response_mode != "banglish" else "english"
    )
    report = score_language_routing_modes(cases, observed)
    assert report["contract_pass_rate"] < 1.0
    assert report["result_counts"]["failed"] == 1
    assert report["model_response_text_accepted"] is False


def test_language_routing_plan_rejects_support_or_authorization_drift(
    tmp_path: Path,
) -> None:
    value = json.loads(PLAN_PATH.read_text())
    changed = copy.deepcopy(value)
    changed["authorization"]["model_inference_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_language_routing_plan(path)

    changed = copy.deepcopy(value)
    changed["model_behavior_contract"]["other_language_release_status"] = "supported"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="support claim"):
        load_language_routing_plan(path)
