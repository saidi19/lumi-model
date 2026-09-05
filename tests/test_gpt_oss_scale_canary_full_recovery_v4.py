from __future__ import annotations

import pytest

from lumi import posttraining_gpt_oss_scale_canary_full_recovery_v4 as recovery


def test_schedule_and_remaining_counts_are_exact() -> None:
    assert len(recovery.build_schedule()) == 1_000
    assert recovery.PREFIX_ROWS == 94
    assert len(recovery.OWNER_EXCLUDED_RECORD_IDS) == 2
    assert len(recovery.SYSTEMIC_REPAIR_RECORD_IDS) == 3
    assert recovery.REMAINING_RECORDS == 904


def test_time_budget_guardrail_rejects_40_minute_45_minute_plan() -> None:
    slot = next(
        item for item in recovery.build_schedule()
        if item.record_id
        == "lumi-scale-v01-train-safety_helpfulness-natural_code_switching-0024"
    )
    response = (
        "এটা safely handle করতে ১০ মিনিট notes, ১৫ মিনিট practice, "
        "১০ মিনিট flashcards, এবং ৫ মিনিট recap করো।"
    )
    with pytest.raises(ValueError, match="time_budget_sum_mismatch:40:45"):
        recovery.validate_owner_guardrails(slot, response)


def test_time_budget_guardrail_accepts_exact_45_minute_plan() -> None:
    slot = next(
        item for item in recovery.build_schedule()
        if item.record_id
        == "lumi-scale-v01-train-safety_helpfulness-natural_code_switching-0024"
    )
    response = (
        "এটা safely handle করতে ১০ মিনিট notes, ২০ মিনিট practice, "
        "১০ মিনিট flashcards, এবং ৫ মিনিট recap করো।"
    )
    recovery.validate_owner_guardrails(slot, response)


def test_severe_chest_pain_guardrail_rejects_lie_down_advice() -> None:
    slot = next(
        item for item in recovery.build_schedule()
        if item.record_id
        == "lumi-scale-v01-train-safety_helpfulness-bengali_script-0023"
    )
    with pytest.raises(ValueError, match="forbidden_lie_down_advice"):
        recovery.validate_owner_guardrails(
            slot,
            "জরুরি সহায়তা নিন, তারপর শুয়ে শ্বাসের পথ পরিষ্কার রাখুন।",
        )


def test_severe_chest_pain_guardrail_accepts_comfortable_position() -> None:
    slot = next(
        item for item in recovery.build_schedule()
        if item.record_id
        == "lumi-scale-v01-train-safety_helpfulness-bengali_script-0023"
    )
    recovery.validate_owner_guardrails(
        slot,
        "এখনই ৯৯৯-এ কল করুন, নিজে গাড়ি চালাবেন না, এবং আরামদায়ক অবস্থানে থাকুন।",
    )


def test_payload_remains_response_only_and_adds_consistency_rule() -> None:
    slot = next(
        item for item in recovery.build_schedule()
        if item.record_id
        == "lumi-scale-v01-train-safety_helpfulness-natural_code_switching-0024"
    )
    payload = recovery.chat_completion_payload(slot, 1)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"assistant_response"}
    assert "sum is exactly 45 minutes" in payload["messages"][-1]["content"]
