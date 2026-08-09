from __future__ import annotations

from collections import Counter

import pytest

from lumi import posttraining_first_party_scale_canary as scale
from lumi import posttraining_first_party_scale_canary_preflight as preflight


def test_schedule_is_exact_and_split_disjoint() -> None:
    schedule = scale.build_schedule()
    assert len(schedule) == 1_000
    assert len({slot.record_id for slot in schedule}) == 1_000
    assert len({slot.user_prompt for slot in schedule}) == 1_000
    assert Counter(slot.split for slot in schedule) == {"train": 800, "validation": 200}
    assert Counter(slot.task for slot in schedule) == preflight.TASK_ALLOCATION
    assert Counter(slot.language_mode for slot in schedule) == preflight.LANGUAGE_ALLOCATION
    train_prompts = {slot.user_prompt for slot in schedule if slot.split == "train"}
    validation_prompts = {
        slot.user_prompt for slot in schedule if slot.split == "validation"
    }
    assert train_prompts.isdisjoint(validation_prompts)


def test_payload_preserves_offline_deterministic_bounds() -> None:
    slot = scale.build_schedule()[0]
    payload = scale.ollama_payload(slot)
    assert payload["model"] == preflight.OLLAMA_MODEL
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "num_ctx": 4096,
        "num_predict": 256,
        "repeat_penalty": 1.05,
        "seed": scale.teacher.seed_for_record(slot.record_id),
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.9,
    }


def test_schema_validator_rejects_changed_user_prompt() -> None:
    slot = next(
        item
        for item in scale.build_schedule()
        if item.task == "factual_qa" and item.language_mode == "english"
    )
    output = {
        "assistant_response": (
            f"The exact answer is {slot.expected_answer}. I used the stated operation and "
            "checked it by reversing the calculation. This gives a clear independent check."
        ),
        "safety_label": slot.safety_label,
        "scenario_id": slot.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": slot.user_prompt + " changed",
    }
    with pytest.raises(ValueError, match="user_prompt_echo_invalid"):
        scale._validate_generated_output(output, slot)  # noqa: SLF001


def test_schedule_hash_is_stable() -> None:
    assert scale.schedule_sha256() == (
        "7c6251fc99159f698424a7d2162a8d0a56389935b0d7d6627186b7cd7fe3a66e"
    )


def test_generation_prompt_forbids_echo_and_repetition() -> None:
    slot = scale.build_schedule()[0]
    prompt = scale.build_generation_prompt(slot)
    assert "Do not restate, quote, translate, or paraphrase the whole user prompt" in prompt
    assert "Use no filler, repeated sentence, repeated clause" in prompt
    assert "silently check exact literals, language, factual grounding" in prompt


def test_quality_validator_rejects_prompt_prefix_echo() -> None:
    scale._validate_response_quality(  # noqa: SLF001
        "Please solve 12 + 5 and state the exact answer with a short verification.",
        "The answer is 17. Subtracting 5 from 17 returns 12, so the result is verified.",
    )
    with pytest.raises(ValueError, match="user_prompt_prefix_echo"):
        scale._validate_response_quality(  # noqa: SLF001
            "Please solve 12 + 5 and state the exact answer with a short verification.",
            "Please solve 12 + 5 and state the exact answer with a short verification. "
            "The result is 17.",
        )


def test_quality_validator_rejects_repeated_content() -> None:
    with pytest.raises(ValueError, match="repeated_sentence"):
        scale._validate_response_quality(  # noqa: SLF001
            "Give me a safe study plan.",
            "Review one topic with care. Review one topic with care. Then take a short break.",
        )
