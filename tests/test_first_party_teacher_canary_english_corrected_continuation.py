from __future__ import annotations

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi.posttraining_first_party_teacher_canary_v10 import (
    ENGLISH_FORMAT_EXAMPLE,
    ENGLISH_PROMPT_METHOD,
    PROMPT_CONTRACT,
    _make_row,
    continuation_payload,
    continuation_prompt,
    continuation_seed,
)


def _multilingual_greeting() -> base.Scenario:
    return next(
        scenario
        for scenario in base.SCENARIOS
        if scenario.task == "multilingual_bengali" and scenario.variant == 0
    )


def test_english_prompt_replaces_confusing_task_label() -> None:
    scenario = _multilingual_greeting()
    prompt = continuation_prompt(scenario, "english")
    assert "Task: multilingual_bengali" not in prompt
    assert "Task alias: friendly_greeting" in prompt
    assert "zero Bengali characters" in prompt
    assert ENGLISH_FORMAT_EXAMPLE["user_prompt"] in prompt
    assert ENGLISH_FORMAT_EXAMPLE["assistant_response"] in prompt


def test_non_english_prompts_preserve_terminal_v9_paths() -> None:
    scenario = _multilingual_greeting()
    natural = continuation_prompt(scenario, "natural_code_switching")
    bengali = continuation_prompt(scenario, "bengali_script")
    banglish = continuation_prompt(scenario, "banglish")
    assert "terminal punctuation" in natural
    assert base.LANGUAGE_INSTRUCTIONS["bengali_script"] in bengali
    assert "Banglish" in banglish


def test_continuation_payload_is_attempt_bound() -> None:
    scenario = _multilingual_greeting()
    first = continuation_payload(
        "lumi-teacher-multilingual_bengali-english-00",
        scenario,
        "english",
        attempt_number=1,
    )
    second = continuation_payload(
        "lumi-teacher-multilingual_bengali-english-00",
        scenario,
        "english",
        attempt_number=2,
    )
    assert first["options"]["seed"] != second["options"]["seed"]
    assert first["options"]["temperature"] == 0.7


def test_english_row_records_prompt_provenance() -> None:
    scenario = _multilingual_greeting()
    row = _make_row(
        "lumi-teacher-multilingual_bengali-english-00",
        scenario,
        "english",
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well, thank you."},
        ],
        attempt_number=1,
        composition=None,
    )
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_10"
    assert row["provenance"]["english_prompt_method"] == ENGLISH_PROMPT_METHOD
    assert row["provenance"]["prompt_contract"] == PROMPT_CONTRACT
    assert row["provenance"]["seed"] == continuation_seed(row["record_id"], 1)


def test_continuation_seed_rejects_fourth_attempt() -> None:
    with pytest.raises(ValueError, match="outside authorization"):
        continuation_seed("lumi-teacher-multilingual_bengali-english-00", 4)
