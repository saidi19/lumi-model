from __future__ import annotations

import copy

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v10 as english_v10
from lumi.posttraining_first_party_teacher_canary_v11 import (
    BANGLISH_TERMINAL_METHOD,
    PROMPT_CONTRACT,
    _make_row,
    compose_banglish_terminal_output,
    continuation_payload,
    continuation_prompt,
    normalize_banglish_terminal,
)


def _multilingual_greeting() -> base.Scenario:
    return next(
        scenario
        for scenario in base.SCENARIOS
        if scenario.task == "multilingual_bengali" and scenario.variant == 0
    )


def _banglish_output(*, user: str, assistant: str) -> dict[str, object]:
    return {
        "assistant_response": assistant,
        "safety_label": "safe",
        "scenario_id": "friendly_greeting",
        "tool_call": None,
        "tool_result": None,
        "user_prompt": user,
    }


def test_missing_banglish_terminals_receive_only_bound_suffixes() -> None:
    scenario = _multilingual_greeting()
    output = _banglish_output(
        user="Tumi kemon acho",
        assistant="Ami bhalo achi dhonnobad",
    )
    normalized, messages, composition = compose_banglish_terminal_output(
        output,
        scenario=scenario,
    )
    assert normalized["user_prompt"] == f"{output['user_prompt']}?"
    assert normalized["assistant_response"] == f"{output['assistant_response']}."
    assert messages[0]["content"] == normalized["user_prompt"]
    assert messages[-1]["content"] == normalized["assistant_response"]
    assert composition["method"] == BANGLISH_TERMINAL_METHOD
    assert composition["user_prompt_appended_terminal"] == "?"
    assert composition["assistant_response_appended_terminal"] == "."


def test_existing_banglish_terminals_are_byte_preserved() -> None:
    user = "Tumi kemon acho?"
    assistant = "Ami bhalo achi dhonnobad!"
    normalized, messages, composition = compose_banglish_terminal_output(
        _banglish_output(user=user, assistant=assistant),
        scenario=_multilingual_greeting(),
    )
    assert normalized["user_prompt"] == user
    assert normalized["assistant_response"] == assistant
    assert messages[0]["content"] == user
    assert messages[-1]["content"] == assistant
    assert composition["user_prompt_action"] == "preserve_existing_terminal"
    assert composition["assistant_response_action"] == "preserve_existing_terminal"


def test_unsupported_trailing_punctuation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported punctuation"):
        normalize_banglish_terminal("Tumi kemon acho,", field="user_prompt")


def test_surrounding_whitespace_fails_closed() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        normalize_banglish_terminal(" Tumi kemon acho", field="user_prompt")


@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english", "natural_code_switching"],
)
def test_v11_prompts_and_payloads_equal_v10(language_mode: str) -> None:
    scenario = _multilingual_greeting()
    record_id = f"lumi-teacher-multilingual_bengali-{language_mode}-01"
    assert continuation_prompt(scenario, language_mode) == english_v10.continuation_prompt(
        scenario,
        language_mode,
    )
    assert continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    ) == english_v10.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )


def test_non_banglish_output_is_not_normalized() -> None:
    output = _banglish_output(
        user="Tumi kemon acho",
        assistant="Ami bhalo achi dhonnobad",
    )
    untouched = copy.deepcopy(output)
    continuation_prompt(_multilingual_greeting(), "bengali_script")
    assert output == untouched


def test_v11_row_binds_exact_banglish_composition() -> None:
    scenario = _multilingual_greeting()
    composition = {
        "assistant_response_action": "append_missing_terminal",
        "assistant_response_after_sha256": "after",
        "assistant_response_appended_terminal": ".",
        "assistant_response_before_sha256": "before",
        "edits_per_field_maximum": 1,
        "method": BANGLISH_TERMINAL_METHOD,
        "user_prompt_action": "append_missing_terminal",
        "user_prompt_after_sha256": "after",
        "user_prompt_appended_terminal": "?",
        "user_prompt_before_sha256": "before",
    }
    row = _make_row(
        "lumi-teacher-multilingual_bengali-banglish-01",
        scenario,
        "banglish",
        [
            {"role": "user", "content": "Tumi kemon acho?"},
            {"role": "assistant", "content": "Ami bhalo achi dhonnobad."},
        ],
        attempt_number=1,
        terminal_composition=None,
        banglish_composition=composition,
    )
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_11"
    assert row["provenance"]["prompt_contract"] == PROMPT_CONTRACT
    assert row["provenance"]["banglish_terminal_composition_method"] == (BANGLISH_TERMINAL_METHOD)
    assert row["provenance"]["deterministic_banglish_terminal_normalization"] is True
    assert row["provenance"]["banglish_terminal_source_internal_sha256"] == base.sha256_value(
        composition
    )
