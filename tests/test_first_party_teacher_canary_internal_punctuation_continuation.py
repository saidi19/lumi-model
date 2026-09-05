from __future__ import annotations

import copy
import json

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v9 as terminal_v9
from lumi import posttraining_first_party_teacher_canary_v13 as safety_v13
from lumi.posttraining_first_party_teacher_canary_v14 import (
    CARRY_RECORDS,
    CONTINUATION_CONTRACT,
    NEW_LANGUAGE_BOUND_SAFETY_ROWS,
    NEW_NATURAL_CODE_SWITCHING_ROWS,
    NEW_RECORDS,
    PROMPT_CONTRACT,
    TERMINAL_COMPOSITION_METHOD,
    TERMINAL_PROMPT_METHOD,
    _make_row,
    _patched_v13_generation,
    compose_terminal_output,
    continuation_payload,
    continuation_prompt,
    normalize_terminal_bengali_body,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


def _output(*, user: str, assistant: str) -> dict[str, object]:
    return {
        "assistant_response": assistant,
        "safety_label": "refuse",
        "scenario_id": "credential_privacy",
        "tool_call": None,
        "tool_result": None,
        "user_prompt": user,
    }


def test_contract_matches_exact_remaining_prefix() -> None:
    assert CARRY_RECORDS == 39
    assert NEW_RECORDS == 41
    assert NEW_NATURAL_CODE_SWITCHING_ROWS == 11
    assert NEW_LANGUAGE_BOUND_SAFETY_ROWS == 9
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 123,
        "maximum_attempts_per_record": 3,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 41,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }


@pytest.mark.parametrize(
    "language_mode",
    ["banglish", "bengali_script", "english"],
)
@pytest.mark.parametrize(
    "scenario_id",
    ["credential_privacy", "urgent_medical_caution", "punctuation_repair"],
)
def test_non_code_switch_prompts_and_payloads_are_byte_equal_v13(
    language_mode: str,
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-{scenario.task}-{language_mode}-02"
    assert continuation_prompt(scenario, language_mode) == safety_v13.continuation_prompt(
        scenario,
        language_mode,
    )
    assert continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    ) == safety_v13.continuation_payload(
        record_id,
        scenario,
        language_mode,
        attempt_number=2,
    )


@pytest.mark.parametrize(
    "scenario_id",
    [
        "credential_privacy",
        "urgent_medical_caution",
        "punctuation_repair",
        "benign_study_help",
    ],
)
def test_code_switch_payload_changes_only_prompt_and_removes_contradiction(
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)
    record_id = f"lumi-teacher-{scenario.task}-natural_code_switching-02"
    previous = safety_v13.continuation_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    corrected = continuation_payload(
        record_id,
        scenario,
        "natural_code_switching",
        attempt_number=2,
    )
    previous_without_prompt = copy.deepcopy(previous)
    corrected_without_prompt = copy.deepcopy(corrected)
    previous_without_prompt.pop("prompt")
    corrected_without_prompt.pop("prompt")
    assert corrected_without_prompt == previous_without_prompt
    assert corrected["prompt"] != previous["prompt"]
    assert "internal punctuation লিখবে না" not in corrected["prompt"]
    assert "internal punctuation byte-exact" in corrected["prompt"]


def test_normalizer_preserves_internal_sentence_punctuation() -> None:
    text = "আমি এটি করতে পারব না। নিরাপদ বিকল্পে সাহায্য করতে পারি।"
    normalized, removed = normalize_terminal_bengali_body(
        text,
        field="assistant_response",
    )
    assert normalized == "আমি এটি করতে পারব না। নিরাপদ বিকল্পে সাহায্য করতে পারি"
    assert removed == "।"
    assert "না। নিরাপদ" in normalized


def test_normalizer_preserves_internal_comma_without_terminal() -> None:
    text = "এটি জরুরি হতে পারে, তাই এখনই সহায়তা নিন"
    normalized, removed = normalize_terminal_bengali_body(
        text,
        field="assistant_response",
    )
    assert normalized == text
    assert removed is None


@pytest.mark.parametrize(
    "text",
    [
        "আমি পারব না।।",
        "আমি পারব না?!",
        "আমি পারব না:",
        "আমি password দিতে পারব না।",
        " আমি পারব না।",
        "আমি\nপারব না।",
    ],
)
def test_normalizer_fails_closed_on_ambiguous_or_non_bengali_body(text: str) -> None:
    with pytest.raises(ValueError):
        normalize_terminal_bengali_body(text, field="assistant_response")


def test_composer_accepts_previously_blocking_two_sentence_refusal() -> None:
    scenario = _scenario("credential_privacy")
    output = _output(
        user="আপনি কি অন্যের পাসওয়ার্ড দিতে পারেন?",
        assistant=(
            "পারব না। আপনি যদি কোনো অ্যাকাউন্ট ফিরে পান তাহলে সংশ্লিষ্ট সার্ভিস প্রতিষ্ঠানের সাথে যোগাযোগ করুন।"
        ),
    )
    with safety_v13._patched_v11_generation(  # noqa: SLF001
        language_mode="natural_code_switching"
    ):
        flattened, messages, composition = compose_terminal_output(output, scenario=scenario)
    assert flattened["user_prompt"].endswith(" (password)?")
    assert flattened["assistant_response"].endswith(" (password)।")
    assert "না। আপনি" in flattened["assistant_response"]
    assert messages[1]["content"] == flattened["assistant_response"]
    assert composition["method"] == TERMINAL_COMPOSITION_METHOD
    assert composition["internal_punctuation_policy"] == "preserve_byte_exact"
    assert composition["assistant_removed_terminal"] == "।"


def test_row_records_new_prompt_and_terminal_provenance() -> None:
    scenario = _scenario("credential_privacy")
    output = _output(
        user="আপনি কি অন্যের পাসওয়ার্ড দিতে পারেন?",
        assistant="পারব না। বৈধ উপায়ে অ্যাকাউন্ট ফিরিয়ে নিন।",
    )
    with safety_v13._patched_v11_generation(  # noqa: SLF001
        language_mode="natural_code_switching"
    ):
        _, messages, composition = compose_terminal_output(output, scenario=scenario)
    with _patched_v13_generation():
        row = _make_row(
            "lumi-teacher-safety_helpfulness-natural_code_switching-01",
            scenario,
            "natural_code_switching",
            messages,
            attempt_number=1,
            terminal_composition=composition,
            banglish_composition=None,
        )
    provenance = row["provenance"]
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_14"
    assert provenance["prompt_contract"] == PROMPT_CONTRACT
    assert provenance["code_switch_terminal_prompt_method"] == TERMINAL_PROMPT_METHOD
    assert provenance["terminal_composition_method"] == TERMINAL_COMPOSITION_METHOD
    assert provenance["internal_punctuation_policy"] == "preserve_byte_exact"
    assert provenance["language_bound_safety_method"] == safety_v13.SAFETY_PROMPT_METHOD
    assert provenance["language_bound_safety_marker"] == "পারব না"


def test_generation_patch_restores_all_predecessor_globals() -> None:
    payload = safety_v13.continuation_payload
    make_row = safety_v13._make_row  # noqa: SLF001
    compose = terminal_v9.compose_terminal_output
    method = terminal_v9.TERMINAL_COMPOSITION_METHOD
    with _patched_v13_generation():
        assert safety_v13.continuation_payload is continuation_payload
        assert safety_v13._make_row is _make_row  # noqa: SLF001
        assert terminal_v9.compose_terminal_output is compose_terminal_output
        assert terminal_v9.TERMINAL_COMPOSITION_METHOD == TERMINAL_COMPOSITION_METHOD
    assert safety_v13.continuation_payload is payload
    assert safety_v13._make_row is make_row  # noqa: SLF001
    assert terminal_v9.compose_terminal_output is compose
    assert terminal_v9.TERMINAL_COMPOSITION_METHOD == method


def test_composition_json_is_hash_stable() -> None:
    scenario = _scenario("credential_privacy")
    output = _output(
        user="আপনি কি অন্যের পাসওয়ার্ড দিতে পারেন?",
        assistant="পারব না। বৈধ পুনরুদ্ধার পদ্ধতি ব্যবহার করুন।",
    )
    with safety_v13._patched_v11_generation(  # noqa: SLF001
        language_mode="natural_code_switching"
    ):
        _, _, first = compose_terminal_output(output, scenario=scenario)
        _, _, second = compose_terminal_output(json.loads(json.dumps(output)), scenario=scenario)
    assert first == second
    assert base.sha256_value(first) == base.sha256_value(second)
