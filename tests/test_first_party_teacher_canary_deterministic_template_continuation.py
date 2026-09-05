from __future__ import annotations

from pathlib import Path

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi import posttraining_first_party_teacher_canary_v4 as schema_v4
from lumi import posttraining_first_party_teacher_canary_v15 as task_v15
from lumi.posttraining_first_party_teacher_canary_v17 import (
    CARRY_RECORDS,
    CONTINUATION_CONTRACT,
    DETERMINISTIC_TEMPLATE_RECORDS,
    FROZEN_WRITING_TEMPLATES,
    MAXIMUM_LOCAL_INFERENCE_CALLS,
    NEW_RECORDS,
    NON_TARGET_RECORDS,
    TEMPLATE_CONTRACT,
    TEMPLATE_METHOD,
    _make_template_row,
    _patched_v4_runtime,
    _targeted,
    deterministic_template_output,
    generate_continuation_attempt,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


def _remaining_matrix() -> list[tuple[base.Scenario, str, str]]:
    return list(
        zip(
            (scenario for scenario in base.SCENARIOS for _ in base.LANGUAGE_MODES),
            (
                language_mode
                for _scenario_item in base.SCENARIOS
                for language_mode in base.LANGUAGE_MODES
            ),
            base.expected_record_ids(),
            strict=True,
        )
    )[CARRY_RECORDS:]


def test_contract_bounds_four_templates_and_twelve_model_rows() -> None:
    assert CARRY_RECORDS == 64
    assert NEW_RECORDS == 16
    assert DETERMINISTIC_TEMPLATE_RECORDS == 4
    assert NON_TARGET_RECORDS == 12
    assert MAXIMUM_LOCAL_INFERENCE_CALLS == 36
    assert CONTINUATION_CONTRACT == {
        "additional_inference_attempts": 40,
        "deterministic_template_records": 4,
        "maximum_attempts_per_record": 3,
        "maximum_local_inference_calls": 36,
        "maximum_network_bytes": 0,
        "maximum_retained_new_records": 16,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 120,
        "total_retained_records": 80,
    }


def test_exact_remaining_matrix_has_only_four_template_targets() -> None:
    matrix = _remaining_matrix()
    targeted = [item for item in matrix if _targeted(item[0], item[1])]
    non_target = [item for item in matrix if not _targeted(item[0], item[1])]
    assert len(matrix) == 16
    assert len(targeted) == DETERMINISTIC_TEMPLATE_RECORDS
    assert len(non_target) == NON_TARGET_RECORDS
    assert {scenario.scenario_id for scenario, _mode, _id in targeted} == set(
        FROZEN_WRITING_TEMPLATES
    )
    assert {mode for _scenario_item, mode, _id in targeted} == {"banglish"}


@pytest.mark.parametrize("scenario_id", sorted(FROZEN_WRITING_TEMPLATES))
def test_each_frozen_template_passes_full_task_anchored_validation(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    output = deterministic_template_output(scenario)
    messages = task_v15.validate_task_anchored_banglish(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    assert messages[0]["content"] == FROZEN_WRITING_TEMPLATES[scenario_id]["user_prompt"]
    assert messages[-1]["content"] == FROZEN_WRITING_TEMPLATES[scenario_id][
        "assistant_response"
    ]
    assert task_v15._has_exact_word(messages[0]["content"], "eta")  # noqa: SLF001
    assert task_v15._has_exact_word(messages[-1]["content"], "ami")  # noqa: SLF001
    for literal in scenario.required_literals:
        assert literal in messages[0]["content"]
        assert literal in messages[-1]["content"]


@pytest.mark.parametrize("scenario_id", sorted(FROZEN_WRITING_TEMPLATES))
def test_template_row_records_zero_inference_and_no_human_claim(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    row, telemetry = _make_template_row(
        f"lumi-teacher-writing_summarization-banglish-{scenario.variant:02d}",
        scenario,
        attempt_number=1,
    )
    provenance = row["provenance"]
    assert row["source_id"] == "lumi_first_party_multilingual_safety_tool_v0_17"
    assert provenance["prompt_contract"] == TEMPLATE_CONTRACT
    assert provenance["deterministic_template_method"] == TEMPLATE_METHOD
    assert provenance["generated_by_local_teacher_inference"] is False
    assert provenance["human_authorship_claimed"] is False
    assert telemetry["local_inference_used"] is False
    assert telemetry["eval_count"] == 0
    assert telemetry["prompt_eval_count"] == 0
    assert telemetry["deterministic_template_method"] == TEMPLATE_METHOD
    assert telemetry["banglish_terminal_composition"]["user_prompt_action"] == (
        "preserve_existing_terminal"
    )
    assert telemetry["banglish_terminal_composition"]["assistant_response_action"] == (
        "preserve_existing_terminal"
    )


def test_target_generation_never_calls_local_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario("meeting_summary")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local teacher must not be called for deterministic templates")

    monkeypatch.setattr(base, "_post_local_json", forbidden)
    row, telemetry = generate_continuation_attempt(
        "lumi-teacher-writing_summarization-banglish-00",
        scenario,
        "banglish",
        attempt_number=1,
        endpoint="http://127.0.0.1:11434/api/generate",
        rejection_path=Path("unused.json"),
    )
    assert row["skill_id"] == "meeting_summary"
    assert telemetry["local_inference_used"] is False


@pytest.mark.parametrize(
    "language_mode",
    ["bengali_script", "english", "natural_code_switching"],
)
def test_non_target_generation_delegates_byte_exact_v15(
    monkeypatch: pytest.MonkeyPatch,
    language_mode: str,
) -> None:
    scenario = _scenario("meeting_summary")
    sentinel = ({"row": language_mode}, {"telemetry": language_mode})
    calls: list[tuple[object, ...]] = []

    def fake_generate(*args: object, **kwargs: object) -> tuple[dict[str, str], dict[str, str]]:
        calls.append((*args, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lumi.posttraining_first_party_teacher_canary_v17._V15_GENERATE",
        fake_generate,
    )
    result = generate_continuation_attempt(
        f"lumi-teacher-writing_summarization-{language_mode}-00",
        scenario,
        language_mode,
        attempt_number=2,
        endpoint="http://127.0.0.1:11434/api/generate",
        rejection_path=Path("unused.json"),
    )
    assert result == sentinel
    assert len(calls) == 1
    assert calls[0][-1] == {
        "attempt_number": 2,
        "endpoint": "http://127.0.0.1:11434/api/generate",
        "rejection_path": Path("unused.json"),
    }


def test_template_output_is_deep_copy_safe() -> None:
    scenario = _scenario("meeting_summary")
    first = deterministic_template_output(scenario)
    second = deterministic_template_output(scenario)
    first["user_prompt"] = "changed"
    assert second["user_prompt"] == FROZEN_WRITING_TEMPLATES["meeting_summary"][
        "user_prompt"
    ]


def test_runtime_patch_restores_predecessor_globals() -> None:
    contract = schema_v4.SCHEMA_COMPLETION_CONTRACT
    validate = schema_v4._validate_schema_completion_plan  # noqa: SLF001
    initialize = schema_v4._initialize_or_load  # noqa: SLF001
    generate = schema_v4.generate_schema_completion_attempt
    with _patched_v4_runtime():
        assert schema_v4.SCHEMA_COMPLETION_CONTRACT is CONTINUATION_CONTRACT
    assert schema_v4.SCHEMA_COMPLETION_CONTRACT is contract
    assert schema_v4._validate_schema_completion_plan is validate  # noqa: SLF001
    assert schema_v4._initialize_or_load is initialize  # noqa: SLF001
    assert schema_v4.generate_schema_completion_attempt is generate
