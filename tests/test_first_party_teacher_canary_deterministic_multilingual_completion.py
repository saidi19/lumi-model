from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from lumi import posttraining_first_party_teacher_canary as base
from lumi.posttraining_first_party_teacher_canary_v18 import (
    CARRY_RECORDS,
    DETERMINISTIC_COMPLETION_CONTRACT,
    FROZEN_WRITING_TEMPLATES,
    MAXIMUM_LOCAL_INFERENCE_CALLS,
    NEW_RECORDS,
    SOURCE_ID,
    TEMPLATE_CONTRACT,
    TEMPLATE_METHOD,
    deterministic_template_output,
    make_deterministic_template_row,
    validate_deterministic_template,
)


def _scenario(scenario_id: str) -> base.Scenario:
    return next(scenario for scenario in base.SCENARIOS if scenario.scenario_id == scenario_id)


def _remaining_matrix() -> list[tuple[str, base.Scenario, str]]:
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    matrix: list[tuple[str, base.Scenario, str]] = []
    for record_id in base.expected_record_ids()[CARRY_RECORDS:]:
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit(
            "-", 2
        )
        matrix.append((record_id, scenarios[(task, int(variant_text))], language_mode))
    return matrix


def test_contract_is_exactly_fifteen_zero_inference_rows() -> None:
    assert CARRY_RECORDS == 65
    assert NEW_RECORDS == 15
    assert MAXIMUM_LOCAL_INFERENCE_CALLS == 0
    assert DETERMINISTIC_COMPLETION_CONTRACT == {
        "deterministic_template_records": 15,
        "maximum_local_inference_calls": 0,
        "maximum_network_bytes": 0,
        "maximum_operations": 15,
        "maximum_retained_new_records": 15,
        "maximum_retained_supervised_tokens_total": 24_000,
        "maximum_runtime_minutes": 60,
        "total_retained_records": 80,
    }


def test_remaining_matrix_is_exact_writing_language_tail() -> None:
    matrix = _remaining_matrix()
    assert len(matrix) == 15
    assert {scenario.scenario_id for _record_id, scenario, _mode in matrix} == set(
        FROZEN_WRITING_TEMPLATES
    )
    assert {mode for _record_id, _scenario_item, mode in matrix} == set(
        base.LANGUAGE_MODES
    )
    assert matrix[0][0] == "lumi-teacher-writing_summarization-bengali_script-00"
    assert matrix[-1][0] == "lumi-teacher-writing_summarization-natural_code_switching-03"


@pytest.mark.parametrize("scenario_id", sorted(FROZEN_WRITING_TEMPLATES))
@pytest.mark.parametrize("language_mode", base.LANGUAGE_MODES)
def test_every_frozen_template_passes_exact_language_and_grounding_contract(
    scenario_id: str,
    language_mode: str,
) -> None:
    scenario = _scenario(scenario_id)
    output = deterministic_template_output(scenario, language_mode)
    messages = validate_deterministic_template(
        output,
        scenario=scenario,
        language_mode=language_mode,
    )
    natural = [
        item["content"] for item in messages if item["role"] in {"user", "assistant"}
    ]
    assert len(natural) == 2
    for text in natural:
        assert text == text.strip()
        assert text[-1] in ".!?।"
        assert all(literal in text for literal in scenario.required_literals)
    if language_mode == "natural_code_switching":
        assert all(base._BENGALI.search(text) for text in natural)  # noqa: SLF001
        assert all(base._LATIN.search(text) for text in natural)  # noqa: SLF001


@pytest.mark.parametrize("record_id,scenario,language_mode", _remaining_matrix())
def test_each_remaining_row_has_truthful_zero_inference_provenance(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
) -> None:
    row, telemetry = make_deterministic_template_row(record_id, scenario, language_mode)
    provenance = row["provenance"]
    assert row["source_id"] == SOURCE_ID
    assert provenance["prompt_contract"] == TEMPLATE_CONTRACT
    assert provenance["deterministic_template_method"] == TEMPLATE_METHOD
    assert provenance["generated_by_local_teacher_inference"] is False
    assert provenance["runtime_external_api_used"] is False
    assert provenance["human_authored"] is False
    assert provenance["human_authorship_claimed"] is False
    assert telemetry["local_inference_used"] is False
    assert telemetry["eval_count"] == 0
    assert telemetry["prompt_eval_count"] == 0


def test_remaining_templates_have_no_exact_or_near_duplicate_rows() -> None:
    texts: dict[str, str] = {}
    for record_id, scenario, language_mode in _remaining_matrix():
        row, _telemetry = make_deterministic_template_row(record_id, scenario, language_mode)
        text = "\n".join(message["content"] for message in row["messages"])
        normalized = " ".join(text.casefold().split())
        assert normalized not in texts.values()
        texts[record_id] = normalized
    ids = list(texts)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            assert SequenceMatcher(None, texts[left], texts[right]).ratio() < 0.92


def test_template_materialization_does_not_call_local_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local teacher must not be called")

    monkeypatch.setattr(base, "_post_local_json", forbidden)
    for record_id, scenario, language_mode in _remaining_matrix():
        make_deterministic_template_row(record_id, scenario, language_mode)
