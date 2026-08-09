from __future__ import annotations

from collections import Counter

from lumi.posttraining_first_party_teacher_canary import (
    LANGUAGE_MODES,
    SCENARIOS,
    TASKS,
    expected_record_ids,
    ollama_payload,
    seed_for_record,
    select_owner_sample,
)


def _fake_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for language in LANGUAGE_MODES:
            record_id = f"lumi-teacher-{scenario.task}-{language}-{scenario.variant:02d}"
            rows.append(
                {
                    "record_id": record_id,
                    "task": scenario.task,
                    "skill_id": scenario.scenario_id,
                    "language_mode": language,
                    "split": "validation" if scenario.variant == 3 else "train",
                    "messages": [
                        {"role": "user", "content": f"user {record_id}"},
                        {"role": "assistant", "content": f"answer {record_id}."},
                    ],
                }
            )
    return rows


def test_exact_coverage_and_splits() -> None:
    record_ids = expected_record_ids()
    assert len(record_ids) == 80
    assert len(set(record_ids)) == 80
    rows = _fake_rows()
    assert Counter(row["split"] for row in rows) == {
        "train": 60,
        "validation": 20,
    }
    assert Counter(row["task"] for row in rows) == {task: 16 for task in TASKS}
    assert Counter(row["language_mode"] for row in rows) == {
        language: 20 for language in LANGUAGE_MODES
    }


def test_seed_and_payload_are_deterministic_and_local() -> None:
    scenario = SCENARIOS[0]
    record_id = expected_record_ids()[0]
    assert seed_for_record(record_id) == seed_for_record(record_id)
    payload = ollama_payload(record_id, scenario, "banglish")
    assert payload["model"].startswith("lumi-qwen3-8b-teacher-canary")
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_predict"] == 384
    assert "/no_think" in payload["prompt"]


def test_owner_sample_is_deterministic_balanced_and_exactly_32() -> None:
    rows = _fake_rows()
    first = select_owner_sample(rows)
    second = select_owner_sample(rows)
    assert first == second
    assert len(first) == 32
    assert len({row["record_id"] for row in first}) == 32
    cells = Counter((row["task"], row["language_mode"]) for row in first)
    assert set(cells) == {(task, language) for task in TASKS for language in LANGUAGE_MODES}
    assert min(cells.values()) >= 1
