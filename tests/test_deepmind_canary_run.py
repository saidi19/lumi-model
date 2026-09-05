from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_deepmind_adapter_preflight import derive_seed
from lumi.posttraining_deepmind_canary_run import (
    EXPECTED_GENERATED_RECORDS,
    EXPECTED_SPLIT_COUNTS,
    SOURCE_ROOT,
    SPLIT_MODULE_COUNTS,
    SPLIT_NAMESPACES,
    TOP_MODULE_FILES,
    apply_prompt_near_dedup,
    audit_source_runtime,
    build_candidate_records,
    generation_schedule,
    load_canary_plan,
    validate_generated_records,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-canary-run-v0.1.json"


def _record(record_id: str, split: str, prompt: str) -> dict:
    return {
        "messages": [
            {"content": prompt, "role": "user"},
            {"content": "The answer is 1.", "role": "assistant"},
        ],
        "record_id": record_id,
        "split": split,
    }


def _generated_contract_records() -> list[dict]:
    records = []
    for shard in generation_schedule():
        for index in range(shard["count"]):
            seed = derive_seed(shard["namespace"], shard["module_file"], index)
            records.append(
                {
                    "answer": str(seed),
                    "module": f"{shard['top_module']}__unit",
                    "question": f"Compute unique value {seed} for this bounded test.",
                    "seed": seed,
                    "split_namespace": shard["namespace"],
                }
            )
    return records


def test_generation_schedule_is_exact_and_balanced() -> None:
    schedule = generation_schedule()
    assert len(schedule) == 16
    assert sum(item["count"] for item in schedule) == EXPECTED_GENERATED_RECORDS
    assert sum(SPLIT_MODULE_COUNTS["train"].values()) == EXPECTED_SPLIT_COUNTS["train"]
    assert sum(SPLIT_MODULE_COUNTS["validation"].values()) == (
        EXPECTED_SPLIT_COUNTS["validation"]
    )


def test_generated_record_validation_enforces_exact_seed_contract() -> None:
    records = _generated_contract_records()
    metrics = validate_generated_records(records)
    assert metrics["generated_records"] == EXPECTED_GENERATED_RECORDS
    assert metrics["conflicting_answer_prompt_groups"] == 0
    assert metrics["different_seed_prompt_groups"] == 0
    assert metrics["different_seed_prompt_overlap"] == 0
    assert metrics["train_validation_prompt_overlap"] == 0
    assert metrics["train_validation_seed_overlap"] == 0


def test_generated_record_validation_rejects_seed_drift() -> None:
    records = _generated_contract_records()
    records[0]["seed"] += 1
    with pytest.raises(ValueError, match="seed derivation"):
        validate_generated_records(records)


def test_generated_record_validation_measures_safe_same_answer_duplicates() -> None:
    records = _generated_contract_records()
    records[1]["question"] = records[0]["question"]
    records[1]["answer"] = records[0]["answer"]
    metrics = validate_generated_records(records)
    assert metrics["different_seed_prompt_groups"] == 1
    assert metrics["different_seed_prompt_overlap"] == 1
    assert metrics["conflicting_answer_prompt_groups"] == 0


def test_generated_record_validation_rejects_conflicting_duplicate_answers() -> None:
    records = _generated_contract_records()
    records[1]["question"] = records[0]["question"]
    with pytest.raises(ValueError, match="conflicting answers"):
        validate_generated_records(records)


def test_candidate_projection_uses_only_question_and_answer() -> None:
    seed = derive_seed(
        SPLIT_NAMESPACES["train"], TOP_MODULE_FILES["arithmetic"], 0
    )
    records, attributions = build_candidate_records(
        [
            {
                "answer": "4",
                "module": "arithmetic__add_or_sub",
                "question": "What is 2 + 2?",
                "seed": seed,
                "split_namespace": SPLIT_NAMESPACES["train"],
            }
        ]
    )
    assert records[0]["messages"] == [
        {"content": "What is 2 + 2?", "role": "user"},
        {"content": "The answer is 4.", "role": "assistant"},
    ]
    assert attributions[0]["trace_fields_used_as_training_targets"] is False
    assert "question" not in attributions[0]
    assert "answer" not in attributions[0]


def test_near_dedup_preserves_validation_over_train() -> None:
    records = [
        _record("train-1", "train", "Find the sum of one two three four five six."),
        _record(
            "validation-1",
            "validation",
            "Find the sum of one two three four five six.",
        ),
    ]
    accepted, ledger, metrics = apply_prompt_near_dedup(records)
    assert [row["record_id"] for row in accepted] == ["validation-1"]
    assert metrics["cross_split_rejected"] == 1
    by_id = {row["record_id"]: row for row in ledger}
    assert by_id["train-1"]["duplicate_of"] == "validation-1"


def test_exact_source_static_audit_excludes_upstream_generation_cli() -> None:
    source = Path(SOURCE_ROOT)
    if not source.is_dir():
        pytest.skip("DeepMind exact source is not materialized")
    audit = audit_source_runtime(source)
    assert audit["source_objects"] == 30
    assert audit["python_source_files"] == 28
    assert audit["upstream_generate_cli_imported_by_adapter"] is False
    assert audit["rng_providers"] == ["numpy.random", "python.random"]


def test_exact_plan_loads_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind canary run plan is not materialized")
    plan = load_canary_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["limits"]["maximum_canary_generated_records"] == 2_000
    assert plan["authorization"]["production_sft_authorized"] is False
