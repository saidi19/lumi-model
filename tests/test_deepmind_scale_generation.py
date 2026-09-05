from __future__ import annotations

from pathlib import Path

from lumi.posttraining_deepmind_canary_run import TOP_MODULE_FILES
from lumi.posttraining_deepmind_scale_generation import (
    EXPECTED_GENERATED_RECORDS,
    EXPECTED_SPLIT_COUNTS,
    SPLIT_MODULE_COUNTS,
    SPLIT_NAMESPACES,
    TOP_MODULES,
    build_candidate_records,
    derive_identity_seed,
    derive_rng_seed,
    deterministic_owner_sample,
    generation_schedule,
    load_scale_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-deepmind-scale-generation-v0.1.json"


def test_scale_schedule_is_exact_balanced_and_contiguous() -> None:
    schedule = generation_schedule()
    assert len(schedule) == 16
    assert sum(item["count"] for item in schedule) == EXPECTED_GENERATED_RECORDS
    assert sum(SPLIT_MODULE_COUNTS["train"].values()) == EXPECTED_SPLIT_COUNTS["train"]
    assert sum(SPLIT_MODULE_COUNTS["validation"].values()) == (
        EXPECTED_SPLIT_COUNTS["validation"]
    )
    cursor = 0
    for shard in schedule:
        assert shard["global_offset"] == cursor
        cursor += shard["count"]
    assert cursor == EXPECTED_GENERATED_RECORDS


def test_rng_mapping_is_injective_across_the_full_authorized_run() -> None:
    seeds = {derive_rng_seed(index) for index in range(EXPECTED_GENERATED_RECORDS)}
    assert len(seeds) == EXPECTED_GENERATED_RECORDS
    train = {derive_rng_seed(index) for index in range(EXPECTED_SPLIT_COUNTS["train"])}
    validation = {
        derive_rng_seed(index)
        for index in range(
            EXPECTED_SPLIT_COUNTS["train"], EXPECTED_GENERATED_RECORDS
        )
    }
    assert train.isdisjoint(validation)


def test_identity_seed_binds_namespace_module_and_index() -> None:
    module = TOP_MODULE_FILES["arithmetic"]
    train = derive_identity_seed(SPLIT_NAMESPACES["train"], module, 0)
    validation = derive_identity_seed(SPLIT_NAMESPACES["validation"], module, 0)
    next_train = derive_identity_seed(SPLIT_NAMESPACES["train"], module, 1)
    assert len({train, validation, next_train}) == 3


def test_scale_candidate_projects_question_and_answer_only() -> None:
    module_file = TOP_MODULE_FILES["arithmetic"]
    seed = derive_identity_seed(SPLIT_NAMESPACES["train"], module_file, 0)
    records, attributions = build_candidate_records(
        [
            {
                "answer": "4",
                "module": "arithmetic__add_or_sub",
                "question": "What is 2 + 2?",
                "rng_seed": derive_rng_seed(0),
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


def test_owner_sample_covers_every_split_module_bucket_twice() -> None:
    records = []
    attributions = []
    for split in ("train", "validation"):
        for module in TOP_MODULES:
            for index in range(3):
                record_id = f"{split}-{module}-{index}"
                records.append(
                    {
                        "messages": [
                            {"content": f"Question {record_id}", "role": "user"},
                            {"content": "The answer is 1.", "role": "assistant"},
                        ],
                        "record_id": record_id,
                        "split": split,
                    }
                )
                attributions.append(
                    {"module": f"{module}__unit", "record_id": record_id}
                )
    sample = deterministic_owner_sample(records, attributions)
    assert len(sample) == 32
    counts = {}
    for row in sample:
        key = (row["split"], row["skill_id"])
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


def test_exact_scale_plan_loads_when_materialized() -> None:
    if not PLAN.exists():
        return
    plan = load_scale_plan(PLAN, repository_root=ROOT)
    assert plan["limits"]["maximum_generated_records"] == 250_000
    assert plan["authorization"]["gpu_authorized"] is False
