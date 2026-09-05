from __future__ import annotations

from lumi import posttraining_gpt_oss_scale_canary_checkpoint_failure_v2 as failure
from lumi import posttraining_gpt_oss_scale_canary_v2 as canary


def test_cell_summary_accounts_for_attempted_accepted_and_remaining() -> None:
    schedule = canary.build_schedule()
    selected = schedule[:3]
    attempts = [
        {"record_id": selected[0].record_id},
        {"record_id": selected[1].record_id},
    ]
    rows = [
        {
            "record_id": selected[0].record_id,
            "task": selected[0].task,
            "language_mode": selected[0].language_mode,
        }
    ]
    cells = failure._cell_summary(schedule, rows, attempts)
    key = f"{selected[0].task}/{selected[0].language_mode}"
    assert cells[key]["attempted"] >= 1
    assert cells[key]["accepted"] == 1
    assert cells[key]["maximum_final_accepted"] == (
        cells[key]["accepted"] + cells[key]["remaining"]
    )


def test_bound_checkpoint_is_mathematically_ineligible() -> None:
    assert 620 + (1_000 - 877) == 743
    assert 743 < failure.TARGET_ACCEPTED

