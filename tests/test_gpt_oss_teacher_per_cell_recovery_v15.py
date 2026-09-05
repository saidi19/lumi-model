from __future__ import annotations

from pathlib import Path

import pytest

from lumi.posttraining_gpt_oss_teacher_per_cell_recovery_v15 import (
    EXACT_ASSISTANT_RESPONSE,
    EXACT_USER_PROMPT,
    LIMITS,
    RECORD_ID,
    _reference_replacement,
    assemble_replacement,
    load_predecessor,
    repair_manifest,
    validate_exact_fields,
)

ROOT = Path(__file__).resolve().parents[1]


def test_predecessor_and_target_membership_are_exact() -> None:
    rows, telemetry, rejections = load_predecessor(ROOT)
    assert len(rows) == 160
    assert len(telemetry) == 160
    assert len(rejections) == 9
    assert sum(row["record_id"] == RECORD_ID for row in rows) == 1
    assert sum(row["record_id"] == RECORD_ID for row in rejections) == 1


def test_reference_replacement_recovers_full_machine_gate() -> None:
    row, telemetry = _reference_replacement()
    rows, combined_telemetry, rejections, audit, owner_sample = assemble_replacement(ROOT, row, telemetry)
    assert len(rows) == 160
    assert sum(int(item["attempts"]) for item in combined_telemetry) == 168
    assert len(rejections) == 8
    assert all(audit["checks"].values())
    assert audit["per_cell_first_attempt_acceptance_rate"]["multilingual_bengali:english"] == 0.8
    assert RECORD_ID in {item["record_id"] for item in owner_sample}


def test_contract_is_exactly_one_call_and_zero_retries() -> None:
    assert LIMITS["maximum_inference_calls"] == 1
    assert LIMITS["maximum_runtime_retries"] == 0
    assert repair_manifest()["attempt"]["attempt_number"] == 1


def test_exact_reference_fields_pass() -> None:
    validate_exact_fields([
        {"role": "user", "content": EXACT_USER_PROMPT},
        {"role": "assistant", "content": EXACT_ASSISTANT_RESPONSE},
    ])


def test_field_deviation_fails_closed() -> None:
    with pytest.raises(ValueError, match="assistant_response is not exact"):
        validate_exact_fields([
            {"role": "user", "content": EXACT_USER_PROMPT},
            {"role": "assistant", "content": EXACT_ASSISTANT_RESPONSE + " Extra"},
        ])
