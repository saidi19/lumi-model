from __future__ import annotations

from pathlib import Path

from lumi import posttraining_first_party_scale_canary as scale
from lumi import posttraining_gpt_oss_deterministic_tool_repair_v6 as repair
from lumi import posttraining_gpt_oss_scale_canary_v2 as base

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-deterministic-tool-repair-v0.6.json"


def _slot(record_id: str):
    return next(slot for slot in repair.build_schedule() if slot.record_id == record_id)


def test_scope_and_execution_boundary_are_local_only() -> None:
    assert len(repair.build_schedule()) == 1_000
    assert repair.PREFIX_ROWS == 768
    assert repair.REPAIR_RECORDS == 216
    assert repair.FIXED_MACHINE_EXCLUSIONS == 14
    assert len(repair.OWNER_EXCLUSIONS) == 2
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    assert plan["execution_boundary"]["additional_inference_calls"] == 0
    assert plan["execution_boundary"]["additional_network_bytes"] == 0
    assert plan["execution_boundary"]["gpu_use"] == 0
    assert plan["execution_boundary"]["remote_actions"] == 0


def test_predecessor_failure_and_clean_projection_are_exact() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    repair.load_predecessor_failure(plan, ROOT)
    rows, telemetry, repair_ids, fixed_failed = repair.load_clean_projection(plan, ROOT)
    assert len(rows) == 768
    assert len(telemetry) == 768
    assert len(repair_ids) == 216
    assert len(fixed_failed) == 14


def test_duplicate_tool_facts_receive_different_deterministic_variants() -> None:
    first = _slot("lumi-scale-v01-train-tool_api-english-0058")
    second = _slot("lumi-scale-v01-train-tool_api-english-0083")
    assert first.tool_arguments == second.tool_arguments
    assert first.tool_result == second.tool_result
    assert repair.deterministic_tool_response(first) != repair.deterministic_tool_response(second)


def test_all_repair_rows_pass_surface_schema_language_and_literal_gates() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    _, _, repair_ids, _ = repair.load_clean_projection(plan, ROOT)
    for slot in repair.build_schedule():
        if slot.record_id not in repair_ids:
            continue
        row = repair.build_deterministic_row(slot)
        response = row["messages"][-1]["content"]
        assert row["record_id"] == slot.record_id
        assert row["schema_version"] == repair.ROW_SCHEMA
        assert row["source_id"] == repair.SOURCE_ID
        assert row["provenance"]["inference_calls"] == 0
        assert "tool_call" not in response
        assert "tool_result" not in response
        assert "[" not in response and "]" not in response


def test_full_candidate_projection_passes_all_frozen_gates() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    rows, telemetry, fixed_failed = repair.build_candidate(plan, ROOT)
    audit = repair._machine_audit(plan, ROOT, rows, telemetry, fixed_failed)  # noqa: SLF001
    assert len(rows) == 984
    assert len(telemetry) == 984
    assert len(repair._terminal_map(rows, telemetry, fixed_failed)) == 1_000  # noqa: SLF001
    assert audit["tokens"] == {
        "train_supervised": 75_439,
        "validation_supervised": 18_974,
    }
    assert audit["raw_tool_envelope_rows"] == 0
    assert audit["critical_factual_safety_or_tool_failures"] == 0
    assert all(audit["checks"].values())


def test_fixed_rejection_metadata_contains_no_raw_text() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    _, _, _, fixed_failed = repair.load_clean_projection(plan, ROOT)
    metadata = repair._rejection_metadata(fixed_failed)  # noqa: SLF001
    assert len(metadata) == 16
    assert all(item["retained_raw_text"] is False for item in metadata)
    assert not any("candidate" in item for item in metadata)


def test_owner_packet_is_exactly_five_rows_per_cell() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    rows, _, _ = repair.build_candidate(plan, ROOT)
    packet = scale._owner_sample(rows)  # noqa: SLF001
    assert len(packet) == 100
    counts: dict[tuple[str, str], int] = {}
    for row in packet:
        cell = (str(row["task"]), str(row["language_mode"]))
        counts[cell] = counts.get(cell, 0) + 1
    assert set(counts.values()) == {5}


def test_v0_5_failure_candidate_is_not_used_as_the_repair_prefix() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    v5_rows = base._read_jsonl(ROOT / plan["bindings"]["v0_5_candidate"]["path"])  # noqa: SLF001
    clean_rows, _, _, _ = repair.load_clean_projection(plan, ROOT)
    assert len(v5_rows) == 800
    assert len(clean_rows) == 768
    assert {row["record_id"] for row in clean_rows} <= {
        row["record_id"] for row in v5_rows
    }
