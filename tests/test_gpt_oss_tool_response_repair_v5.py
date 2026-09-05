from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi import posttraining_gpt_oss_scale_canary_v2 as base
from lumi import posttraining_gpt_oss_tool_response_repair_v5 as repair

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-tool-response-repair-v0.5.json"


def _slot(record_id: str):
    return next(item for item in repair.build_schedule() if item.record_id == record_id)


def test_schedule_and_repair_bounds_are_exact() -> None:
    assert len(repair.build_schedule()) == 1_000
    assert repair.PREFIX_ROWS == 768
    assert repair.REPAIR_RECORDS == 216
    assert repair.FIXED_MACHINE_EXCLUSIONS == 14
    assert len(repair.OWNER_EXCLUSIONS) == 2
    assert repair.MAX_CALLS == 260
    assert repair.MAX_RETRIES == 44


def test_bound_predecessor_projection_is_exact() -> None:
    plan = repair._load_plan(PLAN, ROOT)  # noqa: SLF001
    repair.load_owner_rejection(plan, ROOT)
    rows, telemetry, repair_ids, fixed_failed = repair.load_clean_prefix(plan, ROOT)
    assert len(rows) == 768
    assert len(telemetry) == 768
    assert len(repair_ids) == 216
    assert len(fixed_failed) == 14
    assert not any(repair.is_raw_tool_envelope_defect(row) for row in rows)


def test_owner_packet_contains_exactly_thirteen_bound_defects() -> None:
    plan = base._load_json(PLAN)  # noqa: SLF001
    packet = base._read_jsonl(  # noqa: SLF001
        ROOT / plan["bindings"]["predecessor_owner_packet"]["path"]
    )
    assert sum(repair.is_raw_tool_envelope_defect(row) for row in packet) == 13


def test_generation_prompt_uses_plain_tool_evidence_without_raw_keys() -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-english-0045")
    prompt = "\n".join(item["content"] for item in repair._generation_messages(slot))  # noqa: SLF001
    assert "The calculator evaluated 106 + 32" in prompt
    assert "The verified value is 138" in prompt
    assert "tool_call" not in prompt
    assert "tool_result" not in prompt
    assert '{"arguments"' not in prompt


def test_payload_remains_response_only() -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-english-0045")
    payload = repair.chat_completion_payload(slot, 1)
    schema = payload["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"assistant_response"}
    assert payload["max_tokens"] == 1_024


@pytest.mark.parametrize(
    "response",
    [
        'The verified result is {"tool_call": "calculator"}.',
        'The verified result is {"tool_result": "138"}.',
        "The verified result is [138].",
        "The verified result is shown in the JSON envelope.",
    ],
)
def test_tool_surface_rejects_raw_or_meta_envelope(response: str) -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-english-0045")
    with pytest.raises(ValueError, match="forbidden_raw_envelope_marker"):
        repair.validate_tool_response_surface(slot, response)


def test_tool_surface_accepts_natural_grounded_explanation() -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-english-0045")
    repair.validate_tool_response_surface(
        slot,
        "The verified result is 138. The calculator confirms that 106 plus 32 equals 138.",
    )


def test_parser_accepts_natural_verified_tool_response() -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-english-0045")
    assistant = (
        "The verified result is 138. "
        "The calculator confirms that 106 plus 32 equals 138."
    )
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"assistant_response": assistant}),
                },
            }
        ]
    }
    row = repair._parse_candidate(  # noqa: SLF001
        response,
        slot=slot,
        request_sha256="0" * 64,
        seed=1,
        attempt_number=1,
    )
    assert row["record_id"] == slot.record_id
    assert row["schema_version"] == repair.ROW_SCHEMA


def test_tool_surface_requires_language_anchor() -> None:
    slot = _slot("lumi-scale-v01-train-tool_api-banglish-0047")
    with pytest.raises(ValueError, match="natural_language_anchor_missing"):
        repair.validate_tool_response_surface(slot, "Result holo 1206.")
